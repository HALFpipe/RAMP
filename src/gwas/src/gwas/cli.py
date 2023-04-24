# -*- coding: utf-8 -*-
import logging
import multiprocessing as mp
import shelve
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from gwas.eig import Eigendecomposition
from gwas.log import logger, setup_logging
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from gwas.score import calc_score
from gwas.tri import calc_tri
from gwas.utils import Pool, chromosome_to_int
from gwas.var import NullModelCollection
from gwas.vcf import VCFFile


def main() -> None:
    mp.set_start_method("spawn")

    argument_parser = ArgumentParser()

    argument_parser.add_argument("--vcf", nargs="+", required=True)
    argument_parser.add_argument("--tri", nargs="+", required=False, default=list())
    argument_parser.add_argument("--phenotypes", nargs="+", required=True)
    argument_parser.add_argument("--covariates", nargs="+", required=True)
    argument_parser.add_argument("--output-directory", required=True)
    argument_parser.add_argument(
        "--minor-allele-frequency-cutoff",
        "--maf",
        required=False,
        type=float,
        default=0.05,
    )
    argument_parser.add_argument(
        "--method", required=False, choices=NullModelCollection.methods, default="ml"
    )
    argument_parser.add_argument(
        "--add-principal-components",
        required=False,
        type=int,
        default=0,
    )
    argument_parser.add_argument(
        "--missing-value-strategy",
        required=False,
        choices=["complete_samples", "listwise_deletion"],
        default="listwise_deletion",
    )
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--debug", action="store_true", default=False)

    argument_parser.add_argument("--mem-gb", type=float)

    arguments = argument_parser.parse_args()

    setup_logging(level=arguments.log_level)
    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    size: int | None = None
    if arguments.mem_gb is not None:
        size = int(arguments.mem_gb * 2**30)

    with (
        SharedWorkspace.create(size=size) as sw,
        shelve.open(str(output_directory / "cache")) as cache,
    ):
        try:
            GwasCommand(arguments, output_directory, sw, cache).run()
        except Exception as e:
            logger.exception("Exception: %s", e, exc_info=True)
            if arguments.debug:
                import pdb

                pdb.post_mortem()


@dataclass
class GwasCommand:
    arguments: Namespace
    output_directory: Path

    sw: SharedWorkspace
    cache: shelve.Shelf

    phenotype_paths: list[Path] = field(init=False)
    covariate_paths: list[Path] = field(init=False)

    chromosomes: Sequence[int | str] = field(init=False)
    vcf_samples: list[str] = field(init=False)
    vcf_by_chromosome: Mapping[int | str, VCFFile] = field(init=False)
    tri_paths_by_chromosome: Mapping[int | str, Path] = field(init=False)

    def get_variable_collection(self) -> VariableCollection:
        return VariableCollection.from_txt(
            self.phenotype_paths,
            self.covariate_paths,
            self.sw,
            samples=self.vcf_samples,
            missing_value_strategy=self.arguments.missing_value_strategy,
        )

    def __post_init__(self) -> None:
        logger.info("Arguments are %s", self.arguments)

        # Convert command line arguments to `Path` objects.
        vcf_paths: list[Path] = [Path(p) for p in self.arguments.vcf]
        tri_paths: list[Path] = [Path(p) for p in self.arguments.tri]
        self.phenotype_paths: list[Path] = [Path(p) for p in self.arguments.phenotypes]
        self.covariate_paths: list[Path] = [Path(p) for p in self.arguments.covariates]

        minor_allele_frequency_cutoff = float(
            self.arguments.minor_allele_frequency_cutoff
        )

        # Load VCF file metadata and cache it.
        if "vcf_files" not in self.cache:
            with Pool(processes=min(cpu_count(), len(vcf_paths))) as pool:
                self.cache["vcf_files"] = list(pool.map(VCFFile, vcf_paths))
        vcf_files = self.cache["vcf_files"]
        self.vcf_by_chromosome = {
            vcf_file.chromosome: vcf_file for vcf_file in vcf_files
        }
        self.chromosomes = sorted(self.vcf_by_chromosome.keys(), key=chromosome_to_int)
        self.vcf_samples = sorted(
            set.intersection(*(set(vcf_file.samples) for vcf_file in vcf_files))
        )
        for vcf_file in vcf_files:
            vcf_file.update_samples(self.vcf_samples)

        # Load or calculate triangularized chromosomes.
        self.tri_paths_by_chromosome = calc_tri(
            self.chromosomes,
            self.vcf_by_chromosome,
            self.output_directory,
            self.sw,
            tri_paths,
            minor_allele_frequency_cutoff,
        )

    def run(self) -> None:
        # Load phenotype and covariate data for all samples that have genetic data and
        # count missing values.
        vc = self.get_variable_collection()
        phenotype_names = vc.phenotype_names
        samples = vc.samples
        (
            missing_value_patterns,
            missing_value_pattern_indices,
            missing_value_pattern_counts,
        ) = np.unique(
            np.isfinite(vc.phenotypes.to_numpy()),
            axis=1,
            return_inverse=True,
            return_counts=True,
        )
        missing_value_pattern_count = len(missing_value_patterns)
        vc.free()

        variable_collections: list[VariableCollection] = list()
        for i in range(missing_value_pattern_count):
            missing_value_pattern = missing_value_patterns[:, i]
            chunk_samples = [
                sample for sample, c in zip(samples, missing_value_pattern) if c
            ]
            chunk_phenotypes = [
                p
                for p, j in zip(phenotype_names, missing_value_pattern_indices)
                if i == j
            ]

            vc = self.get_variable_collection()
            vc.subset_phenotypes(chunk_phenotypes)
            vc.subset_samples(chunk_samples)

            if not vc.is_finite:
                raise ValueError(f"Missing values in chunk {i}")

            variable_collections.append(vc)

        for i in tqdm(np.argsort(-missing_value_pattern_counts), desc="chunks"):
            vc = variable_collections[i]
            self.map_phenotype_chunk(vc, suffix=f"chunk_{i + 1}")
            vc.free()

    def map_phenotype_chunk(
        self,
        vc: VariableCollection,
        suffix: str = "",
    ) -> None:
        if len(suffix) > 0:
            suffix = f".{suffix}"

        def get_eig(chromosome: int | str) -> Eigendecomposition:
            # Leave out current chromosome from calculation.
            other_chromosomes = sorted(
                set(self.chromosomes) - {chromosome, "X"}, key=chromosome_to_int
            )
            tri_paths = [self.tri_paths_by_chromosome[c] for c in other_chromosomes]
            # Calculate eigendecomposition and free tris.
            eig = Eigendecomposition.from_files(
                *tri_paths, sw=self.sw, samples=vc.samples, chromosome=chromosome
            )
            return eig

        if self.arguments.add_principal_components > 0:
            import numpy as np

            eig = get_eig("X")  # All autosomes.

            k = self.arguments.add_principal_components
            pc_array = eig.eigenvectors[:, :k]
            pc_names = [f"principal_component_{i + 1:02d}" for i in range(k)]

            # Merge the existing array with the principal components.
            covariates = np.hstack([vc.covariates.to_numpy(), pc_array])

            # Clean up.
            eig.free()
            vc.covariates.free()
            self.sw.squash()

            vc.covariate_names.extend(pc_names)
            vc.covariates = SharedArray.from_numpy(
                covariates, self.sw, prefix="covariates"
            )

        for chromosome in tqdm(self.chromosomes, desc="chromosomes"):
            eig = get_eig(chromosome)

            nm = NullModelCollection.from_eig(eig, vc, method=self.arguments.method)

            vcf_file = self.vcf_by_chromosome[chromosome]
            vcf_file.update_samples(vc.samples)

            score_file_name = f"chr{chromosome}{suffix}.score.txt.zst"
            score_path = self.output_directory / score_file_name

            calc_score(
                vcf_file,
                vc,
                nm,
                eig,
                self.sw,
                score_path,
            )
            nm.free()
            eig.free()
