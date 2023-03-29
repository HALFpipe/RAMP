# -*- coding: utf-8 -*-
import logging
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path

from gwas.eig import Eigendecomposition
from gwas.log import logger
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from gwas.score import calc_score
from gwas.tri import Triangular
from gwas.utils import chromosome_to_int
from gwas.var import NullModelCollection
from gwas.vcf import VCFFile


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format=(
            "%(asctime)s [%(levelname)8s] %(funcName)s: "
            "%(message)s (%(filename)s:%(lineno)s)"
        ),
    )

    argument_parser = ArgumentParser()

    argument_parser.add_argument("--vcf-paths", nargs="+", required=True)
    argument_parser.add_argument(
        "--tri-paths", nargs="+", required=False, default=list()
    )
    argument_parser.add_argument(
        "--eig-paths", nargs="+", required=False, default=list()
    )
    argument_parser.add_argument("--phenotype-path", required=True)
    argument_parser.add_argument("--covariate-path", required=True)
    argument_parser.add_argument("--out-path", required=True)
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

    arguments = argument_parser.parse_args()

    # Convert command line arguments to `Path` objects.
    vcf_paths: list[Path] = [Path(p) for p in arguments.vcf_paths]
    tri_paths: list[Path] = [Path(p) for p in arguments.tri_paths]
    eig_paths: list[Path] = [Path(p) for p in arguments.eig_paths]
    phenotype_path = Path(arguments.phenotype_path)
    covariate_path = Path(arguments.covariate_path)
    out_path = Path(arguments.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    method = arguments.method
    minor_allele_frequency_cutoff = float(arguments.minor_allele_frequency_cutoff)

    # Allocate shared workspace.
    with SharedWorkspace.create() as sw:
        # Load VCF file metadata.
        with Pool(processes=min(cpu_count(), len(vcf_paths))) as pool:
            vcf_files = list(pool.map(VCFFile, vcf_paths))
        vcf_by_chromosome = {vcf_file.chromosome: vcf_file for vcf_file in vcf_files}
        samples = sorted(
            set.intersection(*(set(vcf_file.samples) for vcf_file in vcf_files))
        )

        # Load phenotype and covariate data.
        vc = VariableCollection.from_txt(
            phenotype_path, covariate_path, sw, samples=samples
        )
        for vcf_file in vcf_files:
            vcf_file.update_samples(vc.samples)
        logger.info(f"Found {len(vc.samples)} samples across input files.")

        # Sort `--tri-paths` command line arguments into dictionary.
        tri_paths_by_chromosome: dict[int | str, Path] = dict()
        for tri_path in tri_paths:
            tri = Triangular.from_file(tri_path, sw)
            tri_paths_by_chromosome[tri.chromosome] = tri_path
            tri.free()

        # Create accessor.
        def get_tri(chromosome: int | str) -> Triangular:
            tri_path: Path | None = None
            if chromosome in tri_paths_by_chromosome:
                tri_path = tri_paths_by_chromosome[chromosome]
            if tri_path is None:
                tri_path = out_path / Triangular.get_file_name(chromosome)

            # Check if triangularized file already exists, and load it.
            if tri_path.is_file():
                return Triangular.from_file(tri_path, sw)

            # Triangularize VCF file.
            vcf_file = vcf_by_chromosome[chromosome]
            tri = Triangular.from_vcf(
                vcf_file,
                sw,
                minor_allele_frequency_cutoff,
            )
            if tri is None:
                raise ValueError(f"Could not triangularize {vcf_file.file_path}")

            tri_paths_by_chromosome[chromosome] = tri.to_file(tri_path)
            return tri

        def run(chromosome: int | str, eig: Eigendecomposition | None) -> None:
            eig_path = out_path / Eigendecomposition.get_file_name(chromosome)
            if eig is None:
                if eig_path.is_file():
                    eig = Eigendecomposition.from_file(eig_path, sw)
                else:
                    # Leave out current chromosome from calculation.
                    other_chromosomes = set(chromosomes) - {chromosome}
                    tris = [get_tri(c) for c in other_chromosomes]
                    # Calculate eigendecomposition and free tris.
                    eig = Eigendecomposition.from_tri(*tris, chromosome=chromosome)
                    eig.to_file(eig_path)

            nm = NullModelCollection.from_eig(eig, vc, method=method)

            vcf_file = vcf_by_chromosome[chromosome]
            score_path = out_path / f"chr{chromosome}.score.txt.zst"
            calc_score(
                vcf_file,
                vc,
                nm,
                eig,
                sw,
                score_path,
            )

        chromosomes = sorted(vcf_by_chromosome.keys(), key=chromosome_to_int)
        tasks = set(chromosomes)

        for eig_path in eig_paths:
            if not eig_path.is_file():
                continue
            eig = Eigendecomposition.from_file(eig_path, sw)
            chromosome = eig.chromosome
            run(chromosome, eig)
            tasks.remove(chromosome)

        for task in tasks:
            run(task, None)
