# -*- coding: utf-8 -*-
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from ..compression.arr.base import compression_methods
from ..eig import Eigendecomposition
from ..log import logger
from ..mean import calc_mean
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..pheno import VariableCollection
from ..score.job import JobCollection
from ..tri.calc import calc_tri
from ..utils import chromosome_to_int, parse_chromosome
from ..vcf.base import VCFFile, calc_vcf


@dataclass
class GwasCommand:
    arguments: Namespace
    output_directory: Path

    sw: SharedWorkspace

    phenotype_paths: list[Path] = field(init=False)
    covariate_paths: list[Path] = field(init=False)

    vcf_samples: list[str] = field(init=False)
    vcf_by_chromosome: Mapping[int | str, VCFFile] = field(init=False)
    tri_paths_by_chromosome: Mapping[int | str, Path] = field(init=False)

    variable_collections: list[VariableCollection] = field(init=False)

    @property
    def chromosomes(self) -> Sequence[int | str]:
        return sorted(self.vcf_by_chromosome.keys(), key=chromosome_to_int)

    def get_eigendecomposition(
        self, chromosome: int | str, vc: VariableCollection
    ) -> Eigendecomposition:
        # Leave out current chromosome from calculation.
        other_chromosomes = sorted(
            set(self.chromosomes) - {chromosome, "X"}, key=chromosome_to_int
        )
        tri_paths = [self.tri_paths_by_chromosome[c] for c in other_chromosomes]
        # Calculate eigendecomposition and free tris.
        eig = Eigendecomposition.from_files(*tri_paths, sw=self.sw, samples=vc.samples)
        return eig

    def get_variable_collection(self) -> VariableCollection:
        return VariableCollection.from_txt(
            self.phenotype_paths,
            self.covariate_paths,
            self.sw,
            samples=self.vcf_samples,
            missing_value_strategy=self.arguments.missing_value_strategy,
        )

    @staticmethod
    def split_by_missing_values(
        base_variable_collection: VariableCollection,
        add_principal_components: int = 0,
        get_eigendecomposition: Callable[
            [int | str, VariableCollection], Eigendecomposition
        ]
        | None = None,
    ) -> list[VariableCollection]:
        # Load phenotype and covariate data for all samples that have genetic data and
        # count missing values.
        phenotype_names = base_variable_collection.phenotype_names
        samples = base_variable_collection.samples
        (
            missing_value_patterns,
            missing_value_pattern_indices,
        ) = np.unique(
            np.isfinite(base_variable_collection.phenotypes.to_numpy()),
            axis=1,
            return_inverse=True,
        )
        (_, missing_value_pattern_count) = missing_value_patterns.shape

        variable_collections: list[VariableCollection] = list()
        for i in tqdm(
            range(missing_value_pattern_count), unit="chunks", desc="loading phenotypes"
        ):
            missing_value_pattern = missing_value_patterns[:, i]
            pattern_samples: list[str] = [
                sample
                for sample, has_data in zip(samples, missing_value_pattern)
                if has_data
            ]
            pattern_phenotypes: list[str] = sorted(
                [
                    phenotype_name
                    for phenotype_name, j in zip(
                        phenotype_names, missing_value_pattern_indices
                    )
                    if i == j
                ]
            )
            if len(pattern_samples) == 0:
                logger.warning(
                    f"Phenotypes {pattern_phenotypes} are missing for all samples. "
                    "Skipping"
                )
                continue
            if len(pattern_phenotypes) == 0:
                raise RuntimeError(
                    f"No phenotypes in chunk {i}. This should not happen"
                )

            variable_collection = base_variable_collection.copy()
            variable_collection.subset_phenotypes(pattern_phenotypes)
            variable_collection.subset_samples(pattern_samples)
            if not variable_collection.is_finite:
                # Sanity check.
                raise RuntimeError(
                    f"Missing values remain in chunk {i}. This should not happen"
                )

            if add_principal_components > 0:
                if get_eigendecomposition is None:
                    raise RuntimeError(
                        "Cannot add principal components without eigendecomposition "
                        "function"
                    )
                eig = get_eigendecomposition("X", variable_collection)  # All autosomes

                pc_array = eig.eigenvectors[:, :add_principal_components]
                pc_names = [
                    f"principal_component_{i + 1:02d}"
                    for i in range(add_principal_components)
                ]

                # Merge the existing array with the principal components
                covariates = np.hstack(
                    [variable_collection.covariates.to_numpy(), pc_array]
                )

                # Clean up.
                eig.free()
                variable_collection.covariates.free()
                variable_collection.sw.squash()

                variable_collection.covariate_names.extend(pc_names)
                variable_collection.covariates = SharedArray.from_numpy(
                    covariates, variable_collection.sw, prefix="covariates"
                )

            variable_collections.append(variable_collection)

        # Sort by number of phenotypes
        variable_collections.sort(key=lambda vc: -vc.phenotype_count)

        # Set names
        for i, vc in enumerate(variable_collections):
            vc.name = f"variableCollection-{i + 1:02d}"

        return variable_collections

    def __post_init__(self) -> None:
        logger.debug("Arguments are %s", pformat(vars(self.arguments)))

        # Convert command line arguments to `Path` objects
        vcf_paths: list[Path] = [Path(p) for p in self.arguments.vcf]
        tri_paths: list[Path] = [Path(p) for p in self.arguments.tri]
        self.phenotype_paths: list[Path] = [Path(p) for p in self.arguments.phenotypes]
        self.covariate_paths: list[Path] = [Path(p) for p in self.arguments.covariates]

        # Load VCF file metadata and cache it
        vcf_files = calc_vcf(vcf_paths, self.output_directory)
        self.set_vcf_files(vcf_files)
        # Update samples to only include those that are in all VCF files
        vcf_sample_set: set[str] = set.intersection(
            *(set(vcf_file.samples) for vcf_file in vcf_files)
        )
        for vcf_file in vcf_files:
            vcf_file.set_samples(vcf_sample_set)
        # Ensure that we have the samples in the correct order
        self.vcf_samples = vcf_files[0].samples
        # Load or calculate triangularized genotype data
        self.tri_paths_by_chromosome = calc_tri(
            self.chromosomes,
            self.vcf_by_chromosome,
            self.output_directory,
            self.sw,
            tri_paths,
            self.arguments.kinship_minor_allele_frequency_cutoff,
            self.arguments.kinship_r_squared_cutoff,
        )
        # Split into missing value chunks
        base_variable_collection = self.get_variable_collection()
        self.variable_collections = self.split_by_missing_values(
            base_variable_collection,
            self.arguments.add_principal_components,
            self.get_eigendecomposition,
        )
        if len(self.variable_collections) == 0:
            raise ValueError(
                "No phenotypes to analyze. Please check if the sample IDs "
                "match between your phenotype/covariate files and the VCF files."
            )
        base_variable_collection.free()
        self.set_vcf_files(vcf_files)

    def set_vcf_files(self, vcf_files: list[VCFFile]) -> None:
        self.vcf_by_chromosome = {
            vcf_file.chromosome: vcf_file for vcf_file in vcf_files
        }

    def run(self) -> None:
        chromosomes: Iterable[int | str] = self.chromosomes
        if self.arguments.chromosome is not None:
            chromosomes = map(parse_chromosome, self.arguments.chromosome)
        chromosomes = sorted(chromosomes, key=chromosome_to_int)
        for chromosome in tqdm(chromosomes, desc="chromosomes"):
            vcf_file = self.vcf_by_chromosome[chromosome]
            # Update the VCF file allele frequencies based on variable collections
            if calc_mean(
                vcf_file,
                self.variable_collections,
            ):
                vcf_file.save_to_cache(self.output_directory)
            vcf_file.set_variants_from_cutoffs(
                minor_allele_frequency_cutoff=(
                    self.arguments.score_minor_allele_frequency_cutoff
                ),
                r_squared_cutoff=self.arguments.score_r_squared_cutoff,
            )
            # We need to run the score calculation for each variable collection.
            variable_collections = self.variable_collections.copy()
            # Split the variable collections into chunks that fit efficiently
            # into memory
            chunks: list[list[VariableCollection]] = list()
            # Get available memory
            itemsize = np.float64().itemsize
            available_size = self.sw.unallocated_size // itemsize
            # Loop over the variable collections
            chunk_size: int = 0
            current_chunk: list[VariableCollection] = list()
            while len(variable_collections) > 0:
                vc = variable_collections.pop()
                current_chunk.append(vc)
                # Predict memory usage
                chunk_size += len(self.vcf_samples) * vc.sample_count
                chunk_size += 2 * vc.sample_count * vc.phenotype_count
                if chunk_size / available_size > 0.5:
                    # We are using more than half of the available memory
                    # for just the model data. We need to split the chunk
                    chunks.append(current_chunk)
                    current_chunk = list()
            chunks.append(current_chunk)

            job_collection = JobCollection(
                vcf_file,
                self.get_eigendecomposition,
                self.arguments.null_model_method,
                self.output_directory,
                compression_methods[self.arguments.compression_method],
                self.arguments.num_threads,
                chunks,
            )
            job_collection.dump()
            job_collection.run()
