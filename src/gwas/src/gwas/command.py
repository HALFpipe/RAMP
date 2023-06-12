# -*- coding: utf-8 -*-
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from gwas.compression.arr import compression_methods
from gwas.eig import Eigendecomposition
from gwas.log import logger
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from gwas.score.job import JobCollection
from gwas.tri import calc_tri
from gwas.utils import chromosome_to_int
from gwas.vcf.base import VCFFile, calc_vcf


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
        eig = Eigendecomposition.from_files(
            *tri_paths, sw=self.sw, samples=vc.samples, chromosome=chromosome
        )
        return eig

    def get_variable_collection(self) -> VariableCollection:
        return VariableCollection.from_txt(
            self.phenotype_paths,
            self.covariate_paths,
            self.sw,
            samples=self.vcf_samples,
            missing_value_strategy=self.arguments.missing_value_strategy,
        )

    def split_by_missing_values(self) -> list[VariableCollection]:
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
        (_, missing_value_pattern_count) = missing_value_patterns.shape
        vc.free()

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

            vc = self.get_variable_collection()
            vc.subset_phenotypes(pattern_phenotypes)
            vc.subset_samples(pattern_samples)

            # Sanity check.
            if not vc.is_finite:
                raise ValueError(
                    f"Missing values remain in chunk {i}. This should not happen"
                )

            if self.arguments.add_principal_components > 0:
                eig = self.get_eigendecomposition("X", vc)  # All autosomes.

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

            variable_collections.append(vc)

        # Sort by number of phenotypes.
        variable_collections = [
            variable_collections[i] for i in np.argsort(-missing_value_pattern_counts)
        ]
        return variable_collections

    def __post_init__(self) -> None:
        logger.debug("Arguments are %s", pformat(vars(self.arguments)))

        # Convert command line arguments to `Path` objects.
        vcf_paths: list[Path] = [Path(p) for p in self.arguments.vcf]
        tri_paths: list[Path] = [Path(p) for p in self.arguments.tri]
        self.phenotype_paths: list[Path] = [Path(p) for p in self.arguments.phenotypes]
        self.covariate_paths: list[Path] = [Path(p) for p in self.arguments.covariates]

        # Load VCF file metadata and cache it.
        vcf_files = calc_vcf(vcf_paths, self.output_directory)
        self.vcf_by_chromosome = {
            vcf_file.chromosome: vcf_file for vcf_file in vcf_files
        }
        # Update samples to only include those that are in all VCF files.
        vcf_sample_set: set[str] = set.intersection(
            *(set(vcf_file.samples) for vcf_file in vcf_files)
        )
        for vcf_file in vcf_files:
            vcf_file.set_samples(vcf_sample_set)
        # Ensure that we have the samples in the correct order.
        self.vcf_samples = vcf_files[0].samples
        # Load or calculate triangularized genotype data.
        self.tri_paths_by_chromosome = calc_tri(
            self.chromosomes,
            self.vcf_by_chromosome,
            self.output_directory,
            self.sw,
            tri_paths,
            self.arguments.kinship_minor_allele_frequency_cutoff,
            self.arguments.kinship_r_squared_cutoff,
        )
        # Split into missing value chunks.
        self.variable_collections = self.split_by_missing_values()

    def run(self) -> None:
        for chromosome in tqdm(self.chromosomes, desc="chromosomes"):
            vcf_file = self.vcf_by_chromosome[chromosome]
            vcf_file.set_variants_from_cutoffs(
                minor_allele_frequency_cutoff=(
                    self.arguments.score_minor_allele_frequency_cutoff
                ),
                r_squared_cutoff=self.arguments.score_r_squared_cutoff,
            )
            # We need to run the score calculation for each variable collection.
            variable_collections = self.variable_collections.copy()
            # Split the variable collections into chunks that fit efficiently
            # into memory.
            chunks: list[list[VariableCollection]] = list()
            # Get available memory.
            itemsize = np.float64().itemsize
            available_size = self.sw.unallocated_size // itemsize
            # Loop over the variable collections.
            chunk_size: int = 0
            current_chunk: list[VariableCollection] = list()
            while len(variable_collections) > 0:
                vc = variable_collections.pop()
                current_chunk.append(vc)
                # Predict memory usage.
                chunk_size += len(self.vcf_samples) * vc.sample_count
                chunk_size += 2 * vc.sample_count * vc.phenotype_count
                if chunk_size / available_size > 0.5:
                    # We are using more than half of the available memory
                    # for just the model data. We need to split the chunk.
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
