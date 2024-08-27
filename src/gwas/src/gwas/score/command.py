from argparse import Namespace
from dataclasses import dataclass, field
from pprint import pformat
from typing import Iterable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
from upath import UPath

from ..compression.arr.base import (
    ParquetCompressionMethod,
    TextCompressionMethod,
    compression_methods,
)
from ..covar import calc_covariance
from ..log import logger
from ..mean import calc_mean
from ..mem.wkspace import SharedWorkspace
from ..pheno import VariableCollection
from ..score.job import JobCollection
from ..tri.calc import calc_tri
from ..utils.genetics import chromosome_to_int, parse_chromosome
from ..vcf.base import VCFFile, calc_vcf


def subset_variable_collection(
    base_variable_collection: VariableCollection,
    pattern_samples: list[str],
    pattern_phenotypes: list[str],
    **kwargs,
) -> VariableCollection:
    variable_collection = base_variable_collection.copy(
        samples=pattern_samples,
        phenotype_names=pattern_phenotypes,
        **kwargs,
    )
    variable_collection.remove_zero_variance_covariates()
    if not variable_collection.is_finite:
        # Sanity check.
        raise RuntimeError(
            f"Missing values remain in chunk {variable_collection.name}. This should "
            "not happen"
        )
    return variable_collection


def split_by_missing_values(
    base_variable_collection: VariableCollection, prefix: str
) -> list[VariableCollection]:
    phenotype_names = base_variable_collection.phenotype_names
    samples = base_variable_collection.samples

    # Load phenotype and covariate data for all samples that have genetic data and
    # count missing values.
    (
        missing_value_patterns,
        missing_value_pattern_indices,
    ) = np.unique(
        np.isfinite(base_variable_collection.phenotypes),
        axis=1,
        return_inverse=True,
    )
    missing_value_pattern_indices = missing_value_pattern_indices.ravel()
    (_, missing_value_pattern_count) = missing_value_patterns.shape

    variable_collections: list[VariableCollection] = list()
    for i in tqdm(
        range(missing_value_pattern_count), unit="chunks", desc="loading phenotypes"
    ):
        missing_value_pattern = missing_value_patterns[:, i]
        pattern_samples: list[str] = [
            sample
            for sample, has_data in zip(samples, missing_value_pattern, strict=True)
            if has_data
        ]
        pattern_phenotypes: list[str] = sorted(
            [
                phenotype_name
                for phenotype_name, j in zip(
                    phenotype_names, missing_value_pattern_indices, strict=True
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
            raise RuntimeError(f"No phenotypes in chunk {i}. This should not happen")
        variable_collection = subset_variable_collection(
            base_variable_collection,
            pattern_samples,
            pattern_phenotypes,
            name=f"{prefix}-{i:02d}",
        )
        variable_collections.append(variable_collection)

    return variable_collections


@dataclass
class GwasCommand:
    arguments: Namespace
    output_directory: UPath

    sw: SharedWorkspace

    variable_collection_prefix: str = "variableCollection"

    phenotype_paths: list[UPath] = field(init=False)
    covariate_paths: list[UPath] = field(init=False)

    vcf_samples: list[str] = field(init=False)
    vcf_by_chromosome: Mapping[int | str, VCFFile] = field(init=False)
    tri_paths_by_chromosome: Mapping[int | str, UPath] = field(init=False)

    @property
    def chromosomes(self) -> Sequence[int | str]:
        return sorted(self.vcf_by_chromosome.keys(), key=chromosome_to_int)

    @property
    def selected_chromosomes(self) -> Sequence[int | str]:
        chromosomes: Iterable[int | str] = self.chromosomes
        if self.arguments.chromosome is not None:
            chromosomes = map(parse_chromosome, self.arguments.chromosome)
        return sorted(chromosomes, key=chromosome_to_int)

    def get_variable_collection(self) -> VariableCollection:
        return VariableCollection.from_txt(
            self.phenotype_paths,
            self.covariate_paths,
            self.sw,
            samples=self.vcf_samples,
            missing_value_strategy=self.arguments.missing_value_strategy,
        )

    def update_allele_frequencies(
        self, variable_collections: list[VariableCollection]
    ) -> None:
        # Update the VCF file allele frequencies based on variable collections
        for chromosome in tqdm(
            self.selected_chromosomes, desc="calculating allele frequencies"
        ):
            vcf_file = self.vcf_by_chromosome[chromosome]
            if calc_mean(
                vcf_file,
                variable_collections,
                self.arguments.num_threads,
            ):
                vcf_file.save_to_cache(self.output_directory, self.arguments.num_threads)

    def setup_variable_collections(self) -> list[VariableCollection]:
        # Convert command line arguments to `UPath` objects
        vcf_paths: list[UPath] = [UPath(p) for p in self.arguments.vcf]
        tri_paths: list[UPath] = [UPath(p) for p in self.arguments.tri]
        self.phenotype_paths: list[UPath] = [UPath(p) for p in self.arguments.phenotypes]
        self.covariate_paths: list[UPath] = [UPath(p) for p in self.arguments.covariates]

        # Load VCF file metadata and cache it
        vcf_files = calc_vcf(
            vcf_paths,
            self.output_directory,
            num_threads=self.arguments.num_threads,
            sw=self.sw,
            engine=self.arguments.vcf_engine,
        )
        self.set_vcf_files(vcf_files)

        # Update samples to only include those that are in all VCF files
        base_samples: set[str] = set.intersection(
            *(set(vcf_file.samples) for vcf_file in vcf_files)
        )
        for vcf_file in vcf_files:
            vcf_file.set_samples(base_samples)

        # Ensure that we have the samples in the correct order
        self.vcf_samples = vcf_files[0].samples

        # Load phenotypes and covariates
        base_variable_collection = self.get_variable_collection()
        with threadpool_limits(limits=self.arguments.num_threads):
            calc_covariance(
                base_variable_collection,
                self.output_directory / "covariance",
                compression_methods[self.arguments.compression_method],
                num_threads=self.arguments.num_threads,
            )
        # Split into missing value chunks
        variable_collections: list[VariableCollection] = split_by_missing_values(
            base_variable_collection, self.variable_collection_prefix
        )
        base_variable_collection.free()
        self.sw.squash()
        if len(variable_collections) == 0:
            raise ValueError(
                "No phenotypes to analyze. Please check if the sample IDs "
                "match between your phenotype/covariate files and the VCF files."
            )
        # Sort by number of phenotypes
        variable_collections.sort(key=lambda vc: -vc.phenotype_count)

        self.update_allele_frequencies(variable_collections)

        # Load or calculate triangularized genotype data
        for vcf_file in vcf_files:  # Use all available samples
            vcf_file.set_samples(set(vcf_file.vcf_samples))
        self.tri_paths_by_chromosome = calc_tri(
            self.chromosomes,
            self.vcf_by_chromosome,
            self.output_directory,
            self.sw,
            tri_paths,
            self.arguments.kinship_minor_allele_frequency_cutoff,
            self.arguments.kinship_r_squared_cutoff,
            num_threads=self.arguments.num_threads,
        )

        self.set_vcf_files(vcf_files)

        return variable_collections

    def set_vcf_files(self, vcf_files: list[VCFFile]) -> None:
        self.vcf_by_chromosome = {
            vcf_file.chromosome: vcf_file for vcf_file in vcf_files
        }

    def split_into_chunks(
        self, variable_collections: list[VariableCollection]
    ) -> list[list[VariableCollection]]:
        chunks: list[list[VariableCollection]] = list()

        # Get available memory
        itemsize = np.float64().itemsize
        available_size = self.sw.unallocated_size // itemsize

        # Loop over the variable collections
        chunk_size: int = 0
        current_chunk: list[VariableCollection] = list()
        while len(variable_collections) > 0:
            variable_collection = variable_collections.pop()
            current_chunk.append(variable_collection)

            # Predict memory usage
            chunk_size += len(self.vcf_samples) * variable_collection.sample_count
            chunk_size += (
                2
                * variable_collection.sample_count
                * variable_collection.phenotype_count
            )
            if chunk_size / available_size > 0.5:
                # We are using more than half of the available memory
                # for just the input data, so we split the chunk
                chunks.append(current_chunk)
                current_chunk = list()

        chunks.append(current_chunk)
        logger.debug(f"Split variable collections into {len(chunks)} chunks")
        return chunks

    def run_chromosome(
        self, chromosome: int | str, variable_collections: list[VariableCollection]
    ) -> None:
        vcf_file = self.vcf_by_chromosome[chromosome]
        vcf_file.set_variants_from_cutoffs(
            minor_allele_frequency_cutoff=(
                self.arguments.score_minor_allele_frequency_cutoff
            ),
            r_squared_cutoff=self.arguments.score_r_squared_cutoff,
        )

        # We need to run the score calculation for each variable collection
        # Split the variable collections into chunks that fit efficiently
        # into memory
        chunks = self.split_into_chunks(variable_collections)

        compression_method = compression_methods[self.arguments.compression_method]

        if isinstance(
            compression_method, (TextCompressionMethod, ParquetCompressionMethod)
        ):
            if len(chunks) > 1:
                raise ValueError(
                    "Cannot use text compression method for multiple chunks. "
                    "Please use a different compression method or increase "
                    "available memory"
                )

        self.job_collection = JobCollection(
            vcf_file=vcf_file,
            chromosomes=self.chromosomes,
            tri_paths_by_chromosome=self.tri_paths_by_chromosome,
            null_model_method=self.arguments.null_model_method,
            output_directory=self.output_directory,
            compression_method=compression_method,
            num_threads=self.arguments.num_threads,
            jax_trace=self.arguments.jax_trace,
            variable_collection_chunks=chunks,
        )
        self.job_collection.dump()
        self.job_collection.run()

    def run(self) -> None:
        logger.debug("Arguments are %s", pformat(vars(self.arguments)))
        variable_collections = self.setup_variable_collections()

        for chromosome in tqdm(
            self.selected_chromosomes,
            unit="chromosomes",
            desc="processing chromosomes",
        ):
            self.run_chromosome(chromosome, variable_collections.copy())

        # Clean up
        for variable_collection in variable_collections:
            variable_collection.free()

    def free(self):
        for vcf_file in self.vcf_by_chromosome.values():
            vcf_file.free()
