# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Self

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from ..compression.arr.base import CompressionMethod, FileArray
from ..compression.pipe import CompressedTextReader, CompressedTextWriter
from ..eig import Eigendecomposition
from ..log import logger
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..null_model.base import NullModelCollection
from ..pheno import VariableCollection, VariableSummary
from ..utils import parse_obj_as
from ..vcf.base import VCFFile
from .run import calc_score


@dataclass
class Job:
    vc: VariableCollection
    nm: NullModelCollection
    eig: Eigendecomposition

    phenotype_indices: list[int]


@dataclass(frozen=True, slots=True)
class RegressionWeight:
    value: float
    standard_error: float


@dataclass
class CovariateSummary(VariableSummary):
    pass


@dataclass
class PhenotypeSummary(VariableSummary):
    genetic_variance: float | None = None
    error_variance: float | None = None
    heritability: float | None = None

    regression_weights: dict[str, RegressionWeight] | None = None


@dataclass
class JobSummary:
    status: Literal["pending", "null_model_complete", "score_complete"]

    sample_count: int

    phenotypes: dict[str, PhenotypeSummary]
    covariates: dict[str, VariableSummary]

    def put_null_model_collection(self, nm: NullModelCollection) -> None:
        self.status = "null_model_complete"
        for i, name in enumerate(self.phenotypes.keys()):
            self.phenotypes[name].genetic_variance = float(nm.genetic_variance[i])
            self.phenotypes[name].error_variance = float(nm.error_variance[i])
            self.phenotypes[name].heritability = float(nm.heritability[i])

            regression_weights = nm.regression_weights.to_numpy()
            standard_errors = nm.standard_errors.to_numpy()
            self.phenotypes[name].regression_weights = {
                name: RegressionWeight(float(weight), float(error))
                for name, weight, error in zip(
                    self.covariates.keys(),
                    regression_weights[i],
                    standard_errors[i],
                )
            }

    @classmethod
    def from_variable_collection(cls, vc: VariableCollection) -> Self:
        return cls(
            "pending",
            vc.sample_count,
            {
                name: PhenotypeSummary.from_array(vc.phenotypes.to_numpy()[:, i])
                for i, name in enumerate(vc.phenotype_names)
            },
            {
                name: CovariateSummary.from_array(vc.covariates.to_numpy()[:, i])
                for i, name in enumerate(vc.covariate_names)
            },
        )


@dataclass
class SummaryCollection:
    chunks: dict[str, list[JobSummary]]

    def validate(self, variable_collections: list[list[VariableCollection]]) -> None:
        for job_summary, vc in zip(
            chain.from_iterable(self.chunks.values()),
            chain.from_iterable(variable_collections),
        ):
            job_phenotype_names = list(job_summary.phenotypes.keys())
            if job_phenotype_names != vc.phenotype_names:
                raise ValueError(
                    "Phenotype names do not match"
                    f"{job_phenotype_names} != {vc.phenotype_names}"
                )
            job_covariate_names = list(job_summary.covariates.keys())
            if job_covariate_names != vc.covariate_names:
                raise ValueError(
                    "Covariate names do not match, "
                    f"{job_covariate_names} != {vc.covariate_names}"
                )

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        with CompressedTextReader(file_path) as file_handle:
            chunks_data = yaml.safe_load(file_handle)
        instance = parse_obj_as(cls, chunks_data)

        return instance

    @classmethod
    def from_variable_collection_chunks(
        cls, variable_collection_chunks: list[list[VariableCollection]]
    ) -> Self:
        return cls(
            {
                f"chunk-{i + 1:d}": [
                    JobSummary.from_variable_collection(vc)
                    for vc in variable_collections
                ]
                for i, variable_collections in enumerate(variable_collection_chunks)
            }
        )


@dataclass
class JobCollection:
    vcf_file: VCFFile
    get_eigendecomposition: Callable[
        [int | str, VariableCollection], Eigendecomposition
    ]
    null_model_method: str
    output_directory: Path
    compression_method: CompressionMethod
    num_threads: int

    variable_collection_chunks: list[list[VariableCollection]]
    summary_collection: SummaryCollection = field(init=False)
    stat_file_array: FileArray = field(init=False)

    sw: SharedWorkspace = field(init=False)

    def __post_init__(self) -> None:
        if len(self.variable_collection_chunks) == 0:
            raise ValueError("No phenotypes to analyze")
        self.sw = next(
            chain.from_iterable(self.variable_collection_chunks)
        ).phenotypes.sw
        # Create an array proxy.
        self.stat_file_array = FileArray.create(
            self.file_path,
            (self.vcf_file.variant_count, self.phenotype_count * 2),
            np.float64,
            compression_method=self.compression_method,
            num_threads=self.num_threads,
        )
        # Set column names
        phenotype_names = [
            f"{phenotype_name}_stat-{stat}"
            for variable_collections in self.variable_collection_chunks
            for vc in variable_collections
            for phenotype_name in vc.phenotype_names
            for stat in ["u", "v"]
        ]
        self.stat_file_array.set_axis_metadata(1, pd.Series(phenotype_names))
        # Set row metadata
        self.stat_file_array.set_axis_metadata(
            0, self.vcf_file.vcf_variants.iloc[self.vcf_file.variant_indices]
        )
        # Try to load an existing summary collection.
        chunks_path = self.file_path.with_suffix(".yaml.gz")
        if chunks_path.is_file():
            try:
                summary_collection = SummaryCollection.from_file(chunks_path)
                summary_collection.validate(self.variable_collection_chunks)
                self.summary_collection = summary_collection
                return
            except ValueError as e:
                logger.warning(
                    f'Failed to load summary collection from "{chunks_path}"',
                    exc_info=e,
                )
        # Create a new summary collection.
        self.summary_collection = SummaryCollection.from_variable_collection_chunks(
            self.variable_collection_chunks
        )
        self.dump()

    def dump(self) -> None:
        value = asdict(self.summary_collection)
        with CompressedTextWriter(
            self.file_path.with_suffix(".metadata.yaml.gz")
        ) as file_handle:
            yaml.dump(value, file_handle, sort_keys=False, width=np.inf)

    def run(self) -> None:
        sw = self.sw
        for variable_collections, summaries in zip(
            self.variable_collection_chunks, self.summary_collection.chunks.values()
        ):
            eigendecompositions = [
                self.get_eigendecomposition(self.chromosome, vc)
                for vc in tqdm(
                    variable_collections,
                    unit="eigendecompositions",
                    desc="decomposing kinship matrices",
                )
            ]
            inverse_variance_arrays: list[SharedArray] = list()
            scaled_residuals_arrays: list[SharedArray] = list()
            for eig, vc, summary in zip(
                eigendecompositions, variable_collections, summaries
            ):
                nm = NullModelCollection.from_eig(
                    eig,
                    vc,
                    method=self.null_model_method,
                    num_threads=self.num_threads,
                )
                summary.put_null_model_collection(nm)
                # Extract the matrices we actually need from the nm.
                variance = nm.variance.to_numpy()
                (sample_count, phenotype_count) = variance.shape
                half_scaled_residuals = nm.half_scaled_residuals.to_numpy()
                # Pre-compute the inverse variance.
                inverse_variance_array = sw.alloc(
                    SharedArray.get_name(sw, "inverse-variance"),
                    sample_count,
                    phenotype_count,
                )
                inverse_variance_array[:] = np.reciprocal(variance)
                inverse_variance_arrays.append(inverse_variance_array)
                # Pre-compute the inverse variance scaled residuals.
                scaled_residuals_array = sw.alloc(
                    SharedArray.get_name(sw, "scaled-residuals"),
                    sample_count,
                    phenotype_count,
                )
                scaled_residuals_array[:] = half_scaled_residuals / np.sqrt(variance)
                scaled_residuals_arrays.append(scaled_residuals_array)
                # Free the nm.
                nm.free()
            self.dump()
            calc_score(
                self.vcf_file,
                eigendecompositions,
                inverse_variance_arrays,
                scaled_residuals_arrays,
                self.stat_file_array,
            )
            for summary in summaries:
                summary.status = "score_complete"
            self.dump()

    @property
    def chromosome(self) -> int | str:
        return self.vcf_file.chromosome

    @property
    def phenotype_count(self) -> int:
        return sum(
            vc.phenotype_count
            for vc in chain.from_iterable(self.variable_collection_chunks)
        )

    @property
    def file_path(self) -> Path:
        return self.output_directory / f"chr{self.chromosome}.score"
