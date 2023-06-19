# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Self

import numpy as np
import yaml
from tqdm import tqdm

from ..compression.arr import ArrayProxy, CompressionMethod
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
    chunks: list[list[JobSummary]]

    def validate(self, variable_collections: list[list[VariableCollection]]) -> None:
        for job_summary, vc in zip(
            chain.from_iterable(self.chunks),
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
    def from_variable_collections(
        cls, variable_collections: list[list[VariableCollection]]
    ) -> Self:
        return cls(
            [
                [JobSummary.from_variable_collection(vc) for vc in vcs]
                for vcs in variable_collections
            ]
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

    variable_collections: list[list[VariableCollection]]
    summary_collection: SummaryCollection = field(init=False)
    array_proxy: ArrayProxy = field(init=False)

    sw: SharedWorkspace = field(init=False)

    def __post_init__(self) -> None:
        if len(self.variable_collections) == 0:
            raise ValueError("No phenotypes to analyze")
        self.sw = next(chain.from_iterable(self.variable_collections)).phenotypes.sw
        # Create an array proxy.
        self.array_proxy = ArrayProxy(
            self.file_path.with_suffix(self.compression_method.suffix),
            (self.vcf_file.variant_count, self.phenotype_count, 2),
            np.float64,
            self.compression_method,
            self.num_threads,
        )
        # Try to load an existing summary collection.
        chunks_path = self.file_path.with_suffix(".yaml.gz")
        if chunks_path.is_file():
            try:
                summary_collection = SummaryCollection.from_file(chunks_path)
                summary_collection.validate(self.variable_collections)
                self.summary_collection = summary_collection
                return
            except ValueError as e:
                logger.warning(
                    f'Failed to load summary collection from "{chunks_path}"',
                    exc_info=e,
                )
        # Create a new summary collection.
        self.summary_collection = SummaryCollection.from_variable_collections(
            self.variable_collections
        )
        self.dump()

    def dump(self) -> None:
        value = asdict(self.summary_collection)
        with CompressedTextWriter(
            self.file_path.with_suffix(".yaml.gz")
        ) as file_handle:
            yaml.dump(value, file_handle, sort_keys=False, width=np.inf)

    def run(self) -> None:
        sw = self.sw
        for vcs, summaries in zip(
            self.variable_collections, self.summary_collection.chunks
        ):
            eigendecompositions = [
                self.get_eigendecomposition(self.chromosome, vc)
                for vc in tqdm(
                    vcs,
                    unit="eigendecompositions",
                    desc="decomposing kinship matrices",
                )
            ]
            iv_arrays: list[SharedArray] = list()  # Inverse variance
            ivsr_arrays: list[SharedArray] = list()  # Inverse variance scaled residuals
            for eig, vc, summary in zip(eigendecompositions, vcs, summaries):
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
                scaled_residuals = nm.scaled_residuals.to_numpy()
                # Pre-compute the inverse variance.
                iv = sw.alloc(
                    SharedArray.get_name(sw, "inverse-variance"),
                    sample_count,
                    phenotype_count,
                )
                iv[:] = np.power(variance, -0.5)
                iv_arrays.append(iv)
                # Pre-compute the inverse variance scaled residuals.
                ivsr = sw.alloc(
                    SharedArray.get_name(sw, "inverse-variance-scaled-residuals"),
                    sample_count,
                    phenotype_count,
                )
                ivsr[:] = scaled_residuals * iv.to_numpy()
                ivsr_arrays.append(ivsr)
                # Free the nm.
                nm.free()
            self.dump()
            calc_score(
                self.vcf_file,
                eigendecompositions,
                iv_arrays,
                ivsr_arrays,
                self.array_proxy,
            )

    @property
    def chromosome(self) -> int | str:
        return self.vcf_file.chromosome

    @property
    def phenotype_count(self) -> int:
        return sum(
            vc.phenotype_count for vc in chain.from_iterable(self.variable_collections)
        )

    @property
    def file_path(self) -> Path:
        return self.output_directory / f"chr{self.chromosome}"
