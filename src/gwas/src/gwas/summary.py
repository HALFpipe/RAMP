from dataclasses import dataclass, fields, is_dataclass
from itertools import chain
from typing import Any, Literal, Self, Type, TypeVar, get_args, get_origin

import yaml
from upath import UPath

from .compression.pipe import CompressedTextReader
from .log import logger
from .null_model.base import NullModelCollection
from .pheno import VariableCollection, VariableSummary

T = TypeVar("T")


def parse_obj_as(cls: Type[T], data: Any) -> T:
    """Parses an object as the specified type. Inspired by the Pydantic function of the
    same name.

    Args:
        cls (Type[T]): The type to parse as.
        data (Any): The data to parse.

    Returns:
        T: The parsed object.
    """
    if is_dataclass(cls):
        return cls(
            **{f.name: parse_obj_as(f.type, data.get(f.name)) for f in fields(cls)}
        )  # type: ignore

    origin = get_origin(cls)
    if origin is list:
        (value_cls,) = get_args(cls)
        return [parse_obj_as(value_cls, element) for element in data]  # type: ignore
    elif origin is dict:
        (key_cls, value_cls) = get_args(cls)
        return {
            parse_obj_as(key_cls, key): parse_obj_as(value_cls, value)
            for key, value in data.items()
        }  # type: ignore

    return data


@dataclass(frozen=True, slots=True)
class RegressionWeight:
    value: float
    standard_error: float


@dataclass
class CovariateSummary(VariableSummary):
    pass


@dataclass
class PhenotypeSummary(VariableSummary):
    method: str | None = None
    log_likelihood: float | None = None
    genetic_variance: float | None = None
    error_variance: float | None = None
    heritability: float | None = None

    regression_weights: dict[str, RegressionWeight] | None = None


@dataclass
class VariableCollectionSummary:
    status: Literal["pending", "null_model_complete", "score_complete"]

    sample_count: int

    phenotypes: dict[str, PhenotypeSummary]
    covariates: dict[str, VariableSummary]

    def put_null_model_collection(self, nm: NullModelCollection) -> None:
        self.status = "null_model_complete"
        for i, name in enumerate(self.phenotypes.keys()):
            self.phenotypes[name].method = nm.method
            self.phenotypes[name].log_likelihood = float(nm.log_likelihood[i])
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
                    strict=True,
                )
            }

    @classmethod
    def from_variable_collection(cls, vc: VariableCollection) -> Self:
        return cls(
            "pending",
            vc.sample_count,
            {
                name: PhenotypeSummary.from_array(vc.phenotypes[:, i])
                for i, name in enumerate(vc.phenotype_names)
            },
            {
                name: CovariateSummary.from_array(vc.covariates[:, i])
                for i, name in enumerate(vc.covariate_names)
            },
        )


@dataclass
class SummaryCollection:
    chunks: dict[str, dict[str, VariableCollectionSummary]]

    def validate(self, variable_collections: list[list[VariableCollection]]) -> None:
        for summary, vc in zip(
            chain.from_iterable(chunk.values() for chunk in self.chunks.values()),
            chain.from_iterable(variable_collections),
            strict=True,
        ):
            phenotype_names = list(summary.phenotypes.keys())
            if phenotype_names != vc.phenotype_names:
                raise ValueError(
                    "Phenotype names do not match"
                    f"{phenotype_names} != {vc.phenotype_names}"
                )
            covariate_names = list(summary.covariates.keys())
            if covariate_names != vc.covariate_names:
                raise ValueError(
                    "Covariate names do not match, "
                    f"{covariate_names} != {vc.covariate_names}"
                )

    @classmethod
    def from_file(cls, file_path: UPath) -> Self:
        with CompressedTextReader(file_path) as file_handle:
            chunks_data = yaml.safe_load(file_handle)
        instance = parse_obj_as(cls, chunks_data)

        return instance

    @classmethod
    def from_variable_collection_chunks(
        cls, variable_collection_chunks: list[list[VariableCollection]]
    ) -> Self:
        logger.debug(
            f"Creating summary collection for {len(variable_collection_chunks)} chunks"
        )
        chunks: dict[str, dict[str, VariableCollectionSummary]] = dict()
        for i, variable_collections in enumerate(variable_collection_chunks):
            chunk: dict[str, VariableCollectionSummary] = dict()
            for vc in variable_collections:
                if vc.name is None:
                    raise ValueError("VariableCollection must have a name")
                chunk[vc.name] = VariableCollectionSummary.from_variable_collection(vc)
            chunks[f"chunk-{i + 1:d}"] = chunk
        return cls(chunks)
