import re
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import IO, Any, ClassVar

import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt
from tqdm.auto import tqdm
from upath import UPath

from .. import __version__
from ..compression.pipe import CompressedTextReader, CompressedTextWriter
from ..null_model.base import NullModelCollection
from ..pheno import VariableCollection, VariableSummary
from ..utils.numpy import to_str


def underscore(x: str) -> str:
    return re.sub(r"([a-z\d])([A-Z])", r"\1_\2", x).lower()


@dataclass
class NullModelEstimate:
    name: str
    beta_hat: npt.NDArray[np.float64]
    se_beta_hat: npt.NDArray[np.float64]


@dataclass
class ScorefileHeader:
    program_name: str
    version: str
    samples: int
    analyzed_samples: int
    families: int
    analyzed_families: int
    founders: int
    make_residuals: bool
    analyzed_founders: int
    covariates: list[str]
    covariate_summaries: OrderedDict[str, VariableSummary]
    inverse_normal: bool
    trait_summaries: OrderedDict[str, VariableSummary]
    null_model_estimates: list[NullModelEstimate]
    analyzed_trait: VariableSummary | None
    genetic_variance: npt.NDArray[np.float64]
    error_variance: npt.NDArray[np.float64]
    heritability: npt.NDArray[np.float64]

    @property
    def phenotype_count(self) -> int:
        return len(self.trait_summaries)


def parse_delim(value: str) -> tuple[str, npt.NDArray[np.float64]]:
    name, values = value.split("\t", maxsplit=1)
    return name, np.fromstring(values, sep="\t")


def write_delim(file_handle: IO[str], *values: str) -> None:
    file_handle.write("##" + "\t".join(values) + "\n")


class Scorefile:
    names: ClassVar[tuple[str, ...]] = (
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "N_INFORMATIVE",
        "FOUNDER_AF",
        "ALL_AF",
        "INFORMATIVE_ALT_AC",
        "CALL_RATE",
        "HWE_PVALUE",
        "N_REF",
        "N_HET",
        "N_ALT",
        "U_STAT",
        "SQRT_V_STAT",
        "ALT_EFFSIZE",
        "PVALUE",
    )

    types: ClassVar = (
        object,
        int,
        object,
        object,
        int,
        float,
        float,
        float,
        float,
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        float,
    )

    @classmethod
    def parse_header_line(
        cls,
        line: str,
        header_dict: dict[str, Any],
        summaries: OrderedDict[str, VariableSummary],
    ) -> None:
        value = line.removeprefix("##")
        if "=" in value:
            key, value = value.split("=", maxsplit=1)
            key = underscore(key)

            value = value.strip()
            if "," in value:
                header_dict[key] = value.split(",")
            elif "\t" in value:
                header_dict[key] = value.split("\t")
            else:
                if key in {"heritability"}:
                    header_dict[key] = float(value)
                else:
                    header_dict[key] = value
        elif value.startswith(" - "):
            value = value.removeprefix(" - ")
            if value.startswith("NullModelEstimates"):
                return
            if value.startswith("Name\tBetaHat"):
                return
            name, array = parse_delim(value)
            header_dict["null_model_estimates"].append(
                NullModelEstimate(
                    name,
                    beta_hat=array[::2],
                    se_beta_hat=array[1::2],
                )
            )
        elif "\t" in value:
            cls.parse_header_line_delim(value, header_dict, summaries)

    @staticmethod
    def parse_header_line_delim(
        value: str,
        header_dict: dict[str, Any],
        summaries: OrderedDict[str, VariableSummary],
    ) -> None:
        # Set active summaries list.
        if value.startswith("CovariateSummaries"):
            summaries = header_dict["covariate_summaries"]
            return
        elif value.startswith("TraitSummaries"):
            summaries = header_dict["trait_summaries"]
            return

        name, array = parse_delim(value)
        if name == "Sigma_g2_Hat":
            header_dict["genetic_variance"] = array
            return
        elif name == "Sigma_e2_Hat":
            header_dict["error_variance"] = array
            return
        elif name == "Heritability":
            header_dict["heritability"] = array
            return

        summary = VariableSummary(*array)
        if name == "AnalyzedTrait":
            header_dict["analyzed_trait"] = summary
        else:
            summaries[name] = summary

    @classmethod
    def read_header(cls, file_path: UPath) -> ScorefileHeader:
        header_dict: dict[str, Any] = dict(
            null_model_estimates=list(),
            covariate_summaries=list(),
            analyzed_trait=None,
            trait_summaries=list(),
        )
        summaries: OrderedDict[str, VariableSummary] = OrderedDict()
        with CompressedTextReader(file_path) as file:
            for line in file:
                if line.startswith("##"):
                    cls.parse_header_line(line, header_dict, summaries)
                    continue
                if line.startswith("#"):
                    continue
                break

        for field in fields(ScorefileHeader):
            if field.type is int:
                header_dict[field.name] = int(header_dict[field.name])
            elif field.type is bool:
                header_dict[field.name] = {
                    "ON": True,
                    "OFF": False,
                    "True": True,
                    "False": False,
                }[header_dict[field.name]]

        return ScorefileHeader(**header_dict)

    @classmethod
    def get_dtype(cls, _: ScorefileHeader) -> npt.DTypeLike:
        return np.dtype(list(zip(cls.names, cls.types, strict=True)))

    @classmethod
    def read(cls, file_path: UPath) -> tuple[ScorefileHeader, npt.NDArray[np.float64]]:
        line_count = 0
        with CompressedTextReader(file_path) as file:
            for line in file:
                if line.startswith("#"):
                    continue
                line_count += 1

        header = cls.read_header(file_path)
        dtype = cls.get_dtype(header)

        array = np.empty((line_count,), dtype=dtype)

        j = 0
        with CompressedTextReader(file_path) as file:
            for line in file:
                if line.startswith("#"):
                    continue
                tokens = line.split()
                token: float | str | None = None
                for i, token in enumerate(tokens):
                    if token == "NA":
                        token = np.nan
                    array[j][i] = token
                j += 1

        return header, array

    @classmethod
    def from_stat(
        cls,
        file_path: UPath | str,
        header: ScorefileHeader,
        variants: pd.DataFrame,
        u_stat: npt.NDArray[np.float64],
        v_stat: npt.NDArray[np.float64],
        num_threads: int,
    ) -> None:
        with CompressedTextWriter(file_path, num_threads) as file_handle:
            cls.write_header(file_handle, header)
            cls.write_names(file_handle, np.empty((0,)))
            cls.write_scores(file_handle, variants, u_stat, v_stat)

    @classmethod
    def write(
        cls,
        file_path: UPath | str,
        header: ScorefileHeader,
        array: npt.NDArray[np.float64],
        num_threads: int,
    ) -> None:
        with CompressedTextWriter(file_path, num_threads) as file_handle:
            cls.write_header(file_handle, header)
            cls.write_names(file_handle, np.empty((0,)))
            for record in tqdm(array, leave=False):
                file_handle.write("\t".join(map(to_str, record)) + "\n")

    @staticmethod
    def make_header(
        vc: VariableCollection,
        nm: NullModelCollection,
        phenotype_name: str | None = None,
    ) -> ScorefileHeader:
        covariates = vc.covariates
        phenotypes = vc.phenotypes

        phenotype_index: int | None = None
        if phenotype_name is not None:
            phenotype_index = vc.phenotype_names.index(phenotype_name)

        def make_summaries(
            names: list[str], values: npt.NDArray[np.float64]
        ) -> OrderedDict[str, VariableSummary]:
            summaries: OrderedDict[str, VariableSummary] = OrderedDict()
            for i, name in enumerate(names):
                if phenotype_name is not None and name != phenotype_name:
                    continue

                value = values[:, i]
                summaries[name] = VariableSummary.from_array(value)
            return summaries

        trait_summaries = make_summaries(vc.phenotype_names, phenotypes)
        analyzed_trait = None
        if len(trait_summaries) == 1:
            (analyzed_trait,) = trait_summaries.values()
        covariate_summaries = make_summaries(vc.covariate_names[1:], covariates[:, 1:])

        weights = nm.regression_weights
        errors = nm.standard_errors

        if phenotype_index is not None:
            weights = weights[phenotype_index, :]
            errors = errors[phenotype_index, :]

        null_model_estimates = [
            NullModelEstimate(
                name,
                weights[:, i],
                errors[:, i],
            )
            for i, name in enumerate(vc.covariate_names)
        ]

        return ScorefileHeader(
            program_name=__name__,
            version=__version__,
            samples=vc.sample_count,
            analyzed_samples=vc.sample_count,
            families=vc.sample_count,
            analyzed_families=vc.sample_count,
            founders=vc.sample_count,
            make_residuals=False,
            analyzed_founders=vc.sample_count,
            covariates=vc.covariate_names[1:],  # skip intercept
            covariate_summaries=covariate_summaries,
            inverse_normal=False,
            trait_summaries=trait_summaries,
            null_model_estimates=null_model_estimates,
            analyzed_trait=analyzed_trait,
            genetic_variance=nm.genetic_variance,
            error_variance=nm.error_variance,
            heritability=nm.heritability,
        )

    @classmethod
    def write_header(
        cls,
        file_handle: IO[str],
        header: ScorefileHeader,
    ) -> None:
        file_handle.write(f"##ProgramName={header.program_name}\n")
        file_handle.write(f"##Version={header.version}\n")

        file_handle.write(f"##Samples={header.samples}\n")
        file_handle.write(f"##AnalyzedSamples={header.analyzed_samples}\n")
        file_handle.write(f"##Families={header.families}\n")
        file_handle.write(f"##AnalyzedFamilies={header.analyzed_families}\n")
        file_handle.write(f"##Founders={header.founders}\n")
        file_handle.write(f"##MakeResiduals={header.make_residuals}\n")
        file_handle.write(f"##AnalyzedFounders={header.analyzed_founders}\n")

        file_handle.write(f"##Covariates={','.join(header.covariates)}\n")

        summary_columns = ["min", "25th", "median", "75th", "max", "mean", "variance"]

        def write_summary(name: str, summary: VariableSummary) -> None:
            write_delim(
                file_handle,
                name,
                *map(to_str, summary.values),
            )

        def write_summaries(
            title: str,
            summaries: OrderedDict[str, VariableSummary],
        ) -> None:
            write_delim(file_handle, title, *summary_columns)
            for name, summary in summaries.items():
                write_summary(name, summary)

        write_summaries("CovariateSummaries", header.covariate_summaries)

        inverse_normal = {
            True: "ON",
            False: "OFF",
        }[header.inverse_normal]
        file_handle.write(f"##InverseNormal={inverse_normal}\n")

        write_summaries("TraitSummaries", header.trait_summaries)

        cls.write_null_model(file_handle, header.null_model_estimates)

        if header.analyzed_trait is not None:
            write_summary("AnalyzedTrait", header.analyzed_trait)

        write_delim(file_handle, "Sigma_g2_Hat", *map(to_str, header.genetic_variance))
        write_delim(file_handle, "Sigma_e2_Hat", *map(to_str, header.error_variance))

        if isinstance(header.heritability, float) or header.heritability.size == 1:
            file_handle.write(f"##Heritability={to_str(header.heritability)}\n")
        else:
            write_delim(file_handle, "Heritability", *map(to_str, header.heritability))

    @classmethod
    def write_names(cls, file_handle: IO[str], _: Any) -> None:
        file_handle.write("#" + "\t".join(cls.names) + "\n")

    @staticmethod
    def write_null_model(
        file_handle: IO[str],
        null_model_estimates: list[NullModelEstimate],
    ) -> None:
        file_handle.write("## - NullModelEstimates\n")
        file_handle.write("## - Name\tBetaHat\tSE(BetaHat)\n")

        for n in null_model_estimates:
            write_delim(
                file_handle, f" - {n.name}", to_str(n.beta_hat), to_str(n.se_beta_hat)
            )

    @staticmethod
    def make_metadata(
        variant: Any,
        **kwargs: int,
    ) -> list[str]:
        # RareMetalWorker only calculates these for genotype data, not dosage data,
        # so we just set them to pre-defined values.
        hwe_pvalue = 1.0
        n_ref = 1
        n_het = 1
        n_alt = 1

        n_informative = kwargs.get("n_informative", -1)

        metadata = (
            variant.chromosome,
            variant.position,
            variant.reference_allele,
            variant.alternate_allele,
            n_informative,
            variant.alternate_allele_frequency,
            variant.alternate_allele_frequency,
            np.nan,
            1,
            hwe_pvalue,
            n_ref,
            n_het,
            n_alt,
        )
        return list(map(to_str, metadata))

    @classmethod
    def write_score(
        cls,
        file_handle: IO[str],
        u_stat: float,
        v_stat: float,
    ) -> None:
        alt_effsize = u_stat / v_stat
        chi2 = np.square(u_stat) / v_stat
        pvalue = scipy.stats.chi2.sf(chi2, 1)

        stats = (
            u_stat,
            np.sqrt(v_stat),
            alt_effsize,
            pvalue,
        )
        file_handle.write("\t".join(map(to_str, stats)))

    @classmethod
    def write_scores(
        cls,
        file_handle: IO[str],
        variants: pd.DataFrame,
        u_stat: npt.NDArray[np.float64],
        v_stat: npt.NDArray[np.float64],
        **kwargs: int,
    ) -> None:
        if u_stat.shape != v_stat.shape:
            raise ValueError("U and V must have the same shape.")

        if len(variants) != u_stat.shape[0]:
            raise ValueError("Variant count does not match U and V shape.")

        phenotype_count = u_stat.shape[1]
        for i, variant in enumerate(variants.itertuples()):
            metadata = cls.make_metadata(variant, **kwargs)
            file_handle.write("\t".join(metadata))

            for j in range(phenotype_count):
                file_handle.write("\t")
                cls.write_score(file_handle, u_stat[i, j], v_stat[i, j])

            file_handle.write("\n")
