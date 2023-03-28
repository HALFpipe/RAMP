# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import IO, Any, ClassVar, Generator, Iterable, Sequence

import numpy as np
import scipy
from numpy import typing as npt
from tqdm.auto import tqdm

from gwas import __version__
from gwas.pheno import VariableCollection
from gwas.utils import to_str, underscore
from gwas.var import NullModelCollection
from gwas.vcf import Variant
from gwas.z import CompressedTextReader, CompressedTextWriter


@dataclass
class VariableSummary:
    name: str
    minimum: float
    lower_quartile: float
    median: float
    upper_quartile: float
    maximum: float
    mean: float
    variance: float

    @property
    def values(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.minimum,
                self.lower_quartile,
                self.median,
                self.upper_quartile,
                self.maximum,
                self.mean,
                self.variance,
            ]
        )

    def is_close(self, other: VariableSummary) -> bool:
        return np.allclose(self.values, other.values)


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
    covariate_summaries: list[VariableSummary]
    inverse_normal: bool
    trait_summaries: list[VariableSummary]
    null_model_estimates: list[NullModelEstimate]
    analyzed_trait: VariableSummary | None
    genetic_variance: npt.NDArray[np.float64]
    error_variance: npt.NDArray[np.float64]
    heritability: npt.NDArray[np.float64]

    @property
    def phenotype_count(self) -> int:
        return len(self.trait_summaries)


@dataclass
class AnnotatedVariant(Variant):
    alternate_allele_count: float
    alternate_allele_frequency: float
    call_rate: float

    @classmethod
    def from_array(cls, array: npt.NDArray) -> Generator[AnnotatedVariant, None, None]:
        for record in array:
            yield cls(
                record["CHROM"],
                record["POS"],
                record["REF"],
                record["ALT"],
                record["ALT_AC"],
                record["ALT_AF"],
                record["CALL_RATE"],
            )


def write_delim(file_handle: IO[str], *values: str):
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

    @staticmethod
    def read_header(
        file_path: Path | str,
    ):
        def parse_delim(value: str) -> tuple[str, npt.NDArray[np.float64]]:
            name, values = value.split("\t", maxsplit=1)
            return name, np.fromstring(values, sep="\t")

        header_dict: dict[str, Any] = dict(
            null_model_estimates=list(),
            covariate_summaries=list(),
            analyzed_trait=None,
            trait_summaries=list(),
        )
        summaries = list()
        with CompressedTextReader(file_path) as file:
            for line in file:
                if line.startswith("##"):
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
                            continue
                        if value.startswith("Name\tBetaHat"):
                            continue
                        name, array = parse_delim(value)
                        header_dict["null_model_estimates"].append(
                            NullModelEstimate(
                                name,
                                beta_hat=array[::2],
                                se_beta_hat=array[1::2],
                            )
                        )
                    elif "\t" in value:
                        # Set active summaries list.
                        if value.startswith("CovariateSummaries"):
                            summaries = header_dict["covariate_summaries"]
                            continue
                        elif value.startswith("TraitSummaries"):
                            summaries = header_dict["trait_summaries"]
                            continue

                        name, array = parse_delim(value)
                        if name == "Sigma_g2_Hat":
                            header_dict["genetic_variance"] = array
                            continue
                        elif name == "Sigma_e2_Hat":
                            header_dict["error_variance"] = array
                            continue
                        elif name == "Heritability":
                            header_dict["heritability"] = array
                            continue

                        summary = VariableSummary(name, *array)
                        if name == "AnalyzedTrait":
                            header_dict["analyzed_trait"] = summary
                        else:
                            summaries.append(summary)

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
    def get_dtype(cls, _: ScorefileHeader):
        return np.dtype(list(zip(cls.names, cls.types)))

    @classmethod
    def read(
        cls,
        file_path: Path | str,
    ) -> tuple[ScorefileHeader, npt.NDArray]:
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
        file_path: Path | str,
        header: ScorefileHeader,
        variants: list[AnnotatedVariant],
        u_stat: npt.NDArray[np.float64],
        v_stat: npt.NDArray[np.float64],
    ):
        with CompressedTextWriter(file_path) as file_handle:
            cls.write_header(file_handle, header)
            cls.write_names(file_handle, np.empty((0,)))
            cls.write_scores(file_handle, variants, u_stat, v_stat)

    @classmethod
    def write(
        cls,
        file_path: Path | str,
        header: ScorefileHeader,
        array: npt.NDArray,
    ):
        with CompressedTextWriter(file_path) as file_handle:
            cls.write_header(file_handle, header)
            cls.write_names(file_handle, np.empty((0,)))
            for record in tqdm(array, leave=False):
                file_handle.write("\t".join(map(to_str, record)) + "\n")

    @staticmethod
    def make_header(
        vc: VariableCollection,
        nm: NullModelCollection,
    ) -> ScorefileHeader:
        covariates = vc.covariates.to_numpy()
        phenotypes = vc.phenotypes.to_numpy()

        def make_summaries(names: list[str], values: npt.NDArray[np.float64]):
            summaries = list()
            for i, name in enumerate(names):
                value = values[:, i]
                (minimum, lower_quartile, media, upper_quartile, maximum) = np.quantile(
                    value, [0.00, 0.25, 0.50, 0.75, 1.00]
                )
                summaries.append(
                    VariableSummary(
                        name,
                        minimum,
                        lower_quartile,
                        media,
                        upper_quartile,
                        maximum,
                        value.mean(),
                        value.var(ddof=1),
                    )
                )
            return summaries

        trait_summaries = make_summaries(vc.phenotype_names, phenotypes)
        analyzed_trait = None
        if len(trait_summaries) == 1:
            (analyzed_trait,) = trait_summaries

        weights = nm.regression_weights.to_numpy()
        errors = nm.standard_errors.to_numpy()
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
            covariates=vc.covariate_names,
            covariate_summaries=make_summaries(vc.covariate_names, covariates[:, 1:]),
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
    ):
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

        def write_summary(summary: VariableSummary, title: str | None = None):
            if title is None:
                title = summary.name
            write_delim(
                file_handle,
                title,
                *map(to_str, summary.values),
            )

        def write_summaries(
            title: str,
            summaries: list[VariableSummary],
        ):
            write_delim(file_handle, title, *summary_columns)
            for summary in summaries:
                write_summary(summary)

        write_summaries("CovariateSummaries", header.covariate_summaries)

        inverse_normal = {
            True: "ON",
            False: "OFF",
        }[header.inverse_normal]
        file_handle.write(f"##InverseNormal={inverse_normal}\n")

        write_summaries("TraitSummaries", header.trait_summaries)

        cls.write_null_model(
            file_handle, header.null_model_estimates, header.trait_summaries
        )

        if header.analyzed_trait is not None:
            write_summary(header.analyzed_trait, title="AnalyzedTrait")

        write_delim(file_handle, "Sigma_g2_Hat", *map(to_str, header.genetic_variance))
        write_delim(file_handle, "Sigma_e2_Hat", *map(to_str, header.error_variance))

        if isinstance(header.heritability, float) or header.heritability.size == 1:
            file_handle.write(f"##Heritability={to_str(header.heritability)}\n")
        else:
            write_delim(file_handle, "Heritability", *map(to_str, header.heritability))

    @classmethod
    def write_names(cls, file_handle: IO[str], _):
        file_handle.write("#" + "\t".join(cls.names) + "\n")

    @staticmethod
    def write_null_model(
        file_handle: IO[str],
        null_model_estimates: list[NullModelEstimate],
        _: list[VariableSummary],
    ):
        file_handle.write("## - NullModelEstimates\n")
        file_handle.write("## - Name\tBetaHat\tSE(BetaHat)\n")

        for n in null_model_estimates:
            write_delim(
                file_handle, f" - {n.name}", to_str(n.beta_hat), to_str(n.se_beta_hat)
            )

    @staticmethod
    def make_metadata(
        variant: AnnotatedVariant,
        **kwargs,
    ) -> Iterable[str]:
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
            variant.alternate_allele_count,
            variant.call_rate,
            hwe_pvalue,
            n_ref,
            n_het,
            n_alt,
        )
        return map(to_str, metadata)

    @classmethod
    def write_score(
        cls,
        file_handle: IO[str],
        u_stat: float,
        v_stat: float,
    ):
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
        variants: list[AnnotatedVariant],
        u_stat: npt.NDArray[np.float64],
        v_stat: npt.NDArray[np.float64],
        **kwargs,
    ):
        if u_stat.shape != v_stat.shape:
            raise ValueError("U and V must have the same shape.")

        if len(variants) != u_stat.shape[0]:
            raise ValueError("Variant count does not match U and V shape.")

        phenotype_count = u_stat.shape[1]
        for i, variant in enumerate(variants):
            file_handle.write("\t".join(cls.make_metadata(variant, **kwargs)))

            for j in range(phenotype_count):
                file_handle.write("\t")
                cls.write_score(file_handle, u_stat[i, j], v_stat[i, j])

            file_handle.write("\n")


class CombinedScorefile(Scorefile):
    reduced_names: ClassVar[tuple[str, ...]] = (
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "ALT_AF",
        "ALT_AC",
    )
    reduced_types: ClassVar = (
        object,
        int,
        object,
        object,
        float,
        float,
    )
    score_names: ClassVar[tuple[str, ...]] = (
        "U_STAT",
        "V_STAT",
    )
    score_types: ClassVar = (
        float,
        float,
    )

    @classmethod
    def get_dtype(cls, header: ScorefileHeader):
        phenotype_names = [p.name for p in header.trait_summaries]
        dtype_list = list(zip(cls.reduced_names, cls.reduced_types))
        for phenotype_name in phenotype_names:
            for score_name, score_type in zip(cls.score_names, cls.score_types):
                dtype_list.append((f"{score_name}[{phenotype_name}]", score_type))
        return np.dtype(dtype_list)

    @staticmethod
    def make_metadata(
        variant: AnnotatedVariant,
        **kwargs,
    ) -> Iterable[str]:
        metadata = (
            variant.chromosome,
            variant.position,
            variant.reference_allele,
            variant.alternate_allele,
            variant.alternate_allele_frequency,
            variant.alternate_allele_count,
        )
        return map(to_str, metadata)

    @staticmethod
    def write_null_model(
        file_handle: IO[str],
        null_model_estimates: list[NullModelEstimate],
        trait_summaries: list[VariableSummary],
    ):
        file_handle.write("## - NullModelEstimates\n")

        header: list[str] = list()
        for s in trait_summaries:
            header.append(f"BetaHat[{s.name}]")
            header.append(f"SE(BetaHat)[{s.name}]")
        write_delim(file_handle, " - Name", *header)

        for n in null_model_estimates:
            values: list[str] = list()
            for j in range(n.beta_hat.size):
                values.append(to_str(n.beta_hat[j]))
                values.append(to_str(n.se_beta_hat[j]))

            write_delim(file_handle, f" - {n.name}", *values)

    @classmethod
    def write_names(cls, file_handle: IO[str], vc: VariableCollection):
        names = [*cls.reduced_names]

        for phenotype_name in vc.phenotype_names:
            for score_name in cls.score_names:
                names.append(f"{score_name}[{phenotype_name}]")

        file_handle.write("#" + "\t".join(names) + "\n")

    @classmethod
    def write_score(
        cls,
        file_handle: IO[str],
        u_stat: float,
        v_stat: float,
    ):
        stats = (
            u_stat,
            v_stat,
        )
        file_handle.write("\t".join(map(to_str, stats)))

    @classmethod
    def to_scorefiles(
        cls,
        file_path: Path | str,
        prefix: Path | str,
    ) -> Generator[Path, None, None]:
        file_path = Path(file_path)
        header, array = cls.read(file_path)

        tokens = file_path.name.split(".")
        suffix = tokens.pop(-1)
        if tokens[-1] == "txt":
            suffix = f"{tokens.pop(-1)}.{suffix}"

        for i, summary in enumerate(header.trait_summaries):
            phenotype_name = summary.name
            phenotype_path = Path(f"{prefix}.{phenotype_name}.{suffix}")

            phenotype_header = copy(header)
            phenotype_header.trait_summaries = [summary]
            phenotype_header.analyzed_trait = summary

            for n in phenotype_header.null_model_estimates:
                n.beta_hat = n.beta_hat[i]
                n.se_beta_hat = n.se_beta_hat[i]

            phenotype_header.heritability = header.heritability[i]
            phenotype_header.genetic_variance = header.genetic_variance[i]
            phenotype_header.error_variance = header.error_variance[i]

            variants = list(AnnotatedVariant.from_array(array))
            u_stat = array[f"U_STAT[{phenotype_name}]"]
            v_stat = array[f"V_STAT[{phenotype_name}]"]

            cls.from_stat(phenotype_path, phenotype_header, variants, u_stat, v_stat)

            yield phenotype_path

    @classmethod
    def from_scorefiles(
        cls,
        file_paths: Sequence[Path | str],
    ) -> tuple[ScorefileHeader, npt.NDArray]:
        raise NotImplementedError
        # headers = list()
        # arrays = list()
        # for path in file_paths:
        #     header, array = Scorefile.read(path)
        #     headers.append(header)
        #     arrays.append(array)

        # array = np.vstack(arrays).transpose()

        # u_stat = array["U_STAT"]
        # sqrt_v_stat = array["SQRT_V_STAT"]
        # v_stat = np.square(sqrt_v_stat)
