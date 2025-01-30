import re
from dataclasses import dataclass
from shutil import copyfile
from subprocess import PIPE, STDOUT, run
from tempfile import TemporaryDirectory

import polars as pl
from upath import UPath

from ...tools import ldsc, ldsc_munge_sumstats


def get_ld_scores(cache_path: UPath) -> tuple[UPath, list[UPath], UPath]:
    # TODO
    snplist_path = cache_path / "w_hm3.snplist"
    ld_score_paths = [cache_path / "eur_ancestry", cache_path / "mixed_ancestry"]
    weights_path = cache_path / "1000G_Phase3_weights_hm3_no_MHC" / "weights.hm3_noMHC."

    return snplist_path, ld_score_paths, weights_path


def run_ldsc(
    cache_path: UPath, data_frame: pl.DataFrame, output_path: UPath
) -> tuple[str, list[str]]:
    snplist_path, ld_score_paths, weights_path = get_ld_scores(cache_path)
    snplist_frame = pl.read_csv(
        snplist_path,
        separator="\t",
        has_header=True,
        schema=pl.Schema(
            dict(snp=pl.String(), allele1=pl.Categorical(), allele2=pl.Categorical())
        ),
    )

    with TemporaryDirectory() as temporary_path_str:
        temporary_path = UPath(temporary_path_str)

        sumstats_path = temporary_path / "sumstats.txt"

        snp = (
            pl.when(pl.col("rsid").is_null())
            .then(pl.col("marker_name"))
            .otherwise(pl.col("rsid"))
            .alias("snp")
        )
        allele1_uppercase = pl.col("allele1").str.to_uppercase()
        allele2_uppercase = pl.col("allele2").str.to_uppercase()
        nstudy = (
            pl.col("direction").str.len_chars()
            - pl.col("direction").str.count_matches(r"\?")
        ).alias("nstudy")
        n = pl.col("sample_count").alias("n")

        ldsc_frame = data_frame.with_columns(
            snp,
            allele1_uppercase,
            allele2_uppercase,
            nstudy,
            n,
        ).select(["snp", "allele1", "allele2", "nstudy", "n", "p_value", "effect"])
        ldsc_frame = ldsc_frame.join(snplist_frame, on="snp", how="inner")
        ldsc_frame.write_csv(sumstats_path, separator="\t", quote_style="never")

        munge_sumstats_log = run(
            args=[
                *ldsc_munge_sumstats,
                "--sumstats",
                "sumstats.txt",
                "--merge-alleles",
                str(snplist_path),
                "--out",
                str("trait"),
            ],
            text=True,
            check=True,
            stdout=PIPE,
            stderr=STDOUT,
            cwd=temporary_path,
            timeout=1 * 60 * 60,  # one hour
        ).stdout

        copyfile(
            temporary_path / "trait.sumstats.gz",
            output_path.with_suffix(".ldsc.sumstats.gz"),
        )
        ldsc_logs = [
            run(
                args=[
                    *ldsc,
                    "--h2",
                    "trait.sumstats.gz",
                    "--ref-ld-chr",
                    f"{ld_score_path}/"
                    if ld_score_path.is_dir()
                    else f"{ld_score_path}",
                    "--w-ld-chr",
                    str(weights_path),
                    "--out",
                    "ldsc",
                ],
                text=True,
                check=True,
                stdout=PIPE,
                stderr=STDOUT,
                cwd=temporary_path,
                timeout=1 * 60 * 60,  # one hour
            ).stdout
            for ld_score_path in ld_score_paths
        ]

        return munge_sumstats_log, ldsc_logs


@dataclass
class ValueSE:
    value: float
    se: float


@dataclass
class MungeSumstatsOutput:
    mean_chisq: float
    lambda_gc: float
    max_chisq: float


@dataclass
class LDSCOutput:
    mean_chisq: float
    lambda_gc: float
    intercept: ValueSE
    ratio: ValueSE | str
    total_observed_scale_h2: ValueSE


@dataclass
class LDSCOutputCollection:
    munge_sumstats: MungeSumstatsOutput
    data: dict[str, LDSCOutput]


def parse_logs(munge_sumstats_log: str, ldsc_logs: list[str]) -> LDSCOutputCollection:
    def get_value(key: str, log: str) -> str:
        match = re.search(
            rf"^{re.escape(key)} ?[=:]? ?(?P<value>.+?)\.?$",
            log,
            flags=re.MULTILINE,
        )
        if match is None:
            raise ValueError
        return match.group("value")

    mean_chisq = float(get_value("Mean chi^2", munge_sumstats_log))
    lambda_gc = float(get_value("Lambda GC", munge_sumstats_log))
    max_chisq = float(get_value("Max chi^2", munge_sumstats_log))

    munge_sumstats_output = MungeSumstatsOutput(mean_chisq, lambda_gc, max_chisq)

    ldsc_outputs: dict[str, LDSCOutput] = dict()
    for ldsc_log in ldsc_logs:
        match = re.search(
            r"^--ref-ld-chr (?P<value>.+?) \\$", ldsc_log, flags=re.MULTILINE
        )
        if match is None:
            raise ValueError
        key = UPath(match.group("value")).name

        mean_chisq = float(get_value("Mean Chi^2", ldsc_log))
        lambda_gc = float(get_value("Lambda GC", ldsc_log))

        def parse_value_se(value_se: str) -> ValueSE:
            match = re.match(r"^(?P<value>.+) \((?P<se>.+)\)$", value_se)
            if match is None:
                raise ValueError
            value = float(match.group("value"))
            se = float(match.group("se"))
            return ValueSE(value, se)

        intercept = parse_value_se(get_value("Intercept", ldsc_log))
        ratio_str = get_value("Ratio", ldsc_log)
        if ratio_str in {
            "< 0 (usually indicates GC correction)",
            "NA (mean chi^2 < 1)",
        }:
            ratio: ValueSE | str = ratio_str
        else:
            ratio = parse_value_se(ratio_str)
        total_observed_scale_h2 = parse_value_se(
            get_value("Total Observed scale h2", ldsc_log)
        )

        ldsc_outputs[key] = LDSCOutput(
            mean_chisq, lambda_gc, intercept, ratio, total_observed_scale_h2
        )

    return LDSCOutputCollection(munge_sumstats_output, ldsc_outputs)
