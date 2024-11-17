import re
from dataclasses import dataclass
from subprocess import PIPE, STDOUT, run
from tempfile import TemporaryDirectory

import polars as pl
from upath import UPath

from ...tools import ldsc, ldsc_munge_sumstats


def get_ld_scores(cache_path: UPath) -> tuple[UPath, UPath, UPath]:
    # TODO
    snplist_path = cache_path / "w_hm3.snplist"
    ld_scores_path = cache_path / "baselineLD_v2.3" / "baselineLD."
    weights_path = cache_path / "1000G_Phase3_weights_hm3_no_MHC" / "weights.hm3_noMHC."

    return snplist_path, ld_scores_path, weights_path


def run_ldsc(cache_path: UPath, data_frame: pl.DataFrame) -> tuple[str, str]:
    snplist_path, ld_scores_path, weights_path = get_ld_scores(cache_path)
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
        ldsc_frame.write_csv(sumstats_path, separator="\t")

        munge_sumstats_log = run(
            args=[
                *ldsc_munge_sumstats,
                "--sumstats",
                "sumstats.txt",
                "--merge-alleles",
                str(snplist_path),
                "--out",
                str("munge"),
            ],
            text=True,
            check=True,
            stdout=PIPE,
            stderr=STDOUT,
            cwd=temporary_path,
            timeout=1 * 60 * 60,  # one hour
        ).stdout

        ldsc_log = run(
            args=[
                *ldsc,
                "--h2",
                "munge.sumstats.gz",
                "--ref-ld-chr",
                str(ld_scores_path),
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

        return munge_sumstats_log, ldsc_log


@dataclass
class ValueSE:
    value: float
    se: float


@dataclass
class LDSCOutput:
    munge_sumstats_mean_chisq: float
    munge_sumstats_lambda_gc: float
    munge_sumstats_max_chisq: float

    ldsc_mean_chisq: float
    ldsc_lambda_gc: float
    ldsc_intercept: ValueSE
    ldsc_ratio: ValueSE | str
    ldsc_total_observed_scale_h2: ValueSE


def parse_logs(munge_sumstats_log: str, ldsc_log: str) -> LDSCOutput:
    def get_value(key: str, log: str) -> str:
        match = re.search(
            rf"^{re.escape(key)} ?[=:]? ?(?P<value>.+?)\.?$",
            log,
            flags=re.MULTILINE,
        )
        if match is None:
            raise ValueError
        return match.group("value")

    munge_sumstats_mean_chisq = float(get_value("Mean chi^2", munge_sumstats_log))
    munge_sumstats_lambda_gc = float(get_value("Lambda GC", munge_sumstats_log))
    munge_sumstats_max_chisq = float(get_value("Max chi^2", munge_sumstats_log))

    ldsc_mean_chisq = float(get_value("Mean Chi^2", ldsc_log))
    ldsc_lambda_gc = float(get_value("Lambda GC", ldsc_log))

    def parse_value_se(value_se: str) -> ValueSE:
        match = re.match(r"^(?P<value>.+) \((?P<se>.+)\)$", value_se)
        if match is None:
            raise ValueError
        value = float(match.group("value"))
        se = float(match.group("se"))
        return ValueSE(value, se)

    ldsc_intercept = parse_value_se(get_value("Intercept", ldsc_log))
    ldsc_ratio_str = get_value("Ratio", ldsc_log)
    if ldsc_ratio_str in {
        "< 0 (usually indicates GC correction)",
        "NA (mean chi^2 < 1)",
    }:
        ldsc_ratio: ValueSE | str = ldsc_ratio_str
    else:
        ldsc_ratio = parse_value_se(ldsc_ratio_str)
    ldsc_total_observed_scale_h2 = parse_value_se(
        get_value("Total Observed scale h2", ldsc_log)
    )

    return LDSCOutput(
        munge_sumstats_mean_chisq=munge_sumstats_mean_chisq,
        munge_sumstats_lambda_gc=munge_sumstats_lambda_gc,
        munge_sumstats_max_chisq=munge_sumstats_max_chisq,
        ldsc_mean_chisq=ldsc_mean_chisq,
        ldsc_lambda_gc=ldsc_lambda_gc,
        ldsc_intercept=ldsc_intercept,
        ldsc_ratio=ldsc_ratio,
        ldsc_total_observed_scale_h2=ldsc_total_observed_scale_h2,
    )
