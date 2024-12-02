from shutil import copyfile
from subprocess import PIPE, STDOUT, run
from tempfile import TemporaryDirectory

import polars as pl
from upath import UPath

from ...tools import r_genomicsem

conf_path = str(
    UPath(__file__).absolute().parent.joinpath("layout", "config", "{}.json")
)


def run_genomic_sem_munge(
    cache_path: UPath, data_frame: pl.DataFrame, output_path: UPath
) -> str:
    snplist_path = cache_path / "w_hm3.snplist"

    with TemporaryDirectory() as temporary_path_str:
        temporary_path = UPath(temporary_path_str)

        sumstats_path = temporary_path / "sumstats.txt"

        rsid = pl.col("rsid")
        allele1 = pl.col("allele1")
        allele2 = pl.col("allele2")
        freq1 = pl.col("freq1")
        beta = pl.col("effect").alias("beta")
        se = pl.col("std_err").alias("se")
        p_value = pl.col("p_value")
        n = pl.col("sample_count").alias("n")

        data_frame.select(rsid, allele1, allele2, freq1, beta, se, p_value, n).write_csv(
            sumstats_path, separator="\t", quote_style="never"
        )

        run(
            args=[*r_genomicsem],
            cwd=temporary_path,
            check=True,
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            input=f"""GenomicSEM::munge(
        files = c("{sumstats_path}"),
        hm3 = "{snplist_path}",
        trait.names = c("trait"),
)
""",
        )

        with (temporary_path / "trait_munge.log").open() as file_handle:
            genomicsem_munge_log = file_handle.read()
        copyfile(
            temporary_path / "trait.sumstats.gz",
            output_path.with_suffix(".genomic-sem.sumstats.gz"),
        )

        return genomicsem_munge_log
