import json
from argparse import Namespace
from dataclasses import asdict

import polars as pl
from pyarrow import parquet as pq
from upath import UPath

from ...log import logger
from ..base import Job
from ..index import parse
from .genomic_sem import run_genomic_sem_munge
from .ldsc import parse_logs, run_ldsc
from .legends import join_legends_data_frame
from .metal import run_metal
from .plot import plot_post_meta, plot_pre_meta


def make_output_path(output_directory: UPath, job: Job, tags: dict[str, str]) -> UPath:
    for key in ["population", "age", "feature", "taskcontrast"]:
        if key not in tags:
            continue
        output_directory = output_directory / f"{key}-{tags[key]}"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / job.name
    return output_path


def worker(job: Job, output_directory: UPath, arguments: Namespace) -> None:
    cache_path = output_directory / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    tags = dict(parse(job.name))
    output_directory = output_directory / "worker-outputs"

    data_directory: UPath | None = None
    if arguments.data_directory is not None:
        data_directory = UPath(arguments.data_directory)

    metal_log, table, summaries = run_metal(job, data_directory, arguments.num_threads)

    data_frame = pl.from_arrow(table)
    if not isinstance(data_frame, pl.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(data_frame)}")

    data_frame, reference_population = join_legends_data_frame(
        cache_path, data_frame, tags["population"]
    )

    output_path = make_output_path(output_directory / "munge", job, tags)
    ldsc_munge_sumstats_log, ldsc_logs = run_ldsc(cache_path, data_frame, output_path)
    ldsc_output = parse_logs(ldsc_munge_sumstats_log, ldsc_logs)

    genomic_sem_munge_log = run_genomic_sem_munge(cache_path, data_frame, output_path)

    # plot
    output_path = make_output_path(output_directory / "plot", job, tags)
    plot_pre_meta(job, summaries, output_path.with_suffix(".pre_meta.png"))
    plot_post_meta(
        job,
        summaries,
        data_frame,
        reference_population,
        ldsc_output,
        output_path.with_suffix(".post_meta.png"),
    )

    # save to parquet
    table = data_frame.to_arrow()
    summaries_json = json.dumps(summaries)
    ldsc_output_json = json.dumps(asdict(ldsc_output))
    ldsc_logs_json = json.dumps(ldsc_logs)

    output_path = make_output_path(output_directory / "parquet", job, tags)
    parquet_path = output_path.with_suffix(".parquet")
    logger.debug(f"Writing {parquet_path}")
    with pq.ParquetWriter(
        parquet_path, table.schema, compression="zstd"
    ) as parquet_writer:
        parquet_writer.write_table(table)
        parquet_writer.add_key_value_metadata(
            key_value_metadata=dict(
                summaries=summaries_json,
                metal_log=metal_log,
                ldsc_munge_sumstats_log=ldsc_munge_sumstats_log,
                ldsc_logs=ldsc_logs_json,
                ldsc=ldsc_output_json,
                genomic_sem_munge_log=genomic_sem_munge_log,
            )
        )
