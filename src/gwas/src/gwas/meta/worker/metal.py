from dataclasses import asdict, dataclass
from functools import partial
from itertools import starmap
from subprocess import PIPE, STDOUT, run
from tempfile import TemporaryDirectory
from typing import IO, Iterator, NamedTuple

import numpy as np
import pyarrow as pa
from IPython.lib.pretty import pretty
from numpy import typing as npt
from pyarrow import compute as pc
from pyarrow import csv
from pyarrow import parquet as pq
from tqdm import tqdm
from upath import UPath

from ...compression.arr._write_float import write_float
from ...hg19 import offset
from ...log import logger
from ...pheno import VariableSummary
from ...plot.get import calculate_chi_squared_p_value
from ...plot.make import get_file_path, plot
from ...tools import metal
from ...utils.genetics import make_variant_mask
from ...utils.multiprocessing import make_pool_or_null_context
from ..base import Job, JobInput, Variant, marker_key

columns: list[str] = [
    "label",
    "reference_allele",
    "alternate_allele",
    "sample_count",
    "alternate_allele_frequency",
    "beta",
    "standard_error",
]


def run_metal(
    job: Job, data_directory: UPath | None, num_threads: int
) -> tuple[str, pa.Table, dict[str, dict[str, dict[str, float]]]]:
    with TemporaryDirectory() as temporary_path_str:
        temporary_path = UPath(temporary_path_str)

        input_paths: list[UPath] = [
            temporary_path / f"study-{study}_{ji.phenotype}.txt"
            for study, ji in job.inputs.items()
        ]

        iterable = list(zip(job.inputs.items(), input_paths, strict=True))
        callable = partial(map_write, data_directory)
        pool, iterator = make_pool_or_null_context(iterable, callable, num_threads)

        with pool:
            summaries: dict[str, dict[str, dict[str, float]]] = dict()
            for study, s in tqdm(iterator, unit=" " + "studies", total=len(iterable)):
                summaries[study] = asdict(s)
        logger.debug(f"Summaries are {pretty(summaries)}")

        metal_log = run(
            args=[*metal],
            cwd=temporary_path,
            check=True,
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            input=f"""VERBOSE OFF

EFFECT_PRINT_PRECISION 17
STDERR_PRINT_PRECISION 17

CUSTOMVARIABLE total_sample_count
LABEL total_sample_count as sample_count

LOGPVALUE OFF

AVERAGEFREQ ON
MINMAXFREQ ON

SCHEME STDERR

MARKERLABEL label
ALLELELABEL reference_allele alternate_allele
EFFECTLABEL beta
STDERRLABEL standard_error
FREQLABEL alternate_allele_frequency

{"\n".join(f"PROCESSFILE {input_path}" for input_path in input_paths)}

ANALYZE HETEROGENEITY

QUIT""",
        ).stdout

        return metal_log, make_table(temporary_path / "METAANALYSIS1.TBL"), summaries


def make_table(path: UPath) -> pa.Table:
    index_column_names = ["marker_name", "allele1", "allele2"]
    data_column_names = [
        "freq1",
        "freq_se",
        "min_freq",
        "max_freq",
        "effect",
        "std_err",
        "p_value",
        "direction",
        "het_isq",
        "het_chisq",
        "het_df",
        "het_p_value",
        "sample_count",
    ]
    column_names = [*index_column_names, *data_column_names]
    column_types: list[pa.DataType] = [
        pa.string(),
        pa.string(),
        pa.string(),
        pa.float64(),
        pa.float64(),
        pa.float64(),
        pa.float64(),
        pa.float64(),
        pa.float64(),
        pa.float64(),
        pa.string(),
        pa.float64(),
        pa.float64(),
        pa.uint32(),
        pa.float64(),
        pa.uint32(),
    ]

    read_options = csv.ReadOptions(
        skip_rows=1,
        column_names=column_names,
    )
    parse_options = csv.ParseOptions(delimiter="\t", quote_char=False)
    convert_options = csv.ConvertOptions(
        column_types=pa.schema(
            fields=list(starmap(pa.field, zip(column_names, column_types, strict=False)))
        )
    )

    table = csv.read_csv(
        input_file=path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    # require at least 50 percent of samples to be present
    max_sample_count = pc.max(table["sample_count"])
    table = table.filter(pc.field("sample_count") > max_sample_count.as_py() * 0.5)
    # require more than one study
    study_count: pc.Expression = pc.utf8_length(pc.field("direction"))
    table = table.filter(study_count > 1)

    # only keep strand-ambiguous markers if the range of alternate allele
    # frequencies across samples does not span 0.5
    is_ambiguous_at: pc.Expression = pc.is_in(
        pc.field("allele1"), value_set=pa.array(["a", "t"])
    ) & pc.is_in(pc.field("allele2"), value_set=pa.array(["a", "t"]))
    is_ambiguous_cg: pc.Expression = pc.is_in(
        pc.field("allele1"), value_set=pa.array(["c", "g"])
    ) & pc.is_in(pc.field("allele2"), value_set=pa.array(["c", "g"]))
    is_ambiguous: pc.Expression = is_ambiguous_at | is_ambiguous_cg

    spans_05 = (pc.field("min_freq") <= 0.5) & (pc.field("max_freq") >= 0.5)
    table = table.filter(~(is_ambiguous & spans_05))

    # sort by offset
    key_array: pa.StringArray = pa.array(
        map(marker_key, table["marker_name"].to_pylist()), type=pa.string()
    )
    indices = pc.array_sort_indices(key_array)
    table = table.take(indices)
    return table


@dataclass
class Summaries:
    beta: VariableSummary
    standard_error: VariableSummary
    r_squared: VariableSummary
    log_p_value: VariableSummary
    variant_count: int


def map_write(
    data_directory: UPath | None, mi: tuple[tuple[str, JobInput], UPath]
) -> tuple[str, Summaries]:
    (study, ji), input_path = mi

    summary_inputs: list[SummaryInput] = list()
    plot_inputs: list[PlotInput] = list()
    with input_path.open(mode="w") as phenotype_handle:
        phenotype_handle.write("\t".join(columns) + "\n")
        phenotype_handle.flush()

        for score_path_str in ji.score_paths:
            score_path = UPath(score_path_str)
            if data_directory is not None:
                score_path = data_directory / score_path

            summary_input, plot_input = load_and_write(
                ji.phenotype,
                ji.variable_collection_name,
                ji.sample_count,
                score_path,
                phenotype_handle,
            )
            summary_inputs.append(summary_input)
            plot_inputs.append(plot_input)

    chromosome_int, position, u_stat, v_stat = map(
        np.concatenate, zip(*plot_inputs, strict=True)
    )
    p_value, log_p_value = calculate_chi_squared_p_value(u_stat, v_stat)
    log_p_value_summary = VariableSummary.from_array(log_p_value)

    arrays = map(np.concatenate, zip(*summary_inputs, strict=True))
    beta_summary, standard_error_summary, r_squared_summary = map(
        VariableSummary.from_array, arrays
    )
    variant_count = sum(a.size for a, _, _ in summary_inputs)
    summaries = Summaries(
        beta_summary,
        standard_error_summary,
        r_squared_summary,
        log_p_value_summary,
        variant_count,
    )

    score_path_str = ji.score_paths[0]
    plot_directory = UPath(UPath(score_path_str).parts[0]) / "quality-control"
    if data_directory is not None:
        plot_directory = data_directory / plot_directory
    if not plot_directory.is_dir():
        logger.warning(
            f'Did not find plot directory for study "{study}" at "{plot_directory}"'
        )
    else:
        if not get_file_path(plot_directory, ji.phenotype).is_file() and u_stat.size > 0:
            position = offset[chromosome_int - 1] + position
            plot(
                plot_directory,
                ji.phenotype,
                chromosome_int,
                position,
                p_value,
                log_p_value,
            )

    return study, summaries


def iterate_row_prefixes(
    chunked_arrays: tuple[
        pa.ChunkedArray, pa.ChunkedArray, pa.ChunkedArray, pa.ChunkedArray
    ],
    sample_count: int,
) -> Iterator[bytes]:
    for arrays in zip(*(c.iterchunks() for c in chunked_arrays), strict=False):
        for row in zip(*arrays, strict=True):
            label = Variant(*row).id_str
            _, _, reference_allele, alternate_allele = row
            prefix = f"{label}\t{reference_allele}\t{alternate_allele}\t{sample_count}"
            yield prefix.encode()


class PlotInput(NamedTuple):
    chromosome_int: npt.NDArray[np.uint8]
    position: npt.NDArray[np.int64]
    u_stat: npt.NDArray[np.float64]
    v_stat: npt.NDArray[np.float64]


class SummaryInput(NamedTuple):
    beta: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    r_squared: npt.NDArray[np.float64]


def load_and_write(
    phenotype: str,
    variable_collection_name: str,
    sample_count: int,
    score_path: UPath,
    phenotype_handle: IO[str],
    use_threads: bool = True,
) -> tuple[SummaryInput, PlotInput]:
    columns = [
        "chromosome_int",
        "position",
        "reference_allele",
        "alternate_allele",
        "r_squared",
        f"{variable_collection_name}_alternate_allele_frequency",
        f"{phenotype}_stat-u",
        f"{phenotype}_stat-v",
    ]
    parquet_file = pq.ParquetFile(score_path)
    table = parquet_file.read(columns=columns, use_threads=use_threads)
    (_, _, _, _, r_squared_array, alternate_allele_frequency_array, u_stat_array, _) = (
        table.columns
    )

    # apply cutoffs
    minor_allele_count_cutoff = 5
    ploidy = 2
    minor_allele_frequency_cutoff = minor_allele_count_cutoff / (ploidy * sample_count)
    mask = make_variant_mask(
        allele_frequencies=alternate_allele_frequency_array.to_pandas(),
        r_squared=r_squared_array.to_pandas(),
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
        r_squared_cutoff=0.6,
    )

    # remove invalid statistics
    u_stat: npt.NDArray[np.float64] = u_stat_array.to_numpy()
    np.logical_and(np.isfinite(u_stat), mask, out=mask)
    np.logical_and(np.logical_not(np.isclose(u_stat, 0)), mask, out=mask)
    (variant_indices,) = np.nonzero(mask)

    # extract filtered arrays
    filtered_table = table.take(variant_indices)
    (
        chromosome_int_array,
        position_array,
        reference_allele_array,
        alternate_allele_array,
        r_squared_array,
        alternate_allele_frequency_array,
        u_stat_array,
        v_stat_array,
    ) = filtered_table.columns

    # write to file
    row_prefix_iterator = iterate_row_prefixes(
        (
            chromosome_int_array,
            position_array,
            reference_allele_array,
            alternate_allele_array,
        ),
        sample_count,
    )
    alternate_allele_frequency = alternate_allele_frequency_array.to_numpy()

    u_stat = u_stat_array.to_numpy()
    v_stat: npt.NDArray[np.float64] = v_stat_array.to_numpy()

    beta = u_stat / v_stat
    standard_error = np.power(v_stat, -0.5)

    write_float(
        row_prefix_iterator,
        [v[:, np.newaxis] for v in [alternate_allele_frequency, beta, standard_error]],
        phenotype_handle.fileno(),
    )

    # prepare arrays
    r_squared = r_squared_array.to_numpy()
    summary_inputs = SummaryInput(beta, standard_error, r_squared)

    chromosome_int = filtered_table["chromosome_int"].to_numpy()
    position = filtered_table["position"].to_numpy()
    plot_inputs = PlotInput(chromosome_int, position, u_stat, v_stat)

    return (summary_inputs, plot_inputs)
