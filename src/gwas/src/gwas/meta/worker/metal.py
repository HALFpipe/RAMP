from dataclasses import asdict, dataclass
from functools import partial
from itertools import starmap
from subprocess import PIPE, STDOUT, run
from tempfile import TemporaryDirectory
from typing import IO, Iterator

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
from ...log import logger
from ...pheno import VariableSummary
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

    # require at least 75 percent of samples to be present
    max_sample_count = pc.max(table["sample_count"])
    table = table.filter(pc.field("sample_count") > max_sample_count.as_py() * 0.75)

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
    variant_count: int


def map_write(
    data_directory: UPath | None, mi: tuple[tuple[str, JobInput], UPath]
) -> tuple[str, Summaries]:
    (study, ji), input_path = mi

    summary_values: list[
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ] = list()
    with input_path.open(mode="w") as phenotype_handle:
        phenotype_handle.write("\t".join(columns) + "\n")
        phenotype_handle.flush()

        for score_path_str in ji.score_paths:
            score_path = UPath(score_path_str)
            if data_directory is not None:
                score_path = data_directory / score_path
            summary_values.append(
                write(
                    ji.phenotype,
                    ji.variable_collection_name,
                    ji.sample_count,
                    score_path,
                    phenotype_handle,
                )
            )

    beta, standard_error, r_squared = map(
        VariableSummary.from_array,
        map(np.concatenate, zip(*summary_values, strict=False)),
    )
    variant_count = sum(a.size for a, _, _ in summary_values)
    return study, Summaries(beta, standard_error, r_squared, variant_count)


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


def write(
    phenotype: str,
    variable_collection_name: str,
    sample_count: int,
    score_path: UPath,
    phenotype_handle: IO[str],
    use_threads: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    alternate_allele_frequency_column = (
        f"{variable_collection_name}_alternate_allele_frequency"
    )
    columns = [
        "chromosome_int",
        "position",
        "reference_allele",
        "alternate_allele",
        "r_squared",
        alternate_allele_frequency_column,
        f"{phenotype}_stat-u",
        f"{phenotype}_stat-v",
    ]
    parquet_file = pq.ParquetFile(score_path)
    table = parquet_file.read(columns=columns, use_threads=use_threads)

    u_stat_array = table[f"{phenotype}_stat-u"]
    r_squared_array = table["r_squared"]
    alternate_allele_frequency_array = table[alternate_allele_frequency_column]

    # apply cutoffs
    mask = make_variant_mask(
        allele_frequencies=alternate_allele_frequency_array.to_pandas(),
        r_squared=r_squared_array.to_pandas(),
        minor_allele_frequency_cutoff=0.01,
        r_squared_cutoff=0.6,
    )

    # remove invalid statistics
    u_stat: npt.NDArray[np.float64] = u_stat_array.to_numpy()
    np.logical_and(np.isfinite(u_stat), mask, out=mask)
    np.logical_and(np.logical_not(np.isclose(u_stat, 0)), mask, out=mask)

    (variant_indices,) = np.nonzero(mask)
    table = table.take(variant_indices)

    (
        chromosome_int_array,
        position_array,
        reference_allele_array,
        alternate_allele_array,
        r_squared_array,
        alternate_allele_frequency_array,
        u_stat_array,
        v_stat_array,
    ) = table.columns

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

    return beta, standard_error, r_squared_array.to_numpy()
