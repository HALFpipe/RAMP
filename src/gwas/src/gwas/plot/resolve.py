from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from upath import UPath

from ..compression.arr.base import FileArray, FileArrayReader
from ..log import logger
from ..mem.data_frame import SharedDataFrame, concat
from ..mem.wkspace import SharedWorkspace
from ..summary import SummaryCollection
from ..utils.genetics import chromosomes_list
from ..utils.multiprocessing import IterationOrder, make_pool_or_null_context
from ..vcf.base import VCFFile, base_allele_frequency_columns


@dataclass(frozen=True)
class ScoreFile:
    chromosome: int | str
    reader: FileArrayReader[np.float64]
    variant_count: int


@dataclass(frozen=True)
class Phenotype:
    name: str
    variable_collection_name: str
    u_stat_index: int
    v_stat_index: int


def get_chromosome_metadata(
    input_directory: UPath,
    sw: SharedWorkspace,
    chromosome: int | str,
    num_threads: int = 1,
) -> tuple[ScoreFile, SharedDataFrame, list[str]] | None:
    base_path = input_directory / f"chr{chromosome}.score"
    array_proxy = FileArray.from_file(base_path, np.float64, num_threads)
    if not array_proxy.file_path.exists():
        message = f"Missing score file for chromosome {chromosome} at {base_path}"
        if chromosome == "X":
            logger.warning(message)
        else:
            raise ValueError(message)

    row_metadata, column_names = array_proxy.row_metadata, array_proxy.column_names

    if row_metadata is None or column_names is None:
        message = f"Missing axis metadata for chromosome {chromosome}"
        if chromosome == "X":
            logger.warning(message)
            return None
        else:
            raise ValueError(message)

    score_file = ScoreFile(
        chromosome=chromosome,
        reader=array_proxy,
        variant_count=len(row_metadata.index),
    )

    update_data_frame_types(row_metadata)
    shared_row_metadata = SharedDataFrame.from_pandas(row_metadata, sw)

    return score_file, shared_row_metadata, column_names


def update_data_frame_types(data_frame: pd.DataFrame) -> None:
    VCFFile.update_data_frame_types(data_frame)

    if "is_imputed" in data_frame.columns:
        data_frame["is_imputed"] = data_frame["is_imputed"].astype(np.bool_)

    float_columns: set[str] = {"r_squared"}
    for a in base_allele_frequency_columns:
        for column in data_frame.columns:
            if not column.endswith(a):
                continue
            float_columns.add(column)
    for column in float_columns:
        if column not in data_frame.columns:
            continue
        data_frame[column] = data_frame[column].astype(np.float64)


def get_metadata(
    input_directory: UPath, sw: SharedWorkspace, num_threads: int = 1
) -> tuple[list[str], list[ScoreFile], SharedDataFrame]:
    score_files: list[ScoreFile] = []
    column_names: list[str] | None = None
    variant_metadata: list[SharedDataFrame] = list()

    get = partial(get_chromosome_metadata, input_directory, sw, num_threads=num_threads)

    chromosomes = chromosomes_list()
    pool, iterator = make_pool_or_null_context(
        chromosomes,
        get,
        num_threads=num_threads,
        chunksize=None,
        iteration_order=IterationOrder.ORDERED,
    )

    with pool:
        for result in tqdm(
            iterator, desc="loading metadata", unit="chromosomes", total=len(chromosomes)
        ):
            if result is None:
                continue
            score_file, chromosome_variant_metadata, chromosome_column_names = result

            if column_names is None:
                column_names = chromosome_column_names
            elif column_names != chromosome_column_names:
                raise ValueError("Column names do not match between chromosomes")

            score_files.append(score_file)
            variant_metadata.append(chromosome_variant_metadata)

    if column_names is None:
        raise ValueError("No column names were found for the score files")

    shared_data_frame = concat(variant_metadata)

    return column_names, score_files, shared_data_frame


def get_variable_collection_names(
    input_directory: UPath,
) -> dict[str, str]:
    metadata_path = input_directory / "chr1.metadata.yaml.gz"
    if not metadata_path.exists():
        raise ValueError(f"Missing metadata file: {metadata_path}")
    summary_collection = SummaryCollection.from_file(metadata_path)
    variable_collection_names: dict[str, str] = dict()
    for chunk in summary_collection.chunks.values():
        for variable_collection_name, variable_collection in chunk.items():
            for phenotype_name in variable_collection.phenotypes.keys():
                variable_collection_names[phenotype_name] = variable_collection_name
    return variable_collection_names


def resolve_score_files(
    input_directory: UPath,
    phenotype_names: list[str],
    sw: SharedWorkspace,
    num_threads: int = 1,
) -> tuple[list[Phenotype], list[ScoreFile], SharedDataFrame]:
    """
    Verifies that each directory contains either all required .score.txt.zst files
    for chromosomes 1 to 22 and optionally X.

    Parameters:
        input_dir (str): Input directory path.

    Returns:

        score_files (list[ScoreFile]): A list of all resolved score files
        where score path and metadata could be found.

    Raises:
        NotADirectoryError: If any specified path is not a directory.
        ValueError: If one of the chromosomes of 1 to 22 is missing.
        Only warns when ChrX is missing.
    """

    if not input_directory.is_dir():
        raise NotADirectoryError(f"Could not find input directory: {input_directory}")

    column_names, score_files, variant_metadata = get_metadata(
        input_directory, sw, num_threads=num_threads
    )

    variable_collection_names = get_variable_collection_names(input_directory)

    phenotypes = []
    for phenotype_name in phenotype_names:
        phenotype = Phenotype(
            name=phenotype_name,
            variable_collection_name=variable_collection_names[phenotype_name],
            u_stat_index=column_names.index(f"{phenotype_name}_stat-u"),
            v_stat_index=column_names.index(f"{phenotype_name}_stat-v"),
        )
        phenotypes.append(phenotype)

    return phenotypes, score_files, variant_metadata
