# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gwas.compression.arr.base import FileArrayReader

from ..compression.arr.base import FileArray
from ..log import logger
from ..summary import SummaryCollection
from ..utils import IterationOrder, chromosomes_list, make_pool_or_null_context


@dataclass(frozen=True)
class ScoreFile:
    chromosome: int | str
    path: str | Path
    variant_count: int


@dataclass(frozen=True)
class Phenotype:
    name: str
    variable_collection_name: str
    u_stat_index: int
    v_stat_index: int


def get_chromosome_metadata(
    input_directory: Path, chromosome: int | str
) -> tuple[ScoreFile, pd.DataFrame, list[str]] | None:
    score_path = input_directory / f"chr{chromosome}.score.b2array"
    if not score_path.exists():
        message = f"Missing score file for chromosome {chromosome} at {score_path}"
        if chromosome == "X":
            logger.warning(message)
        else:
            raise ValueError(message)

    array_proxy: FileArrayReader = FileArray.from_file(score_path, np.float64)
    row_metadata, column_metadata = array_proxy.axis_metadata

    if row_metadata is None or column_metadata is None:
        message = f"Missing axis metadata for chromosome {chromosome}"
        if chromosome == "X":
            logger.warning(message)
            return None
        else:
            raise ValueError(message)

    chromosome_variant_metadata = pd.DataFrame(row_metadata)
    chromosome_column_names: list[str] = [str(p) for p in column_metadata]
    score_file = ScoreFile(
        chromosome=chromosome,
        path=score_path,
        variant_count=len(chromosome_variant_metadata.index),
    )

    return score_file, chromosome_variant_metadata, chromosome_column_names


def get_metadata(
    input_directory: Path, num_threads: int = 1
) -> tuple[list[str], list[ScoreFile], pd.DataFrame]:
    score_files: list[ScoreFile] = []
    column_names: list[str] | None = None
    variant_metadata: list[pd.DataFrame] = list()

    get = partial(get_chromosome_metadata, input_directory)

    chromosomes = chromosomes_list()
    pool, iterator = make_pool_or_null_context(
        chromosomes,
        get,
        num_threads=num_threads,
        chunksize=None,
        iteration_order=IterationOrder.ORDERED,
    )

    with pool:
        for result in tqdm(iterator, desc="loading metadata", unit="chromosomes"):
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
    return column_names, score_files, pd.concat(variant_metadata)


def get_variable_collection_names(
    input_directory: Path,
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
    input_directory: Path, phenotype_names: list[str], num_threads: int = 1
) -> tuple[list[Phenotype], list[ScoreFile], pd.DataFrame]:
    """
    Verifies that each directory contains either all required .b2array score files
    or all axis_metadata.pkl.zst metadata files, for chromosomes 1-22 to X.

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
        input_directory, num_threads=num_threads
    )

    for column in [
        "chromosome_int",
        "reference_allele",
        "alternate_allele",
        "format_str",
    ]:
        variant_metadata[column] = variant_metadata[column].astype("category")

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
