# -*- coding: utf-8 -*-
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from tqdm.auto import tqdm

from ..compression.pipe import CompressedBytesReader, CompressedTextReader
from ..log import logger
from ..utils import chromosomes_list


def load_axis_metadata(metadata_path: Path) -> Sequence[pd.DataFrame | pd.Series]:
    with CompressedBytesReader(metadata_path) as file_handle:
        metadata = pickle.load(file_handle)
    assert isinstance(metadata, list)
    return metadata


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


def get_metadata(
    input_directory: Path,
) -> tuple[list[str], list[ScoreFile], pd.DataFrame]:
    score_files: list[ScoreFile] = []
    column_names: list[str] | None = None
    variant_metadata: list[pd.DataFrame] = list()

    chromosomes = chromosomes_list()
    for chromosome in tqdm(chromosomes, desc="loading metadata", unit="chromosomes"):
        score_path = input_directory / f"chr{chromosome}.score.b2array"
        if not score_path.exists():
            message = f"Missing chromosome {chromosome} score file: {score_path}"
            if chromosome == "X":
                logger.warning(message)
            else:
                raise ValueError(message)

        axis_metadata_path = (
            input_directory / f"chr{chromosome}.score.axis-metadata.pkl.zst"
        )
        if not axis_metadata_path.exists():
            message = (
                f"Missing chromosome {chromosome} axis metadata file:"
                f" {axis_metadata_path}"
            )
            if chromosome == "X":
                logger.warning(message)
            else:
                raise ValueError(message)

        axis_metadata = load_axis_metadata(axis_metadata_path)
        chromosome_variant_metadata = pd.DataFrame(axis_metadata[0])

        chromosome_column_names: list[str] = [str(p) for p in axis_metadata[1]]
        score_file = ScoreFile(
            chromosome=chromosome,
            path=score_path,
            variant_count=len(chromosome_variant_metadata.index),
        )

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
    with CompressedTextReader(metadata_path) as file_handle:
        metadata = yaml.load(file_handle, Loader=yaml.CBaseLoader)
    variable_collection_names: dict[str, str] = dict()
    for chunk in metadata["chunks"].values():
        for variable_collection_name, variable_collection in chunk.items():
            for phenotype_name in variable_collection["phenotypes"].keys():
                variable_collection_names[phenotype_name] = variable_collection_name
    return variable_collection_names


def resolve_score_files(
    input_directory: Path,
    phenotype_names: list[str],
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

    column_names, score_files, variant_metadata = get_metadata(input_directory)

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
