# -*- coding: utf-8 -*-
from multiprocessing import Pool
from pathlib import Path

import blosc2
import pandas as pd

from gwas.plots.helpers import chi2_pvalue, find_phenotype_index, load_metadata


def create_dataframe_single_chr(
    score_path: str | Path, axis_metadata_path: str | Path, phenotype_label: str
) -> pd.DataFrame:
    """
    Generates a pandas DataFrame for genetic variants of a single chromosome.

    Args:
        args: A tuple containing three elements:
            - score_path (Path): The path to the chromosome's score file.
            - axis_metadata_path (Path): The path to the chromosome's axis metadata file.
            - phenotype_label (str): The label of the phenotype to calculate p-values
            for.

    Returns:
        pd.DataFrame: A DataFrame with columns ['SNP', 'CHR', 'BP', 'P'], containing the
        SNP identifier, chromosome number, base pair position, and p-value for each
        variant.

    Raises:
        ValueError: If the length of the b2 array does not match the length of the
        metadata.
    """

    axis_metadata = load_metadata(Path(axis_metadata_path))
    metadata = pd.DataFrame(axis_metadata[0])
    phenotypes_list = axis_metadata[1]

    b2_array = blosc2.open(urlpath=score_path)

    if b2_array.shape[0] != len(metadata.index):
        raise ValueError(
            "The length of b2_array does not match the length of the metadata."
        )

    u_stat_idx, v_stat_idx = find_phenotype_index(
        phenotype_label=phenotype_label, phenotypes_list=phenotypes_list
    )

    variant_count = len(metadata.index)

    stats = b2_array[
        :, u_stat_idx : v_stat_idx + 1
    ]  # +1 since slicing range is exclusive

    chrom = metadata.chromosome_int
    pos = metadata.position

    p_values = chi2_pvalue(ustat=stats[:, 0], vstat=stats[:, 1])
    chr_data = {
        "SNP": ["rs0000000"] * variant_count,
        "CHR": chrom,
        "BP": pos,
        "P": p_values,
    }

    return pd.DataFrame(chr_data)


def create_dataframe_all_chr(
    score_path: str | Path, metadata_path: str | Path, label: str, cpu_count: int
) -> pd.DataFrame:
    """Generate dataframes with information needed for manhattan plots / qq plots
     for each chromosome which will be concatenated together.
    Uses multiprocessing to utilize cpu and share the workload.
    """
    chromosomes = list(range(1, 23)) + ["X"]
    tasks = [
        (
            Path(score_path) / f"chr{chromosome}.score.b2array",
            Path(metadata_path) / f"chr{chromosome}.score.axis-metadata.pkl.zst",
            label,
        )
        for chromosome in chromosomes
    ]
    dfs = []

    with Pool(processes=cpu_count) as pool:
        dfs = pool.starmap(create_dataframe_single_chr, tasks)

    final_df = pd.concat(dfs)
    return final_df
