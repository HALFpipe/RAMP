# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(description="Generate Manhattan & QQ Plots")

    group = parser.add_argument_group(
        "paths", description="Input and output directories"
    )
    group.add_argument(
        "--input-directory",
        "--workdir",
        "--wd",
        type=str,
        nargs="+",
        action="extend",
        required=True,
        help="at least two input directories containing scores and metadata",
    )
    args = parser.parse_args()

    return args


args = parse_args()

input_dirs = args.input_directory


def verify(dirs: List):
    for dir in dirs:
        if not Path(dir).is_dir():
            raise NotADirectoryError("specified input directory is not found")

        score_files = list(Path(dir).glob("*.b2array"))
        metadata_files = list(Path(dir).glob(".axis-metadata.pkl.zst"))

        if len(score_files) != 23:
            raise ValueError(
                f"Expected 23 score files in '{dir}', found {len(score_files)}"
            )
        if len(metadata_files) != 23:
            raise ValueError(
                f"Expected 23 score files in '{dir}', found {len(score_files)}"
            )
