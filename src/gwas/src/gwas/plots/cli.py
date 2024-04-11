# -*- coding: utf-8 -*-
import argparse
import logging
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from gwas.plots.helpers import (
    check_existing_files,
    filter_rois,
    generate_and_save_manhattan_plot,
    resolve_chromosomes,
)
from gwas.plots.worker import create_dataframe_all_chr


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
        nargs=1,
        required=True,
        help="Insert path containing the chromosome score b2array data and metadata",
    )
    group.add_argument(
        "--output_directory",
        type=str,
        help="Insert path to output directory for dataframes and plots",
        required=True,
    )

    parser.add_argument(
        "--csvroi_file",
        help="Insert path to the csv rois file",
        default=None,
    )

    parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    parser.add_argument(
        "--save_df",
        help=(
            "If set the generated dataframe for a phenotype will be saved in the"
            "specified output directory. Useful for debugging."
        ),
        type=bool,
        default=False,
    )
    parser.add_argument("--num-threads", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    from gwas.log import logger, setup_logging

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    input_dir = args.input_directory
    resolved_chrs = resolve_chromosomes(input_dir, logger=logger)

    setup_logging(level=args.log_level, log_path=output_directory)

    labels_list = []
    if args.csvroi_file:
        labels_list = filter_rois(csv_path=args.csvroi_file)

    for label in tqdm(range(len(labels_list)), desc="Processing labels"):
        if check_existing_files(directory=output_directory, label=label):
            continue
        cur_df = create_dataframe_all_chr(
            chromosome_list=resolved_chrs,
            label=label,
            cpu_count=args.num_threads,
        )
        if args.save_df:
            folder_name = "integramoods_gwas_dfs"
            folder_path = output_directory / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)

            filename = f"integramoods_gwas_{label}_df.pkl"
            file_path = folder_path / filename
            cur_df.to_pickle(path=file_path)

        generate_and_save_manhattan_plot(
            dataframe=cur_df, directory=output_directory, label=label
        )


if __name__ == "__main__":
    main()
