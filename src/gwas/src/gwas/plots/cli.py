# -*- coding: utf-8 -*-
import argparse
import logging
import multiprocessing as mp
from pathlib import Path

from gwas.plots.helpers import (
    check_existing_files,
    enrich_phenotype_names,
    generate_and_save_manhattan_plot,
    resolve_chromosomes,
    verify_metadata,
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
    group.add_argument(
        "--include-phenotype-list",
        type=str,
        nargs=1,
        help=(
            "Include only phenotypes that are present within this list."
            "Each new line should be a new phenotype"
        ),
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

    verify_metadata(input_dir)
    resolved_chrs = resolve_chromosomes(input_dir, logger=logger)

    phenotype_stat_names = enrich_phenotype_names(args.include_phenotype_list)

    setup_logging(level=args.log_level, log_path=output_directory)

    for phenotype, stats in phenotype_stat_names.items():
        if check_existing_files(directory=output_directory, label=phenotype):
            continue
        cur_df = create_dataframe_all_chr(
            chromosome_list=resolved_chrs,
            pheno_stats=stats,
            cpu_count=args.num_threads,
        )
        if args.save_df:
            folder_name = "integramoods_gwas_dfs"
            folder_path = output_directory / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)

            filename = f"integramoods_gwas_{phenotype}_df.pkl"
            file_path = folder_path / filename
            cur_df.to_pickle(path=file_path)

        generate_and_save_manhattan_plot(
            dataframe=cur_df, directory=output_directory, label=phenotype
        )


if __name__ == "__main__":
    main()
