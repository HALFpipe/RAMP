import logging
import sys
from argparse import ArgumentParser, Namespace
from typing import Literal

from upath import UPath


def parse_arguments(argv: list[str]) -> Namespace:
    """Parses command-line arguments"""
    argument_parser = ArgumentParser(description="Generate Manhattan & QQ Plots")

    paths_group = argument_parser.add_argument_group(
        "paths", description="Input and output directories"
    )
    paths_group.add_argument(
        "--input-directory",
        "--workdir",
        "--wd",
        type=str,
        required=True,
        help="Insert path containing the chromosome score files and metadata",
    )
    paths_group.add_argument(
        "--output-directory",
        type=str,
        help="Insert path to output directory for dataframes and plots",
        required=True,
    )
    paths_group.add_argument(
        "--phenotype-list",
        type=str,
        help=(
            "Include only phenotypes that are present within this list."
            "Each new line should be a new phenotype"
        ),
        required=True,
    )

    filter_group = argument_parser.add_argument_group(
        "filter", description="Filter data based on these criteria"
    )
    filter_group.add_argument(
        "--minor-allele-frequency-cutoff",
        "--maf",
        required=False,
        type=float,
        default=0.005,
    )
    filter_group.add_argument(
        "--r-squared-cutoff",
        "--r2",
        required=False,
        type=float,
        default=0.5,
    )

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--num-threads", type=int, default=None)
    argument_parser.add_argument("--mem-gb", type=float)

    return argument_parser.parse_args(argv)


def main() -> None:
    run(sys.argv[1:])


def run(argv: list[str], error_action: Literal["raise", "ignore"] = "ignore") -> None:
    arguments = parse_arguments(argv)

    from ..log import logger, setup_logging

    output_directory = UPath(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(level=arguments.log_level, path=output_directory)

    from ..utils.threads import apply_num_threads, cpu_count

    if arguments.num_threads is None:
        arguments.num_threads = cpu_count()
    apply_num_threads(arguments.num_threads)

    from ..mem.wkspace import SharedWorkspace

    if arguments.mem_gb is not None:
        size = int(arguments.mem_gb * 2**30)

    with SharedWorkspace.create(size=size) as sw:
        try:
            from .command import plot

            plot(arguments, output_directory, sw)
        except Exception as e:
            logger.exception("Exception: %s", e, exc_info=True)
            if arguments.debug:
                import pdb

                pdb.post_mortem()
            if error_action == "raise":
                raise e
