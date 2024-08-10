import logging
import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Literal


def parse_arguments(argv: list[str]) -> Namespace:
    """Parses command-line arguments"""
    argument_parser = ArgumentParser(description="Generate Manhattan & QQ Plots")

    argument_parser.add_argument("path")
    argument_parser.add_argument("--compression-method")

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--num-threads", type=int, default=mp.cpu_count())

    return argument_parser.parse_args(argv)


def main() -> None:
    run(sys.argv[1:])


def run(argv: list[str], error_action: Literal["raise", "ignore"] = "ignore") -> None:
    arguments = parse_arguments(argv)

    from gwas.log import logger, setup_logging

    setup_logging(level=arguments.log_level)

    os.environ["NUMEXPR_MAX_THREADS"] = str(arguments.num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(arguments.num_threads)

    try:
        from .command import convert

        convert(arguments)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
        if error_action == "raise":
            raise e
