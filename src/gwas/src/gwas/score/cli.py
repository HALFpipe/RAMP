# -*- coding: utf-8 -*-
import logging
import multiprocessing as mp
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

import numpy as np


def main() -> None:
    mp.set_start_method("spawn")
    run(sys.argv[1:])


def parse_arguments(argv: list[str]) -> Namespace:
    from gwas.compression.arr.base import compression_methods
    from gwas.null_model.base import NullModelCollection
    from gwas.utils import chromosomes_set

    argument_parser = ArgumentParser()

    # Input files
    argument_parser.add_argument("--vcf", nargs="+", required=True)
    argument_parser.add_argument("--tri", nargs="+", required=False, default=list())
    argument_parser.add_argument("--phenotypes", nargs="+", required=True)
    argument_parser.add_argument("--covariates", nargs="+", required=True)

    # Output files
    argument_parser.add_argument("--output-directory", required=True)

    # Data processing options
    chromosomes_list = [str(c) for c in chromosomes_set()]
    argument_parser.add_argument(
        "--chromosome",
        choices=chromosomes_list,
        nargs="+",
        required=False,
        default=[*chromosomes_list],
    )
    argument_parser.add_argument(
        "--kinship-minor-allele-frequency-cutoff",
        "--kin-maf",
        required=False,
        type=float,
        default=0.05,
    )
    argument_parser.add_argument(
        "--kinship-r-squared-cutoff",
        "--kinship-r2",
        required=False,
        type=float,
        default=-np.inf,
    )
    argument_parser.add_argument(
        "--score-minor-allele-frequency-cutoff",
        "--score-maf",
        required=False,
        type=float,
        default=-np.inf,
    )
    argument_parser.add_argument(
        "--score-r-squared-cutoff",
        "--score-r2",
        required=False,
        type=float,
        default=-np.inf,
    )
    argument_parser.add_argument(
        "--null-model-method",
        required=False,
        choices=NullModelCollection.methods,
        default="fastlmm",
    )
    argument_parser.add_argument(
        "--compression-method",
        required=False,
        choices=compression_methods.keys(),
        default="zstd_text",
    )
    argument_parser.add_argument(
        "--add-principal-components",
        required=False,
        type=int,
        default=0,
    )
    argument_parser.add_argument(
        "--missing-value-strategy",
        required=False,
        choices=["complete_samples", "listwise_deletion"],
        default="listwise_deletion",
    )

    # Program options
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("--mem-gb", type=float)
    argument_parser.add_argument("--num-threads", type=int, default=mp.cpu_count())

    return argument_parser.parse_args(argv)


def run(argv: list[str], error_action: Literal["raise", "ignore"] = "ignore") -> None:
    arguments = parse_arguments(argv)

    from gwas.log import logger, setup_logging
    from gwas.mem.wkspace import SharedWorkspace

    setup_logging(level=arguments.log_level)
    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    size: int | None = None
    if arguments.mem_gb is not None:
        size = int(arguments.mem_gb * 2**30)

    with SharedWorkspace.create(size=size) as sw:
        try:
            from .command import GwasCommand

            GwasCommand(arguments, output_directory, sw).run()
        except Exception as e:
            logger.exception("Exception: %s", e, exc_info=True)
            if arguments.debug:
                import pdb

                pdb.post_mortem()
            if error_action == "raise":
                raise e
