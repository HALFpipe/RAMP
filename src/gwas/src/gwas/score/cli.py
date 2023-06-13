# -*- coding: utf-8 -*-
import logging
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def main() -> None:
    mp.set_start_method("spawn")

    from gwas.compression.arr import compression_methods
    from gwas.log import logger, setup_logging
    from gwas.mem.wkspace import SharedWorkspace
    from gwas.null_model.base import NullModelCollection

    argument_parser = ArgumentParser()

    # Input files.
    argument_parser.add_argument("--vcf", nargs="+", required=True)
    argument_parser.add_argument("--tri", nargs="+", required=False, default=list())
    argument_parser.add_argument("--phenotypes", nargs="+", required=True)
    argument_parser.add_argument("--covariates", nargs="+", required=True)

    # Output files.
    argument_parser.add_argument("--output-directory", required=True)

    # Data processing options.
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
        default="ml",
    )
    argument_parser.add_argument(
        "--compression-method",
        required=False,
        choices=compression_methods.keys(),
        default="blosc2_zstd_bitshuffle",
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

    # Program options.
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("--mem-gb", type=float)
    argument_parser.add_argument("--num-threads", type=int, default=mp.cpu_count())

    arguments = argument_parser.parse_args()

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
