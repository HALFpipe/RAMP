import logging
import sys
from argparse import ArgumentParser, Namespace
from typing import Literal

from upath import UPath


def parse_arguments(argv: list[str]) -> Namespace:
    from ..compression.arr.base import compression_methods
    from ..defaults import (
        default_kinship_minor_allele_frequency_cutoff,
        default_kinship_r_squared_cutoff,
        default_score_minor_allele_frequency_cutoff,
        default_score_r_squared_cutoff,
    )
    from ..null_model.base import NullModelCollection
    from ..utils.genetics import chromosomes_set
    from ..utils.threads import cpu_count
    from ..vcf.base import Engine

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
    )
    argument_parser.add_argument(
        "--kinship-minor-allele-frequency-cutoff",
        "--kin-maf",
        required=False,
        type=float,
        default=default_kinship_minor_allele_frequency_cutoff,
    )
    argument_parser.add_argument(
        "--kinship-r-squared-cutoff",
        "--kinship-r2",
        required=False,
        type=float,
        default=default_kinship_r_squared_cutoff,
    )
    argument_parser.add_argument(
        "--score-minor-allele-frequency-cutoff",
        "--score-maf",
        required=False,
        type=float,
        default=default_score_minor_allele_frequency_cutoff,
    )
    argument_parser.add_argument(
        "--score-r-squared-cutoff",
        "--score-r2",
        required=False,
        type=float,
        default=default_score_r_squared_cutoff,
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
        default="zstd_high_text",
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

    argument_parser.add_argument("--mem-gb", type=float)
    argument_parser.add_argument("--num-threads", type=int, default=cpu_count())
    argument_parser.add_argument(
        "--vcf-engine",
        choices=Engine,
        type=Engine.__getitem__,
        default=Engine.cpp,
    )

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("--jax-trace", action="store_true", default=False)
    argument_parser.add_argument(
        "--dump-traceback-later", action="store_true", default=False
    )

    return argument_parser.parse_args(argv)


def main() -> None:
    run(sys.argv[1:])


def run(argv: list[str], error_action: Literal["raise", "ignore"] = "ignore") -> None:
    arguments = parse_arguments(argv)

    from ..log import logger, setup_logging

    output_directory = UPath(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(level=arguments.log_level, path=output_directory)

    from ..utils.threads import apply_num_threads

    apply_num_threads(arguments.num_threads, arguments.dump_traceback_later)

    from ..mem.wkspace import SharedWorkspace

    size: int | None = None
    if arguments.mem_gb is not None:
        size = int(arguments.mem_gb * 2**30 * 0.9)

    with SharedWorkspace.create(size=size) as sw:
        try:
            from .command import GwasCommand

            GwasCommand(arguments, output_directory, sw).run()
        except Exception as e:
            logger.exception("Exception: %s", e, exc_info=True, stack_info=True)
            if arguments.debug:
                import pdb

                pdb.post_mortem()
            if error_action == "raise":
                raise e
