import logging
import sys
from argparse import ArgumentParser, Namespace
from typing import Literal

from upath import UPath

from .. import __version__
from ..utils.threads import cpu_count


def parse_arguments(argv: list[str]) -> Namespace:
    """Parses command-line arguments"""
    argument_parser = ArgumentParser(description="Generate Manhattan & QQ Plots")

    paths_group = argument_parser.add_argument_group(
        "paths", description="Input and output paths"
    )
    paths_group.add_argument(
        "--study",
        type=str,
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        required=True,
        help="Add a study",
    )
    paths_group.add_argument("--data-directory", type=str)
    paths_group.add_argument(
        "--output-directory",
        type=str,
        help="Where to save the output files",
        required=True,
    )

    modify_group = argument_parser.add_argument_group(
        title="modify", description="Arguments for modifying the inputs"
    )
    modify_group.add_argument(
        "--replace-phenotype",
        type=str,
        nargs=2,
        metavar=("STRING", "REPLACEMENT"),
        action="append",
        help="Replace substrings in phenotype names",
    )
    modify_group.add_argument(
        "--replace-key",
        type=str,
        nargs=2,
        metavar=("KEY", "REPLACEMENT"),
        action="append",
        help="Change key to replacement",
    )
    modify_group.add_argument(
        "--drop-key",
        type=str,
        metavar="KEY",
        action="append",
        help="Drop key from index",
    )
    modify_group.add_argument(
        "--replace-value",
        type=str,
        nargs=3,
        metavar=("KEY", "VALUE", "REPLACEMENT"),
        action="append",
        help="For the given key, change one value to another",
    )
    modify_group.add_argument(
        "--remove",
        nargs="+",
        metavar="KEY=VALUE",
        action="append",
        help="Remove all phenotypes matching the query",
    )
    modify_group.add_argument(
        "--alternative",
        type=str,
        nargs=3,
        metavar=("KEY", "VALUE", "ALTERNATIVE"),
        action="append",
        help="Allow phenotypes with an alternative value for key when selecting "
        "which phenotypes to combine in a meta-analysis",
    )
    modify_group.add_argument(
        "--id-matching",
        choices=("simple", "dbsnp"),
        default="simple",
        help="Which variant IDs to use for meta-analysis",
    )

    meta_group = argument_parser.add_argument_group(
        title="meta", description="Arguments for configuring the meta-analysis"
    )
    meta_group.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        metavar="KEY",
        help="Group phenotypes by key",
        required=True,
    )
    meta_group.set_defaults(genomic_control=False)
    meta_group.add_argument(
        "--with-genomic-control",
        dest="genomic_control",
        action="store_true",
    )
    meta_group.add_argument(
        "--minor-allele-frequency-cutoff",
        "--maf",
        required=False,
        type=float,
        default=0.01,
    )
    meta_group.add_argument(
        "--r-squared-cutoff",
        "--r2",
        required=False,
        type=float,
        default=0.6,
    )

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--num-threads", type=int, default=cpu_count())
    argument_parser.add_argument("--version", action="version", version=__version__)

    return argument_parser.parse_args(argv)


def prepare() -> None:
    run_prepare(sys.argv[1:])


def run_prepare(
    argv: list[str], error_action: Literal["raise", "ignore"] = "ignore"
) -> None:
    arguments = parse_arguments(argv)

    from ..log import logger, setup_logging

    output_directory = UPath(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(level=arguments.log_level, path=output_directory)

    from ..utils.threads import apply_num_threads

    apply_num_threads(arguments.num_threads)

    try:
        from .prepare import prepare

        prepare(arguments, output_directory)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
        if error_action == "raise":
            raise e


def worker() -> None:
    run_worker(sys.argv[1:])


def run_worker(
    argv: list[str], error_action: Literal["raise", "ignore"] = "ignore"
) -> None:
    argument_parser = ArgumentParser(description="Generate Manhattan & QQ Plots")

    argument_parser.add_argument("job")
    argument_parser.add_argument("output_directory")

    argument_parser.add_argument("--data-directory", type=str)

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--num-threads", type=int, default=cpu_count())

    arguments = argument_parser.parse_args(argv)

    from ..log import logger, setup_logging

    output_directory = UPath(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(level=arguments.log_level, path=output_directory)

    from ..utils.threads import apply_num_threads

    apply_num_threads(arguments.num_threads)

    try:
        import yaml

        from ..compression.pipe import CompressedTextReader
        from ..summary import parse_obj_as
        from .base import Job
        from .worker.base import worker

        job_path = UPath(arguments.job)
        with CompressedTextReader(job_path) as file_handle:
            job = parse_obj_as(
                Job, yaml.load(stream=file_handle, Loader=yaml.CSafeLoader)
            )

        worker(job, UPath(arguments.output_directory), arguments)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
        if error_action == "raise":
            raise e


def merge() -> None:
    run_merge(sys.argv[1:])


def run_merge(
    argv: list[str], error_action: Literal["raise", "ignore"] = "ignore"
) -> None:
    argument_parser = ArgumentParser(description="Generate Manhattan & QQ Plots")

    argument_parser.add_argument("input_directory")
    argument_parser.add_argument("output_directory")

    argument_parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        metavar="KEY",
        help="Group phenotypes by key",
        required=True,
    )
    argument_parser.add_argument("--batch-size", type=int, default=1000)

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--num-threads", type=int, default=cpu_count())

    arguments = argument_parser.parse_args(argv)

    from ..log import logger, setup_logging

    output_directory = UPath(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(level=arguments.log_level, path=output_directory)

    try:
        from .merge import merge

        merge(arguments)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
        if error_action == "raise":
            raise e
