import logging
import sys
from argparse import ArgumentParser, Namespace
from typing import Literal

from upath import UPath


def parse_arguments(argv: list[str]) -> Namespace:
    """Parses command-line arguments"""
    argument_parser = ArgumentParser(description="Calculate HDL")

    argument_parser.add_argument("--input-path", type=UPath, required=True)
    argument_parser.add_argument("--ld-path", type=UPath, required=True)
    argument_parser.add_argument("--output-path", type=UPath, required=True)

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--mem-gb", type=float)
    argument_parser.add_argument("--num-threads", type=int, default=None)

    return argument_parser.parse_args(argv)


def main() -> None:
    run(sys.argv[1:])


def run(argv: list[str], error_action: Literal["raise", "ignore"] = "ignore") -> None:
    arguments = parse_arguments(argv)

    from ..log import logger, setup_logging

    setup_logging(level=arguments.log_level)

    from ..utils.threads import apply_num_threads, cpu_count

    if arguments.num_threads is None:
        arguments.num_threads = cpu_count()

    apply_num_threads(arguments.num_threads)

    from ..mem.wkspace import SharedWorkspace

    size: int | None = None
    if arguments.mem_gb is not None:
        size = int(arguments.mem_gb * 2**30 * 0.9)

    with SharedWorkspace.create(size=size) as sw:
        try:
            from .command import hdl

            hdl(sw, arguments)
        except Exception as e:
            logger.exception("Exception: %s", e, exc_info=True)
            if arguments.debug:
                import pdb

                pdb.post_mortem()
            if error_action == "raise":
                raise e
