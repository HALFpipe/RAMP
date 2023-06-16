# -*- coding: utf-8 -*-
import logging
from argparse import Action, ArgumentError, ArgumentParser, Namespace
from typing import Any, Sequence

from ..log import logger
from .populations import populations, super_populations
from .run import run


class GroupAction(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        by_list: list[str] | None = getattr(namespace, "by", None)
        if by_list is None:
            raise ArgumentError(self, "Must specify `--by` before `--group`")
        if not isinstance(values, (list, tuple)):
            raise ArgumentError(
                self, "Must specify name, lower bound and upper bound for `--group`"
            )

        by = by_list[-1]
        name, lower_bound, upper_bound = values
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
        values = (by, name, lower_bound, upper_bound)

        items = getattr(namespace, self.dest, None)
        if items is None:
            items = list()
        items.append(values)
        setattr(namespace, self.dest, items)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)8s] %(funcName)s: "
            "%(message)s (%(filename)s:%(lineno)s)"
        ),
    )

    argument_parser = ArgumentParser()

    # Input files
    argument_parser.add_argument("--mds", metavar="CSV", required=True)
    argument_parser.add_argument("--covariates", nargs="+")
    argument_parser.add_argument("--phenotypes", nargs="+")

    # Output files
    argument_parser.add_argument("--output-directory", required=True)
    argument_parser.add_argument(
        "--output-sample-ids",
        required=False,
        choices=["fid", "iid", "both_with_underscore"],
        default="both_with_underscore",
    )

    # Split options
    group = argument_parser.add_mutually_exclusive_group()
    group.add_argument("--by-population", action="store_true", default=False)
    group.add_argument("--by-super-population", action="store_true", default=False)
    argument_parser.add_argument("--ignore-population", nargs="+", default=list())
    argument_parser.add_argument(
        "--main-population", required=False, choices=populations, default="CEU"
    )
    argument_parser.add_argument(
        "--main-super-population",
        required=False,
        choices=super_populations.keys(),
        default="EUR",
    )
    argument_parser.add_argument("--component-count", type=int)
    argument_parser.add_argument(
        "--population-standard-deviations",
        "--population-std-devs",
        required=False,
        type=float,
        default=6,
    )
    argument_parser.add_argument(
        "--by",
        type=str,
        action="append",
    )
    argument_parser.add_argument(
        "--group",
        type=str,
        action=GroupAction,
        nargs=3,
        metavar=("NAME", "LOWER_BOUND", "UPPER_BOUND"),
    )
    argument_parser.add_argument("--minimum-sample-size", type=int, default=50)

    # Covariate options
    argument_parser.add_argument(
        "--add-components-to-covariates",
        action="store_true",
        default=False,
    )

    # Sample matching options
    argument_parser.add_argument(
        "--rename-samples-replace",
        type=str,
        nargs=2,
        action="append",
        metavar=("FROM", "TO"),
        default=list(),
    )
    argument_parser.add_argument(
        "--rename-samples-from-file",
        type=str,
        nargs=3,
        action="append",
        metavar=("FROM_COLUMN", "TO_COLUMN", "CSV_PATH"),
        default=list(),
    )
    argument_parser.add_argument(
        "--match-case-insensitive",
        action="store_true",
        default=False,
    )
    argument_parser.add_argument(
        "--match-alphanumeric-only",
        action="store_true",
        default=False,
    )

    argument_parser.add_argument("--debug", action="store_true", default=False)

    arguments = argument_parser.parse_args()

    try:
        run(arguments)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
