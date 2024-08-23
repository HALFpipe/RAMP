import logging
import sys
from argparse import ArgumentParser, Namespace
from typing import Literal

import numpy as np
from upath import UPath


def main() -> None:
    run(sys.argv[1:])


def parse_arguments(argv: list[str]) -> Namespace:
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--vcf", required=True)
    argument_parser.add_argument("--output-directory", required=True)

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

    # Program options
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("--mem-gb", type=float)
    argument_parser.add_argument("--num-threads", type=int, default=None)

    return argument_parser.parse_args(argv)


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

    size: int | None = None
    if arguments.mem_gb is not None:
        size = int(arguments.mem_gb * 2**30)

    with SharedWorkspace.create(size=size) as sw:
        try:
            from ..vcf.base import calc_vcf
            from .base import Triangular
            from .tsqr import TallSkinnyQR

            (vcf_file,) = calc_vcf(
                [UPath(arguments.vcf)],
                output_directory,
                num_threads=arguments.num_threads,
                sw=sw,
            )
            vcf_file.set_samples(set(vcf_file.vcf_samples))
            vcf_file.set_variants_from_cutoffs(
                arguments.kinship_minor_allele_frequency_cutoff,
                arguments.kinship_r_squared_cutoff,
            )

            tsqr = TallSkinnyQR(
                vcf_file=vcf_file,
                sw=sw,
            )

            tri = tsqr.map_reduce()

            tri_path = output_directory / Triangular.get_file_name(vcf_file.chromosome)
            tri.to_file(tri_path)
            tri.free()
        except Exception as e:
            logger.exception("Exception: %s", e, exc_info=True)
            if arguments.debug:
                import pdb

                pdb.post_mortem()
            if error_action == "raise":
                raise e
