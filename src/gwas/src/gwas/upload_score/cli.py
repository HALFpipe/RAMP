import logging
import sys
from argparse import ArgumentParser, Namespace
from subprocess import call
from typing import Literal

from upath import UPath

from ..compression.arr.base import compression_methods
from ..log import logger, setup_logging
from ..utils.shutil import unwrap_which
from ..utils.threads import cpu_count

score_prefixes: list[str] = [
    "chr*.score",
    "covariance",
]


path_patterns: list[str] = [
    "chr*.metadata.yaml.gz",
    *(
        f"{prefix}{compression_method.suffix}"
        for prefix in score_prefixes
        for compression_method in compression_methods.values()
    ),
    "chr*.score.axis-metadata.pkl.zst",  # legacy
    "covariance.axis-metadata.pkl.zst",  # legacy
    "populations.pdf",
    "*stat-*_statmap.nii.gz",
    "*stat-*_statmap.json",
    "*mask.nii.gz",
    "*mean.nii.gz",
    "*std.nii.gz",
    "*contrast_matrix.tsv",
]
base_path = UPath("/")


def call_upload_client(arguments: Namespace, path_strs: list[str]) -> None:
    upload_executable = unwrap_which("upload")
    command: list[str] = [
        upload_executable,
        "upload-client",
        "--token",
        arguments.token,
        "--endpoint",
        arguments.endpoint,
        "--path",
        *path_strs,
    ]
    if arguments.log_level == "DEBUG":
        command.append("--debug")

    logger.debug(f"Running command: {' '.join(command)}")
    call(command, cwd=base_path)


def upload(arguments: Namespace) -> None:
    paths: set[UPath] = set()
    for input_directory_str in arguments.input_directory:
        input_directory = UPath(input_directory_str).absolute()
        for path_pattern in path_patterns:
            paths.update(input_directory.glob(f"**/{path_pattern}"))

        logger.info(f"Uploading {len(paths)} files")
    path_strs = sorted(f"{path.relative_to(base_path)}" for path in paths)
    call_upload_client(arguments, path_strs)


def parse_arguments(argv: list[str]) -> Namespace:
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--input-directory", nargs="+", required=True)

    argument_parser.add_argument("--token", required=True)
    argument_parser.add_argument("--endpoint", default="https://upload.gwas.science")

    # Program options
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("--num-threads", type=int, default=cpu_count())

    return argument_parser.parse_args(argv)


def run(argv: list[str], error_action: Literal["raise", "ignore"] = "ignore") -> None:
    arguments = parse_arguments(argv)
    setup_logging(level=arguments.log_level)

    try:
        upload(arguments)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
        if error_action == "raise":
            raise e


def main() -> None:
    run(sys.argv[1:])
