# -*- coding: utf-8 -*-
import logging
import sys
from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import call
from tempfile import TemporaryDirectory

from ..log import logger, setup_logging
from ..utils import unwrap_which

upload_executable = unwrap_which("upload")

path_patterns: list[str] = [
    "chr*.metadata.yaml.gz",
    "chr*.score.b2array",
    "chr*.score.axis-metadata.pkl.zst",
    "covariance.b2array",
    "covariance.axis-metadata.pkl.zst",
    "populations.pdf",
    "*stat-*_statmap.nii.gz",
    "*stat-*_statmap.json",
    "*mask.nii.gz",
    "*mean.nii.gz",
    "*std.nii.gz",
    "*contrast_matrix.tsv",
]


def upload(arguments: Namespace) -> None:
    upload_paths: list[Path] = list()
    for input_directory_str in arguments.input_directory:
        input_directory = Path(input_directory_str).absolute()
        for path_pattern in path_patterns:
            upload_paths.extend(input_directory.glob(f"**/{path_pattern}"))

    with TemporaryDirectory() as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        paths: set[str] = set()
        for upload_path in upload_paths:
            link_path = tmp_path / upload_path.name
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(upload_path)
            paths.add(link_path.name)

        logger.info(f"Uploading {len(paths)} files")

        command: list[str] = [
            upload_executable,
            "upload-client",
            "--token",
            arguments.token,
            "--endpoint",
            arguments.endpoint,
            "--path",
            *sorted(paths),
        ]
        if arguments.log_level == "DEBUG":
            command.append("--debug")

        logger.debug(f"Running command: {' '.join(command)}")
        call(command, cwd=tmp_path)


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


def main():
    arguments = parse_arguments(sys.argv[1:])
    setup_logging(level=arguments.log_level)

    try:
        upload(arguments)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
