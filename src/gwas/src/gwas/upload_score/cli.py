import logging
import sys
from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
from subprocess import call
from tempfile import TemporaryDirectory
from typing import Iterable, Literal

from upath import UPath

from ..log import logger, setup_logging
from ..utils import unwrap_which

path_patterns: list[str] = [
    "chr*.metadata.yaml.gz",
    "chr*.score.txt.zst",
    "covariance.txt.zst",
    "chr*.score.b2array",  # legacy
    "chr*.score.axis-metadata.pkl.zst",  # legacy
    "covariance.b2array",  # legacy
    "covariance.axis-metadata.pkl.zst",  # legacy
    "populations.pdf",
    "*stat-*_statmap.nii.gz",
    "*stat-*_statmap.json",
    "*mask.nii.gz",
    "*mean.nii.gz",
    "*std.nii.gz",
    "*contrast_matrix.tsv",
]


def call_upload_client(
    arguments: Namespace, tmp_path: UPath, paths: Iterable[str]
) -> None:
    upload_executable = unwrap_which("upload")
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


def get_relative_path(path: UPath, k: int) -> UPath:
    return UPath(*path.parts[-k - 1 :])


def has_duplicates(paths: Iterable[UPath], k: int) -> bool:
    seen: set[UPath] = set()
    for path in paths:
        path = get_relative_path(path, k)
        if path in seen:
            return True
        seen.add(path)
    return False


def upload(arguments: Namespace) -> None:
    upload_paths: set[UPath] = set()
    for input_directory_str in arguments.input_directory:
        input_directory = UPath(input_directory_str).absolute()
        for path_pattern in path_patterns:
            upload_paths.update(input_directory.glob(f"**/{path_pattern}"))

    k = 0
    while has_duplicates(upload_paths, k):
        k += 1

    with TemporaryDirectory() as tmp_path_str:
        tmp_path = UPath(tmp_path_str)
        paths: set[str] = set()
        for upload_path in upload_paths:
            if upload_path.name.startswith("sub-"):
                # Skip BIDS subject files
                continue
            link_path = tmp_path / get_relative_path(upload_path, k)
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(upload_path)
            paths.add(link_path.relative_to(tmp_path).as_posix())

        logger.info(f"Uploading {len(paths)} files")
        call_upload_client(arguments, tmp_path, paths)


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
