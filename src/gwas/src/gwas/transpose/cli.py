# -*- coding: utf-8 -*-
import logging
import sys
from argparse import ArgumentParser, Namespace
from contextlib import ExitStack
from multiprocessing import cpu_count
from pathlib import Path

from tqdm.auto import tqdm

from ..compression.pipe import CompressedBytesReader, CompressedBytesWriter
from ..log import logger, setup_logging


def transpose(arguments: Namespace, output_directory: Path) -> None:
    column_directory = output_directory / "transpose-columns"
    column_directory.mkdir(parents=True, exist_ok=True)

    columns: list[str] | None = None
    column_file_paths: list[Path] | None = None
    score_paths = [Path(score) for score in arguments.score]
    with ExitStack() as stack:
        compressed_text_readers = [
            stack.enter_context(CompressedBytesReader(score_path))
            for score_path in score_paths
        ]
        headers = {
            tuple(reader.readline().split(b"\t")) for reader in compressed_text_readers
        }
        if len(headers) != 1:
            raise ValueError("Found mismatched headers across score files")
        columns = [
            column_bytes.decode("utf-8").strip() for column_bytes in headers.pop()
        ]
        column_file_paths = [
            (column_directory / f"{column}.txt.zst") for column in columns
        ]

    if columns is None or column_file_paths is None:
        raise RuntimeError("Unknown column file paths")

    with ExitStack() as stack:
        compressed_text_readers = [
            stack.enter_context(CompressedBytesReader(score_path))
            for score_path in score_paths
        ]
        column_stream_writers = [
            stack.enter_context(CompressedBytesWriter(column_file_path, num_threads=1))
            for column_file_path in column_file_paths
        ]
        for reader in tqdm(compressed_text_readers):
            for line in tqdm(reader, leave=False):
                for column_data, column_stream_writer in zip(
                    line.split(b"\t"), column_stream_writers, strict=False
                ):
                    column_stream_writer.write(column_data.strip())
                    column_stream_writer.write(b"\t")
                    # column_stream_writer.flush(zstd.FLUSH_FRAME)
        for column_stream_writer in column_stream_writers:
            column_stream_writer.write(b"\n")


def parse_arguments(argv: list[str]) -> Namespace:
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--score", nargs="+", required=True)
    argument_parser.add_argument("--output-directory", required=True)

    # Program options
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("--num-threads", type=int, default=cpu_count())

    return argument_parser.parse_args(argv)


def main() -> None:
    arguments = parse_arguments(sys.argv[1:])

    output_directory = Path.cwd()
    if arguments.output_directory is not None:
        output_directory = Path(arguments.output_directory)

    setup_logging(level=arguments.log_level, path=output_directory)

    try:
        transpose(arguments, output_directory)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
