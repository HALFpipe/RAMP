# -*- coding: utf-8 -*-
from contextlib import AbstractContextManager
from pathlib import Path
from subprocess import DEVNULL, PIPE, Popen
from typing import IO

from .utils import unwrap_which


class CompressedTextReader(AbstractContextManager):
    def __init__(self, file_path: Path | str) -> None:
        self.file_path: Path = Path(file_path)

        self.process_handle: Popen | None = None
        self.file_handle: IO[str] | None = None

    def __enter__(self) -> IO[str]:
        if self.file_path.suffix in {".vcf", ".txt"}:
            self.file_handle = self.file_path.open(mode="rt")
            return self.file_handle

        decompress_command: list[str] = {
            ".zst": ["zstd", "--long=31", "-c", "-d"],
            ".lrz": ["lrzcat", "--quiet"],
            ".gz": ["bgzip", "-c", "-d"],
            ".xz": ["xzcat"],
        }[self.file_path.suffix]

        executable = unwrap_which(decompress_command[0])
        decompress_command[0] = executable

        self.process_handle = Popen(
            [*decompress_command, str(self.file_path)],
            stderr=DEVNULL,
            stdin=DEVNULL,
            stdout=PIPE,
            text=True,
            bufsize=1,
        )

        if self.process_handle.stdout is None:
            raise IOError

        self.file_handle = self.process_handle.stdout
        return self.file_handle

    def __exit__(self, exc_type, value, traceback) -> None:
        if self.process_handle is not None:
            self.process_handle.__exit__(exc_type, value, traceback)
        elif self.file_handle is not None:
            self.file_handle.close()
        self.process_handle = None
        self.file_handle = None


class CompressedTextWriter(AbstractContextManager):
    def __init__(self, file_path: Path | str) -> None:
        self.file_path: Path = Path(file_path)
        self.input_file_handle: IO[str] | None = None
        self.output_file_handle: IO[bytes] | None = None
        self.process_handle: Popen | None = None

    def __enter__(self) -> IO[str]:
        if self.file_path.suffix in {".vcf", ".txt"}:
            self.input_file_handle = self.file_path.open(mode="wt")
            return self.input_file_handle
        else:
            self.output_file_handle = self.file_path.open(mode="wb")

        compress_command: list[str] = {
            ".zst": ["zstd", "-11"],
            ".lrz": ["lrzip"],
            ".gz": ["bgzip"],
            ".xz": ["xz"],
        }[self.file_path.suffix]

        executable = unwrap_which(compress_command[0])
        compress_command[0] = executable

        self.process_handle = Popen(
            compress_command,
            stderr=DEVNULL,
            stdin=PIPE,
            stdout=self.output_file_handle,
            text=True,
            bufsize=1,
        )

        if self.process_handle.stdin is None:
            raise IOError

        self.input_file_handle = self.process_handle.stdin
        return self.input_file_handle

    def __exit__(self, exc_type, value, traceback) -> None:
        if self.process_handle is not None:
            self.process_handle.__exit__(exc_type, value, traceback)
        elif self.input_file_handle is not None:
            self.input_file_handle.close()
        self.process_handle = None
        self.input_file_handle = None
        self.output_file_handle = None
