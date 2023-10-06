# -*- coding: utf-8 -*-
import pickle
from contextlib import AbstractContextManager
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import DEVNULL, PIPE, Popen
from types import TracebackType
from typing import IO, Any, Mapping, Type, TypeVar

from ..log import logger
from ..utils import unwrap_which

pipe_max_size: int = int(Path("/proc/sys/fs/pipe-max-size").read_text())
decompress_commands: Mapping[str, list[str]] = {
    ".zst": ["zstd", "--long=31", "-c", "-d"],
    ".lrz": ["lrzcat", "--quiet"],
    ".gz": ["bgzip", "-c", "-d"],
    ".xz": ["xz", "--decompress", "--stdout"],
    ".bz2": ["bzip2", "--decompress", "--stdout"],
    ".lz4": ["lz4", "-c", "-d"],
}
T = TypeVar("T", bytes, str)


class CompressedReader(AbstractContextManager[IO[T]]):
    def __init__(self, file_path: Path | str, is_text: bool = True) -> None:
        self.file_path: Path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(self.file_path)
        self.is_text = is_text

        self.process_handle: Popen | None = None
        self.file_handle: IO[T] | None = None

    def __enter__(self) -> IO[T]:
        return self.open()

    def open(self) -> IO[T]:
        suffix = self.file_path.suffix
        if suffix not in decompress_commands:
            raise ValueError(f'Compression for file suffix "{suffix}" is not supported')
        decompress_command: list[str] = decompress_commands[suffix]

        executable = unwrap_which(decompress_command[0])
        decompress_command[0] = executable

        bufsize = 1 if self.is_text else -1
        self.process_handle = Popen(
            [*decompress_command, str(self.file_path)],
            stderr=DEVNULL,
            stdin=DEVNULL,
            stdout=PIPE,
            text=self.is_text,
            bufsize=bufsize,
            pipesize=pipe_max_size,
        )

        if self.process_handle.stdout is None:
            raise IOError

        self.file_handle = self.process_handle.stdout
        return self.file_handle

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(exc_type, value, traceback)

    def close(
        self,
        exc_type: Type[BaseException] | None = None,
        value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        if self.process_handle is not None:
            self.process_handle.__exit__(exc_type, value, traceback)
        elif self.file_handle is not None:
            self.file_handle.close()
        self.process_handle = None
        self.file_handle = None


class CompressedBytesReader(CompressedReader[bytes]):
    def __init__(self, file_path: Path | str) -> None:
        super().__init__(file_path, is_text=False)

    def __enter__(self) -> IO[bytes]:
        return super().__enter__()


class CompressedTextReader(CompressedReader[str]):
    def __init__(self, file_path: Path | str) -> None:
        super().__init__(file_path, is_text=True)

    def open(self) -> IO[str]:
        if self.file_path.suffix in {".vcf", ".txt"}:
            # File is not compressed
            self.file_handle = self.file_path.open(mode="rt")
            return self.file_handle
        return super().open()


class CompressedWriter(AbstractContextManager[IO[T]]):
    def __init__(
        self,
        file_path: Path | str,
        is_text: bool = True,
        num_threads: int = cpu_count(),
        compression_level: int | None = None,
    ) -> None:
        self.file_path: Path = Path(file_path)
        self.is_text = is_text
        self.num_threads = num_threads
        self.compression_level = compression_level

        self.input_file_handle: IO | None = None
        self.output_file_handle: IO[bytes] | None = None
        self.process_handle: Popen | None = None

    def __enter__(self) -> IO[T]:
        return self.open()

    def open(self) -> IO[T]:
        suffix = self.file_path.suffix
        if suffix not in decompress_commands:
            raise ValueError(
                f'Decompression for file suffix "{suffix}" is not supported'
            )

        self.output_file_handle = self.file_path.open(mode="wb")

        zstd_compression_level_flag = "-22"
        if self.compression_level is not None:
            zstd_compression_level_flag = f"-{self.compression_level:d}"

        compress_commands: Mapping[str, list[str]] = {
            ".zst": [
                "zstd",
                "-B0",
                f"-T{self.num_threads:d}",
                "--ultra",
                zstd_compression_level_flag,
            ],
            ".lrz": ["lrzip"],
            ".gz": ["bgzip", "--threads", f"{self.num_threads:d}"],
            ".xz": ["xz"],
            ".bz2": ["bzip2", "-c"],
            ".lz4": ["lz4"],
        }
        compress_command: list[str] = compress_commands[suffix]

        executable = unwrap_which(compress_command[0])
        compress_command[0] = executable

        bufsize = 1 if self.is_text else -1
        self.process_handle = Popen(
            compress_command,
            stderr=DEVNULL,
            stdin=PIPE,
            stdout=self.output_file_handle,
            text=self.is_text,
            bufsize=bufsize,
            pipesize=pipe_max_size,
        )

        if self.process_handle.stdin is None:
            raise IOError

        self.input_file_handle = self.process_handle.stdin
        return self.input_file_handle

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(exc_type, value, traceback)

    def close(
        self,
        exc_type: Type[BaseException] | None = None,
        value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        if self.process_handle is not None:
            self.process_handle.__exit__(exc_type, value, traceback)
        elif self.input_file_handle is not None:
            self.input_file_handle.close()
        self.process_handle = None
        self.input_file_handle = None
        self.output_file_handle = None


class CompressedBytesWriter(CompressedWriter[bytes]):
    def __init__(self, file_path: Path | str, **kwargs) -> None:
        super().__init__(file_path, is_text=False, **kwargs)

    def __enter__(self) -> IO[bytes]:
        return super().__enter__()


class CompressedTextWriter(CompressedWriter[str]):
    def __init__(self, file_path: Path | str, **kwargs) -> None:
        super().__init__(file_path, is_text=True, **kwargs)

    def open(self) -> IO[str]:
        if self.file_path.suffix in {".vcf", ".txt"}:
            self.input_file_handle = self.file_path.open(mode="wt")
            return self.input_file_handle
        return super().open()


cache_suffix: str = ".pickle.zst"


def load_from_cache(cache_path: Path, key: str) -> Any:
    file_path = cache_path / f"{key}{cache_suffix}"
    if not file_path.is_file():
        logger.debug(f'Cache entry "{file_path}" not found')
        return None
    with CompressedBytesReader(file_path) as file_handle:
        try:
            return pickle.load(file_handle)
        except pickle.UnpicklingError as e:
            logger.warning(f'Failed to load "{file_path}"', exc_info=e)
            return None


def save_to_cache(cache_path: Path, key: str, value: Any) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    with CompressedBytesWriter(cache_path / f"{key}{cache_suffix}") as file_handle:
        pickle.dump(value, file_handle)
