from collections import deque
from contextlib import AbstractContextManager
from subprocess import PIPE, Popen
from types import TracebackType
from typing import IO, Mapping, Type, TypeVar

from upath import UPath

from ..log import logger
from ..utils.shutil import unwrap_which

decompress_commands: Mapping[str, list[str]] = {
    ".zst": [
        "zstd",
        "--long=31",
        "--decompress",
        "--stdout",
        "--no-progress",
    ],
    ".gz": ["bgzip", "--stdout", "--decompress"],
    ".xz": ["xz", "--decompress", "--stdout"],
    ".bz2": ["bzip2", "--decompress", "--stdout"],
    ".lz4": ["lz4", "-c", "-d"],
}
T = TypeVar("T", bytes, str)


class CompressedReader(AbstractContextManager[IO[T]]):
    def __init__(self, file_path: UPath | str, is_text: bool = True) -> None:
        self.file_path: UPath = UPath(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(self.file_path)
        self.is_text = is_text

        self.input_file_handle: IO[bytes] | None = None
        self.process_handle: Popen[T] | None = None
        self.output_file_handle: IO[T] | None = None

    def __enter__(self) -> IO[T]:
        return self.open()

    def open(self) -> IO[T]:
        suffix = self.file_path.suffix
        if suffix not in decompress_commands:
            raise ValueError(f'Compression for file suffix "{suffix}" is not supported')
        decompress_command: list[str] = decompress_commands[suffix]

        executable = unwrap_which(decompress_command[0])
        decompress_command[0] = executable

        self.input_file_handle = self.file_path.open(mode="rb")

        bufsize = 1 if self.is_text else -1
        self.process_handle = Popen(
            decompress_command,
            stderr=PIPE,
            stdin=self.input_file_handle,
            stdout=PIPE,
            text=self.is_text,
            bufsize=bufsize,
        )

        if self.process_handle.stdout is None:
            raise IOError

        self.output_file_handle = self.process_handle.stdout
        return self.output_file_handle

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
        output_file_handle = self.output_file_handle
        if output_file_handle is not None:
            # Read until EOF
            # Taken from https://stackoverflow.com/a/50938015
            deque(output_file_handle, maxlen=0)
        if self.process_handle is not None:
            stderr = self.process_handle.stderr
            if stderr is None:
                raise IOError
            data: str | bytes = stderr.read()
            stderr.close()
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")

            self.process_handle.__exit__(exc_type, value, traceback)
            returncode = self.process_handle.returncode

            if returncode:
                raise ValueError(
                    f'Decompression failed with code {returncode} and message "{data}"'
                )
            elif data:
                logger.warning(f'Decompression received stderr: "{data}"')

        self.process_handle = None
        self.input_file_handle = None
        self.output_file_handle = None


class CompressedBytesReader(CompressedReader[bytes]):
    def __init__(self, file_path: UPath | str) -> None:
        super().__init__(file_path, is_text=False)

    def __enter__(self) -> IO[bytes]:
        return super().__enter__()


class CompressedTextReader(CompressedReader[str]):
    def __init__(self, file_path: UPath | str) -> None:
        super().__init__(file_path, is_text=True)

    def open(self) -> IO[str]:
        if self.file_path.suffix not in decompress_commands:
            # File is not compressed
            self.output_file_handle = self.file_path.open("r")
            if self.output_file_handle is None:
                raise IOError
            return self.output_file_handle
        return super().open()


def make_compress_command(
    suffix: str, num_threads: int, compression_level: int | None = None
) -> list[str]:
    zstd_compression_level_flag = "-19"
    if compression_level is not None:
        zstd_compression_level_flag = f"-{compression_level:d}"

    compress_commands: Mapping[str, list[str]] = {
        ".zst": [
            "zstd",
            f"--threads={num_threads:d}",
            "--ultra",
            zstd_compression_level_flag,
            "--no-progress",
        ],
        ".gz": ["bgzip", "--threads", f"{num_threads:d}"],
        ".xz": ["xz"],
        ".bz2": ["bzip2", "-c"],
        ".lz4": ["lz4"],
    }

    return compress_commands[suffix]


class CompressedWriter(AbstractContextManager[IO[T]]):
    def __init__(
        self,
        file_path: UPath | str,
        num_threads: int,
        is_text: bool = True,
        compression_level: int | None = None,
    ) -> None:
        self.file_path: UPath = UPath(file_path)
        self.is_text = is_text
        self.num_threads = num_threads
        self.compression_level = compression_level

        self.input_file_handle: IO[T] | None = None
        self.output_file_handle: IO[bytes] | None = None
        self.process_handle: Popen[T] | None = None

    def __enter__(self) -> IO[T]:
        return self.open()

    def open(self) -> IO[T]:
        suffix = self.file_path.suffix
        if suffix not in decompress_commands:
            raise ValueError(
                f'Decompression for file suffix "{suffix}" is not supported'
            )

        self.output_file_handle = self.file_path.open(mode="wb")

        compress_command: list[str] = make_compress_command(
            suffix,
            num_threads=self.num_threads,
            compression_level=self.compression_level,
        )

        executable = unwrap_which(compress_command[0])
        compress_command[0] = executable

        logger.debug(f'Compressing to "{self.file_path}" with "{compress_command}"')

        bufsize = 1 if self.is_text else -1
        self.process_handle = Popen(
            compress_command,
            stderr=PIPE,
            stdin=PIPE,
            stdout=self.output_file_handle,
            text=self.is_text,
            bufsize=bufsize,
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
        if self.input_file_handle is not None:
            self.input_file_handle.close()

        if self.process_handle is not None:
            stderr = self.process_handle.stderr
            if stderr is None:
                raise IOError
            data: str | bytes = stderr.read()
            stderr.close()
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")

            self.process_handle.__exit__(exc_type, value, traceback)
            returncode = self.process_handle.returncode
            if returncode:
                raise ValueError(
                    f'Compression failed with code {returncode} and message "{data}"'
                )
            elif data:
                logger.warning(f'Compression received stderr: "{data}"')

        if self.output_file_handle is not None:
            self.output_file_handle.close()

        self.process_handle = None
        self.input_file_handle = None
        self.output_file_handle = None


class CompressedBytesWriter(CompressedWriter[bytes]):
    def __init__(
        self,
        file_path: UPath | str,
        num_threads: int,
        compression_level: int | None = None,
    ) -> None:
        super().__init__(
            file_path,
            is_text=False,
            num_threads=num_threads,
            compression_level=compression_level,
        )

    def __enter__(self) -> IO[bytes]:
        return super().__enter__()


class CompressedTextWriter(CompressedWriter[str]):
    def __init__(
        self,
        file_path: UPath | str,
        num_threads: int,
        compression_level: int | None = None,
    ) -> None:
        super().__init__(
            file_path,
            is_text=True,
            num_threads=num_threads,
            compression_level=compression_level,
        )

    def open(self) -> IO[str]:
        if self.file_path.suffix in {".vcf", ".txt"}:
            self.input_file_handle = self.file_path.open("w")
            if self.input_file_handle is None:
                raise IOError
            return self.input_file_handle
        return super().open()
