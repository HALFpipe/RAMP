from pathlib import Path
from shlex import join
from subprocess import check_call

from ..tools import tabix
from .pipe import decompress_commands, make_compress_command


def to_bgzip(path: Path, prefix: Path, num_threads: int) -> Path:
    prefix.mkdir(parents=True, exist_ok=True)
    gz_path = prefix / f"{path.stem}.gz"

    if not gz_path.is_file():
        decompress_command = join(decompress_commands[path.suffix])
        compress_command = join(make_compress_command(".gz", num_threads=num_threads))
        pipe_command = f"{decompress_command} {path} | {compress_command} > {gz_path}"
        check_call(["bash", "-c", pipe_command])

    tbi_path = gz_path.with_suffix(".gz.tbi")
    if not tbi_path.is_file():
        check_call([*tabix, "--preset", "vcf", str(gz_path)])

    csi_path = gz_path.with_suffix(".gz.csi")
    if not csi_path.is_file():
        check_call([*tabix, "--preset", "vcf", "--csi", str(gz_path)])

    return gz_path
