# -*- coding: utf-8 -*-
from pathlib import Path
from subprocess import check_call

from gwas.utils import unwrap_which

gcta64 = unwrap_which("gcta64")
plink2 = unwrap_which("plink2")
rmw = unwrap_which("raremetalworker")
tabix = unwrap_which("tabix")


def to_bgzip(base_path: Path, zstd_path: Path) -> Path:
    gz_path = base_path / f"{zstd_path.stem}.gz"

    if not gz_path.is_file():
        check_call(
            [
                "bash",
                "-c",
                f"zstd -c -d --long=31 {zstd_path} | bgzip > {gz_path}",
            ]
        )

    tbi_path = gz_path.with_suffix(".gz.tbi")
    if not tbi_path.is_file():
        check_call(
            [
                tabix,
                "-p",
                "vcf",
                str(gz_path),
            ]
        )

    return gz_path


def is_bfile(path: Path) -> bool:
    return all(
        (path.parent / f"{path.name}{suffix}").is_file()
        for suffix in {".bed", ".bim", ".fam"}
    )


def is_pfile(path: Path) -> bool:
    return all(
        (path.parent / f"{path.name}{suffix}").is_file()
        for suffix in {".pgen", ".pvar", ".psam"}
    )
