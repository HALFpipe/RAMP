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
    check_call(
        [
            "bash",
            "-c",
            f"zstd -c -d --long=31 {zstd_path} | bgzip > {gz_path}",
        ]
    )

    check_call(
        [
            tabix,
            "-p",
            "vcf",
            str(gz_path),
        ]
    )

    return gz_path
