# -*- coding: utf-8 -*-
from pathlib import Path
from subprocess import call

import numpy as np
import pytest

from gwas.vcf import VCFFile

vcf_path_zstd = Path("/scratch/ds-opensnp/100/chr22.dose.vcf.zst")


@pytest.mark.slow
@pytest.mark.parametrize("compression", ["zst", "xz", "gz"])
def test_vcf_file(tmp_path, compression):
    compress_command = dict(
        zst=None,
        xz="xz",
        gz="gzip",
    ).get(compression)

    if compress_command is not None:
        vcf_path = tmp_path / f"{vcf_path_zstd.stem}.{compression}"
        call(
            [
                "bash",
                "-c",
                f"zstd -c -d --long=31 {vcf_path_zstd} | "
                f"{compress_command} > {str(vcf_path)}",
            ]
        )
    else:
        vcf_path = vcf_path_zstd

    vcf_file = VCFFile(vcf_path)

    array = np.zeros((vcf_file.variant_count, vcf_file.sample_count))

    with vcf_file:
        vcf_file.read(array)

    assert np.abs(array).sum() > 0

    array = np.zeros((1000, vcf_file.sample_count))

    with vcf_file:
        vcf_file.read(array)

    assert np.abs(array).sum() > 0
