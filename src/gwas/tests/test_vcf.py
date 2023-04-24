# -*- coding: utf-8 -*-
from subprocess import call

import numpy as np
import pytest

from gwas.vcf import VCFFile

chromosome = 22


@pytest.mark.slow
@pytest.mark.parametrize("compression", ["zst", "xz", "gz"])
def test_vcf_file(tmp_path, compression, vcf_paths_by_chromosome):
    compress_command = dict(
        zst=None,
        xz="xz",
        gz="gzip",
    ).get(compression)

    vcf_path_zstd = vcf_paths_by_chromosome[chromosome]
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


def test_record_count(benchmark, vcf_paths_by_chromosome):
    vcf_path = vcf_paths_by_chromosome[chromosome]
    benchmark(VCFFile, vcf_path)
