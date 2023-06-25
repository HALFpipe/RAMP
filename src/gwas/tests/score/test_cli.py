# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from gwas.compression.pipe import CompressedTextReader
from gwas.pheno import VariableCollection
from gwas.score.cli import run

from ..conftest import chromosomes
from ..utils import check_bias
from .conftest import RmwScore

sample_size_label = "small"


def test_run(
    tmp_path: Path,
    vcf_paths_by_size_and_chromosome: Mapping[str, Mapping[int | str, Path]],
    tri_paths_by_size_and_chromosome: Mapping[str, Mapping[str | int, Path]],
    vc: VariableCollection,
    chromosome: int | str,
    rmw_score: RmwScore,
) -> None:
    vcf_paths = [
        str(vcf_paths_by_size_and_chromosome[sample_size_label][c]) for c in chromosomes
    ]
    tri_paths = [
        str(tri_paths_by_size_and_chromosome[sample_size_label][c])
        for c in chromosomes
        if c != "X"
    ]
    phenotype_path = str(tmp_path / "phenotypes.tsv")
    data_frame = pd.DataFrame(
        vc.phenotypes.to_numpy(), columns=vc.phenotype_names, index=vc.samples
    )
    data_frame.to_csv(phenotype_path, sep="\t", index=True, header=True, na_rep="n/a")
    covariate_path = str(tmp_path / "covariates.tsv")
    data_frame = pd.DataFrame(
        vc.covariates.to_numpy(), columns=vc.covariate_names, index=vc.samples
    )
    data_frame.to_csv(covariate_path, sep="\t", index=True, header=True, na_rep="n/a")
    run(
        [
            "--vcf",
            *vcf_paths,
            "--tri",
            *tri_paths,
            "--phenotypes",
            phenotype_path,
            "--covariates",
            covariate_path,
            "--output-directory",
            str(tmp_path),
            "--chromosome",
            str(chromosome),
            "--compression-method",
            "zstd_text",
        ]
    )

    with CompressedTextReader(
        tmp_path / f"chr{chromosome}.score.txt.zst"
    ) as file_handle:
        data_frame = pd.read_table(file_handle)

    u_stat_columns = [f"{name}_stat-u" for name in vc.phenotype_names]
    u_stat = data_frame[u_stat_columns].values
    rmw_u_stat = rmw_score.array["U_STAT"]

    is_ok = True
    for i in range(vc.phenotype_count):
        finite = np.isfinite(rmw_u_stat[:, i])
        has_no_bias = check_bias(u_stat[:, i], rmw_u_stat[:, i], finite)
        is_ok = is_ok and has_no_bias

    v_stat_columns = [f"{name}_stat-v" for name in vc.phenotype_names]
    sqrt_v_stat = np.sqrt(data_frame[v_stat_columns].values)
    rmw_sqrt_v_stat = rmw_score.array["SQRT_V_STAT"]

    for i in range(vc.phenotype_count):
        finite = np.isfinite(rmw_sqrt_v_stat[:, i])
        has_no_bias = check_bias(sqrt_v_stat[:, i], rmw_sqrt_v_stat[:, i], finite)
        is_ok = is_ok and has_no_bias

    assert is_ok
