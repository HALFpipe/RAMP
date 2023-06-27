# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from gwas.compression.pipe import CompressedTextReader
from gwas.null_model.base import NullModelCollection
from gwas.pheno import VariableCollection
from gwas.score.cli import run
from gwas.score.job import SummaryCollection

from ..conftest import chromosomes
from ..utils import check_bias
from .conftest import RmwScore


def test_run(
    tmp_path: Path,
    vcf_paths_by_chromosome: Mapping[int | str, Path],
    tri_paths_by_chromosome: Mapping[str | int, Path],
    vc: VariableCollection,
    nm: NullModelCollection,
    chromosome: int | str,
    rmw_score: RmwScore,
) -> None:
    vcf_paths = [str(vcf_paths_by_chromosome[c]) for c in chromosomes]
    tri_paths = [str(tri_paths_by_chromosome[c]) for c in chromosomes if c != "X"]
    phenotype_path = str(tmp_path / "phenotypes.tsv")
    phenotype_frame = pd.DataFrame(
        vc.phenotypes.to_numpy(), columns=vc.phenotype_names, index=vc.samples
    )
    phenotype_frame.to_csv(
        phenotype_path, sep="\t", index=True, header=True, na_rep="n/a"
    )
    covariate_path = str(tmp_path / "covariates.tsv")
    covariate_frame = pd.DataFrame(
        vc.covariates.to_numpy(), columns=vc.covariate_names, index=vc.samples
    )
    covariate_frame.to_csv(
        covariate_path, sep="\t", index=True, header=True, na_rep="n/a"
    )
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
        ],
        error_action="raise",
    )

    sc = SummaryCollection.from_file(tmp_path / f"chr{chromosome}.metadata.yaml.gz")
    for summaries in sc.chunks.values():
        for summary in summaries.values():
            assert summary.sample_count == vc.sample_count
            for i, phenotype_name in enumerate(vc.phenotype_names):
                phenotype_summary = summary.phenotypes[phenotype_name]
                assert phenotype_summary.method == "fastlmm"
                assert phenotype_summary.heritability is not None
                assert np.isclose(phenotype_summary.heritability, nm.heritability[i])
                assert phenotype_summary.genetic_variance is not None
                assert np.isclose(
                    phenotype_summary.genetic_variance, nm.genetic_variance[i]
                )
                assert phenotype_summary.error_variance is not None
                assert np.isclose(
                    phenotype_summary.error_variance, nm.error_variance[i]
                )
                assert np.isclose(
                    phenotype_summary.mean,
                    phenotype_frame[phenotype_name].mean(),
                )
                assert np.isclose(
                    phenotype_summary.variance,
                    np.var(phenotype_frame[phenotype_name], ddof=1),
                )

    with CompressedTextReader(
        tmp_path / f"chr{chromosome}.score.txt.zst"
    ) as file_handle:
        data_frame = pd.read_table(file_handle)

    u_stat_columns = [f"{name}_stat-u" for name in vc.phenotype_names]
    u_stat = data_frame[u_stat_columns].values
    rmw_u_stat = rmw_score.array["U_STAT"]

    v_stat_columns = [f"{name}_stat-v" for name in vc.phenotype_names]
    v_stat = data_frame[v_stat_columns].values
    sqrt_v_stat = np.sqrt(v_stat)
    rmw_sqrt_v_stat = rmw_score.array["SQRT_V_STAT"]

    is_ok = True
    for i in range(vc.phenotype_count):
        rmw_finite = np.isfinite(rmw_u_stat[:, i]) & np.isfinite(rmw_sqrt_v_stat[:, i])
        rmw_missing_rate = (~rmw_finite).mean()
        finite = ~(np.isclose(u_stat[:, i], 0) & np.isclose(v_stat[:, i], 1))
        missing_rate = (~finite).mean()

        assert missing_rate <= max(
            rmw_missing_rate * 2, 0.1
        ), "Missing rate is too high"

        to_compare = finite & rmw_finite

        has_no_bias = check_bias(u_stat[:, i], rmw_u_stat[:, i], to_compare)
        is_ok = is_ok and has_no_bias

        has_no_bias = check_bias(sqrt_v_stat[:, i], rmw_sqrt_v_stat[:, i], to_compare)
        is_ok = is_ok and has_no_bias

    assert is_ok
