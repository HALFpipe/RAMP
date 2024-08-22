from shutil import copyfile
from typing import Mapping

import numpy as np
import pandas as pd
from pytest import FixtureRequest
from upath import UPath

from gwas.compression.cache import cache_suffix
from gwas.eig.base import Eigendecomposition
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.pheno import VariableCollection, combine
from gwas.score.cli import parse_arguments
from gwas.score.command import GwasCommand
from gwas.score.job import SummaryCollection
from gwas.vcf.base import VCFFile

from ..conftest import chromosomes
from ..utils import assert_both_close, check_bias
from .conftest import RmwScore


def test_run(
    tmp_path: UPath,
    sw: SharedWorkspace,
    vcf_paths_by_chromosome: Mapping[int | str, UPath],
    tri_paths_by_chromosome: Mapping[str | int, UPath],
    cache_path: UPath,
    variable_collections: list[VariableCollection],
    eigendecompositions: list[Eigendecomposition],
    null_model_collections: list[NullModelCollection],
    chromosome: int | str,
    rmw_score: RmwScore,
    request: FixtureRequest,
) -> None:
    for eig, b in zip(eigendecompositions, variable_collections, strict=True):
        assert eig.samples == b.samples

    allocation_names = set(sw.allocations.keys())
    new_allocation_names: set[str] = set()

    vcf_paths = [vcf_paths_by_chromosome[c] for c in chromosomes]
    tri_paths = [tri_paths_by_chromosome[c] for c in chromosomes if c != "X"]

    for vcf_path in vcf_paths:
        vcf_metadata_cache_path = (
            cache_path / f"{VCFFile.cache_key(vcf_path)}{cache_suffix}"
        )
        copyfile(vcf_metadata_cache_path, tmp_path / vcf_metadata_cache_path.name)

    phenotype_frames = [
        pd.DataFrame(
            variable_collection.phenotypes.to_numpy(),
            columns=variable_collection.phenotype_names,
            index=variable_collection.samples,
        )
        for variable_collection in variable_collections
    ]
    phenotype_frame = combine(phenotype_frames)
    phenotype_path = str(tmp_path / "phenotypes.tsv")
    phenotype_frame.to_csv(
        phenotype_path, sep="\t", index=True, header=True, na_rep="n/a"
    )
    phenotype_names = [
        phenotype_name
        for variable_collection in variable_collections
        for phenotype_name in variable_collection.phenotype_names
    ]

    covariate_frames = [
        pd.DataFrame(
            variable_collection.covariates.to_numpy(),
            columns=variable_collection.covariate_names,
            index=variable_collection.samples,
        )
        for variable_collection in variable_collections
    ]
    covariate_frame = combine(covariate_frames)
    covariate_path = str(tmp_path / "covariates.tsv")
    covariate_frame.to_csv(
        covariate_path, sep="\t", index=True, header=True, na_rep="n/a"
    )

    arguments = parse_arguments(
        [
            "--vcf",
            *map(str, vcf_paths),
            "--tri",
            *map(str, tri_paths),
            "--phenotypes",
            phenotype_path,
            "--covariates",
            covariate_path,
            "--score-r-squared-cutoff=-inf",
            "--score-minor-allele-frequency-cutoff=-inf",
            "--output-directory",
            str(tmp_path),
            "--chromosome",
            str(chromosome),
            "--compression-method",
            "zstd_text",
            "--log-level",
            "DEBUG",
        ]
    )
    command = GwasCommand(arguments, tmp_path, sw)
    request.addfinalizer(command.free)

    command_variable_collections = command.setup_variable_collections()
    for a, b in zip(command_variable_collections, variable_collections, strict=True):
        assert a.phenotype_names == b.phenotype_names
        assert a.samples == b.samples

        new_allocation_names.add(a.phenotypes.name)
        new_allocation_names.add(a.covariates.name)

        request.addfinalizer(a.free)

    for v in command.vcf_by_chromosome.values():
        new_allocation_names.update(v.shared_vcf_variants.allocation_names)

    command = GwasCommand(arguments, tmp_path, sw)
    request.addfinalizer(command.free)

    command.run()

    for v in command.vcf_by_chromosome.values():
        new_allocation_names.update(v.shared_vcf_variants.allocation_names)

    sc = SummaryCollection.from_file(tmp_path / f"chr{chromosome}.metadata.yaml.gz")
    (summaries,) = sc.chunks.values()
    summaries_by_phenotype = {
        phenotype_name: summary
        for summary in summaries.values()
        for phenotype_name in summary.phenotypes.keys()
    }

    for variable_collection, null_model_collection in zip(
        variable_collections, null_model_collections, strict=True
    ):
        for i, phenotype_name in enumerate(variable_collection.phenotype_names):
            summary = summaries_by_phenotype[phenotype_name]
            assert summary.sample_count == variable_collection.sample_count
            phenotype_summary = summary.phenotypes[phenotype_name]
            assert phenotype_summary.method == "fastlmm"
            assert phenotype_summary.heritability is not None
            assert np.isclose(
                phenotype_summary.heritability,
                null_model_collection.heritability[i],
                atol=1e-4,
            )
            assert phenotype_summary.genetic_variance is not None
            assert phenotype_summary.error_variance is not None
            assert_both_close(
                phenotype_summary.genetic_variance,
                null_model_collection.genetic_variance[i],
                phenotype_summary.error_variance,
                null_model_collection.error_variance[i],
                atol=1e-4,
                rtol=1e-3,
            )
            assert np.isclose(
                phenotype_summary.mean,
                phenotype_frame[phenotype_name].mean(),
            )
            assert np.isclose(
                phenotype_summary.variance,
                np.var(phenotype_frame[phenotype_name], ddof=1),
            )

    score_path = tmp_path / f"chr{chromosome}.score.txt.zst"
    data_frame = pd.read_table(score_path, skiprows=1, compression="zstd")

    u_stat_columns = [f"{name}_stat-u" for name in phenotype_names]
    u_stat = data_frame[u_stat_columns].to_numpy()
    rmw_u_stat = rmw_score.array["U_STAT"]

    v_stat_columns = [f"{name}_stat-v" for name in phenotype_names]
    v_stat = data_frame[v_stat_columns].to_numpy()
    sqrt_v_stat = np.sqrt(v_stat)
    rmw_sqrt_v_stat = rmw_score.array["SQRT_V_STAT"]

    for i in range(len(phenotype_names)):
        rmw_finite = np.isfinite(rmw_u_stat[:, i]) & np.isfinite(rmw_sqrt_v_stat[:, i])
        rmw_missing_rate = (~rmw_finite).mean()
        finite = ~(np.isclose(u_stat[:, i], 0) & np.isclose(v_stat[:, i], 1))
        missing_rate = (~finite).mean()

        assert missing_rate <= max(rmw_missing_rate * 2, 0.1), "Missing rate is too high"

        to_compare = finite & rmw_finite

        assert check_bias(u_stat[:, i], rmw_u_stat[:, i], to_compare)
        assert check_bias(sqrt_v_stat[:, i], rmw_sqrt_v_stat[:, i], to_compare)

    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
