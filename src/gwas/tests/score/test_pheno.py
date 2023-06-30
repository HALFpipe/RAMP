# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection

from .simulation import missing_value_rate


def test_pheno(
    tmp_path: Path,
    sw: SharedWorkspace,
) -> None:
    np.random.seed(47)

    sample_count = 100
    phenotype_count = 16
    covariate_count = 4

    samples = [str(i) for i in range(sample_count)]
    permutation = np.random.permutation(sample_count)

    phenotype_names = [f"phenotype_{i + 1:02d}" for i in range(phenotype_count)]
    covariate_names = [f"covariate_{i + 1:02d}" for i in range(covariate_count)]

    phenotypes = np.random.rand(sample_count, phenotype_count)
    phenotypes[
        np.random.choice(
            a=[False, True],
            size=phenotypes.shape,
            p=[1 - missing_value_rate, missing_value_rate],
        )
    ] = np.nan
    covariates = np.random.rand(sample_count, covariate_count)

    phenotype_frame = pd.DataFrame(
        phenotypes[permutation, :],
        columns=phenotype_names,
        index=[samples[i] for i in permutation],
    )
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_frame.to_csv(
        phenotype_path, sep="\t", index=True, header=True, na_rep="n/a"
    )

    covariate_frame = pd.DataFrame(
        covariates[permutation, :],
        columns=covariate_names,
        index=[samples[i] for i in permutation],
    )
    covariate_path = tmp_path / "covariates.tsv"
    covariate_frame.to_csv(
        covariate_path, sep="\t", index=True, header=True, na_rep="n/a"
    )

    variable_collection = VariableCollection.from_txt(
        [phenotype_path],
        [covariate_path],
        sw,
        samples=samples,
        missing_value_strategy="listwise_deletion",
    )
    assert_array_equal(variable_collection.phenotypes.to_numpy(), phenotypes)
    assert_array_equal(variable_collection.covariates.to_numpy()[:, 1:], covariates)
    variable_collection.free()

    samples.pop(0)
    variable_collection = VariableCollection.from_txt(
        [phenotype_path],
        [covariate_path],
        sw,
        samples=samples,
        missing_value_strategy="listwise_deletion",
    )
    assert_array_equal(variable_collection.phenotypes.to_numpy(), phenotypes[1:, :])
    assert_array_equal(
        variable_collection.covariates.to_numpy()[:, 1:], covariates[1:, :]
    )
    variable_collection.free()
