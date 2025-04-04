import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt
from numpy.testing import assert_array_equal
from pytest import FixtureRequest
from upath import UPath

from gwas.compression.arr.base import (
    Blosc2CompressionMethod,
    FileArray,
    compression_methods,
)
from gwas.covar import calc_and_save_covariance
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from gwas.utils.threads import cpu_count

from ..utils import check_bias
from .simulation import missing_value_rate

try:
    import blosc2 as blosc2
except ImportError:
    pass

sample_count = 100
phenotype_count = 16
covariate_count = 4

samples = [f"{i + 1:03d}" for i in range(sample_count)]

phenotype_names = [f"phenotype_{i + 1:02d}" for i in range(phenotype_count)]
covariate_names = [f"covariate_{i + 1:02d}" for i in range(covariate_count)]


@pytest.fixture(scope="session")
def permutation() -> npt.NDArray[np.int_]:
    np.random.seed(46)
    return np.random.permutation(sample_count)


@pytest.fixture(scope="session")
def phenotypes() -> npt.NDArray[np.float64]:
    np.random.seed(47)
    phenotypes = np.random.rand(sample_count, phenotype_count)
    phenotypes[
        np.random.choice(
            a=[False, True],
            size=phenotypes.shape,
            p=[1 - missing_value_rate, missing_value_rate],
        )
    ] = np.nan
    return phenotypes


@pytest.fixture(scope="session")
def covariates() -> npt.NDArray[np.float64]:
    np.random.seed(48)
    return np.random.rand(sample_count, covariate_count)


@pytest.fixture(scope="session")
def phenotype_path(
    phenotypes: npt.NDArray[np.float64],
    permutation: npt.NDArray[np.int_],
    tmp_path_factory: pytest.TempPathFactory,
) -> UPath:
    tmp_path = UPath(tmp_path_factory.mktemp("phenotypes"))
    phenotype_frame = pd.DataFrame(
        phenotypes[permutation, :],
        columns=phenotype_names,
        index=[samples[i] for i in permutation],
    )
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_frame.to_csv(
        phenotype_path, sep="\t", index=True, header=True, na_rep="n/a"
    )
    return phenotype_path


@pytest.fixture(scope="session")
def covariate_path(
    covariates: npt.NDArray[np.float64],
    permutation: npt.NDArray[np.int_],
    tmp_path_factory: pytest.TempPathFactory,
) -> UPath:
    tmp_path = UPath(tmp_path_factory.mktemp("covariates"))
    covariate_frame = pd.DataFrame(
        covariates[permutation, :],
        columns=covariate_names,
        index=[samples[i] for i in permutation],
    )
    covariate_path = tmp_path / "covariates.tsv"
    covariate_frame.to_csv(
        covariate_path, sep="\t", index=True, header=True, na_rep="n/a"
    )
    return covariate_path


def test_pheno(
    phenotypes: npt.NDArray[np.float64],
    covariates: npt.NDArray[np.float64],
    phenotype_path: UPath,
    covariate_path: UPath,
    sw: SharedWorkspace,
    request: FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())

    variable_collection0 = VariableCollection.from_txt(
        [phenotype_path],
        [covariate_path],
        sw,
        samples=samples,
        missing_value_strategy="listwise_deletion",
    )
    request.addfinalizer(variable_collection0.free)
    assert variable_collection0.samples == samples
    assert variable_collection0.sample_count == sample_count
    assert variable_collection0.covariate_names[1:] == covariate_names
    assert variable_collection0.covariate_count == covariate_count + 1
    assert variable_collection0.phenotype_names == phenotype_names
    assert variable_collection0.phenotype_count == phenotype_count
    assert_array_equal(variable_collection0.phenotypes, phenotypes)
    assert_array_equal(variable_collection0.covariates[:, 1:], covariates)

    truncated_samples = samples[1:]
    variable_collection1 = VariableCollection.from_txt(
        [phenotype_path],
        [covariate_path],
        sw,
        samples=truncated_samples,
        missing_value_strategy="listwise_deletion",
    )
    request.addfinalizer(variable_collection1.free)
    assert variable_collection1.samples == truncated_samples
    assert variable_collection1.sample_count == sample_count - 1
    assert variable_collection1.covariate_names[1:] == covariate_names
    assert variable_collection1.covariate_count == covariate_count + 1
    assert variable_collection1.phenotype_names == phenotype_names
    assert variable_collection1.phenotype_count == phenotype_count
    assert_array_equal(variable_collection1.phenotypes, phenotypes[1:, :])
    assert_array_equal(variable_collection1.covariates[:, 1:], covariates[1:, :])

    truncated_phenotypes = phenotype_names[1:]
    variable_collection2 = variable_collection0.copy(
        samples=truncated_samples, phenotype_names=truncated_phenotypes
    )
    request.addfinalizer(variable_collection2.free)
    assert variable_collection1.samples == truncated_samples
    assert variable_collection2.sample_count == sample_count - 1
    assert variable_collection1.covariate_names[1:] == covariate_names
    assert variable_collection1.covariate_count == covariate_count + 1
    assert variable_collection2.phenotype_names == truncated_phenotypes
    assert variable_collection2.phenotype_count == phenotype_count - 1
    assert_array_equal(variable_collection2.phenotypes, phenotypes[1:, 1:])
    assert_array_equal(variable_collection2.covariates[:, 1:], covariates[1:, :])

    new_allocation_names = {
        variable_collection0.name,
        variable_collection1.name,
        variable_collection2.name,
    }
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


def test_pheno_zero_variance(
    sw: SharedWorkspace,
    request: FixtureRequest,
) -> None:
    np.random.seed(47)

    allocation_names = set(sw.allocations.keys())

    phenotypes = np.random.rand(sample_count, phenotype_count)
    covariates = np.random.rand(sample_count, covariate_count)
    covariates[:, 2] = 1337

    variable_collection = VariableCollection.from_arrays(
        samples, phenotype_names, phenotypes, covariate_names, covariates, sw
    )
    request.addfinalizer(variable_collection.free)

    assert variable_collection.sample_count == sample_count
    assert variable_collection.covariates.shape == (sample_count, covariate_count)
    assert variable_collection.covariate_names == [
        "intercept",
        "covariate_01",
        "covariate_02",
        "covariate_04",
    ]

    new_allocation_names = {
        variable_collection.name,
    }
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


@pytest.mark.parametrize("compression_method_name", compression_methods.keys())
def test_covariance(
    compression_method_name: str,
    phenotype_path: UPath,
    covariate_path: UPath,
    sw: SharedWorkspace,
    tmp_path: UPath,
    request: FixtureRequest,
) -> None:
    compression_method = compression_methods[compression_method_name]
    if isinstance(compression_method, Blosc2CompressionMethod):
        if "blosc2" not in sys.modules:
            pytest.skip("blosc2 not installed")

    allocation_names = set(sw.allocations.keys())

    variable_collection = VariableCollection.from_txt(
        [phenotype_path],
        [covariate_path],
        sw,
        samples=samples,
        missing_value_strategy="listwise_deletion",
    )
    request.addfinalizer(variable_collection.free)

    covariance_path = tmp_path / "covariance.tsv"
    covariance_path = calc_and_save_covariance(
        variable_collection, covariance_path, compression_method, num_threads=cpu_count()
    )
    reader = FileArray.from_file(covariance_path, np.float64, num_threads=cpu_count())
    with reader:
        covariance = reader[:, :]

    kwargs: dict[str, Any] = dict(sep="\t", index_col=0, header=0)
    pandas_frame = pd.read_table(str(covariate_path), **kwargs).combine_first(
        pd.read_table(str(phenotype_path), **kwargs)
    )
    pandas_frame.insert(0, "intercept", 1.0)
    pandas_covariance_frame = pandas_frame.cov()
    pandas_covariance = pandas_covariance_frame.to_numpy()
    check_bias(covariance, pandas_covariance)

    array = variable_collection.to_numpy()
    c = np.ma.array(array, mask=np.isnan(array))
    numpy_covariance = np.ma.cov(c, rowvar=False, allow_masked=True).filled(np.nan)
    np.testing.assert_allclose(covariance, numpy_covariance)

    assert pandas_covariance_frame.columns.tolist() == variable_collection.names

    new_allocation_names = {variable_collection.name}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
