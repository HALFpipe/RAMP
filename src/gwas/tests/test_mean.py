import numpy as np
import pytest

from gwas.mean import calc_mean, make_sample_boolean_array
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from gwas.testing.simulate import generate_missing_value_patterns
from gwas.utils.threads import cpu_count
from gwas.vcf.base import VCFFile

from .conftest import ReadResult
from .utils import check_bias


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
def test_mean(
    vcf_file: VCFFile,
    numpy_read_result: ReadResult,
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
):
    allocation_names = set(sw.allocations.keys())
    new_allocation_names: set[str] = set()

    missing_value_pattern_count = 10

    np.random.seed(10)
    missing_value_patterns = generate_missing_value_patterns(
        missing_value_rate=0.1,
        missing_value_pattern_count=missing_value_pattern_count,
        sample_count=vcf_file.sample_count,
    )

    variable_collections: list[VariableCollection] = list()
    for i in range(missing_value_pattern_count):
        missing_value_pattern = missing_value_patterns[:, i]

        samples = [
            sample
            for sample, missing in zip(
                vcf_file.samples, missing_value_pattern, strict=True
            )
            if not missing
        ]

        vc = VariableCollection.from_arrays(
            samples,
            ["phenotype-0"],
            np.random.rand(len(samples), 1),
            ["covariate-0"],
            np.random.rand(len(samples), 1),
            sw,
            missing_value_strategy="listwise_deletion",
        )
        new_allocation_names.add(vc.name)
        request.addfinalizer(vc.free)
        variable_collections.append(vc)

    # Test make_sample_boolean_array
    sample_boolean_array = make_sample_boolean_array(
        variable_collections, vcf_file.samples
    )
    new_allocation_names.add(sample_boolean_array.name)
    request.addfinalizer(sample_boolean_array.free)
    np.testing.assert_array_equal(
        sample_boolean_array.to_numpy(),
        np.logical_not(
            np.hstack([np.zeros((vcf_file.sample_count, 1)), missing_value_patterns])
        ),
    )

    # Test calc_mean
    vcf_file.clear_allele_frequency_columns()
    assert calc_mean(vcf_file, variable_collections, num_threads=cpu_count())
    for column in vcf_file.shared_vcf_variants.columns:
        if column.name in vcf_file.allele_frequency_columns:
            new_allocation_names.update(column.allocation_names)
    mean_frame = vcf_file.vcf_variants[vcf_file.allele_frequency_columns]

    numpy_vcf_variants, numpy_dosages = numpy_read_result

    assert check_bias(
        mean_frame["alternate_allele_frequency"].to_numpy(),
        numpy_vcf_variants["alternate_allele_frequency"].to_numpy(),
    )

    numpy_alternate_allele_frequency = numpy_dosages.mean(axis=1) / 2
    np.testing.assert_allclose(
        mean_frame["alternate_allele_frequency"],
        numpy_alternate_allele_frequency,
    )

    for i in range(missing_value_pattern_count):
        missing_value_pattern = missing_value_patterns[:, i]
        numpy_alternate_allele_frequency = (
            numpy_dosages.mean(axis=1, where=np.logical_not(missing_value_pattern)) / 2
        )
        np.testing.assert_allclose(
            mean_frame[f"{variable_collections[i].name}_alternate_allele_frequency"],
            numpy_alternate_allele_frequency,
        )

        assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
