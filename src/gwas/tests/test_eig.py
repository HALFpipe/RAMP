# -*- coding: utf-8 -*-
import gzip
from contextlib import chdir
from multiprocessing import cpu_count
from pathlib import Path
from random import sample, seed
from subprocess import check_call
from typing import Mapping, Sequence

import numpy as np
import pytest
import scipy
from gwas.eig.base import Eigendecomposition
from gwas.eig.multiple import calc_eigendecompositions
from gwas.log import logger
from gwas.mem.arr import SharedArray, SharedFloat64Array
from gwas.mem.wkspace import SharedWorkspace
from gwas.tri.base import Triangular
from gwas.tri.tsqr import scale
from gwas.vcf.base import VCFFile

from .conftest import chromosomes
from .utils import bcftools, rmw, tabix, to_bgzip

minor_allele_frequency_cutoff: float = 0.05


def load_genotypes(
    vcf_files: list[VCFFile],
    samples: list[str],
    chromosomes: Sequence[int | str],
    sw: SharedWorkspace,
) -> SharedFloat64Array:
    allocation_names = set(sw.allocations.keys())

    vcf_file = vcf_files[0]
    sample_count = vcf_file.sample_count

    name = SharedArray.get_name(sw, "genotypes")
    array = sw.alloc(name, sample_count, 1)
    a = array.to_numpy(include_trailing_free_memory=True)

    variant_count = 0
    for vcf_file in vcf_files:
        if vcf_file.chromosome not in chromosomes:
            logger.debug("Skipping chromosome %s", vcf_file.chromosome)
            continue
        vcf_file.set_variants_from_cutoffs(
            minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
        )
        with vcf_file:
            start = variant_count
            end = variant_count + vcf_file.variant_count
            vcf_file.read(
                a[:, start:end].transpose(),
            )
            variant_count += vcf_file.variant_count

    array.resize(sample_count, variant_count)
    array.transpose()

    a = array.to_numpy()
    scale(a)

    sample_count = len(samples)
    sample_indices = [vcf_file.samples.index(sample) for sample in samples]
    a[:, :sample_count] = a[:, sample_indices]
    array.resize(variant_count, sample_count)

    new_allocation_names = {name}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
    return array


@pytest.mark.slow
@pytest.mark.parametrize("subset_proportion", [0.8, 1])
def test_eig(
    subset_proportion: float,
    vcf_files_by_size_and_chromosome: Mapping[str, Mapping[int | str, VCFFile]],
    tri_paths_by_size_and_chromosome: Mapping[str, Mapping[str | int, Path]],
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())

    sample_size_label = "small"
    chromosomes: Sequence[int | str] = [7, 8, 9, 10]

    vcf_files = list(vcf_files_by_size_and_chromosome[sample_size_label].values())
    tri_paths_by_chromosome = tri_paths_by_size_and_chromosome[sample_size_label]
    vcf_file = vcf_files[0]
    samples = vcf_file.samples
    if not np.isclose(subset_proportion, 1):
        seed(42)
        subset = set(sample(samples, k=int(len(samples) * subset_proportion)))
        samples = [sample for sample in samples if sample in subset]  # Keep order

    array = load_genotypes(
        vcf_files,
        samples,
        chromosomes,
        sw,
    )
    request.addfinalizer(array.free)

    variant_count, _ = array.shape
    a = array.to_numpy().transpose()
    c = np.cov(a, ddof=0)

    _, scipy_singular_values, scipy_eigenvectors = scipy.linalg.svd(
        a.transpose(),
        full_matrices=False,
    )

    # Check that QR is equal
    (numpy_tri,) = scipy.linalg.qr(a.transpose(), mode="r")

    tri_arrays = [
        Triangular.from_file(tri_paths_by_chromosome[c], sw, np.float64)
        for c in chromosomes
    ]
    if samples != vcf_file.samples:
        for tri in tri_arrays:
            tri.subset_samples(samples)
    sw.squash()
    tri_array = SharedFloat64Array.merge(*tri_arrays)
    request.addfinalizer(tri_array.free)

    _, tri_singular_values, _ = scipy.linalg.svd(
        tri_array.to_numpy().transpose(),
        full_matrices=False,
    )
    assert np.allclose(scipy_singular_values, tri_singular_values)

    (tri_r,) = scipy.linalg.qr(tri_array.to_numpy().transpose(), mode="r")
    assert np.allclose(
        tri_r.transpose() @ tri_r,
        numpy_tri.transpose() @ numpy_tri,
    )
    assert np.allclose(tri_r.transpose() @ tri_r / variant_count, c, atol=1e-3)

    tri_paths = [tri_paths_by_chromosome[c] for c in chromosomes]
    eig_array = Eigendecomposition.from_files(
        *tri_paths, sw=sw, samples=samples, chromosome="X"
    )
    request.addfinalizer(eig_array.free)

    scipy_eigenvalues = np.square(scipy_singular_values) / variant_count
    assert np.allclose(scipy_eigenvalues, eig_array.eigenvalues)
    assert np.abs(scipy_eigenvalues - eig_array.eigenvalues).mean() < 1e-14

    # Check reconstructing covariance
    eig_c = (
        eig_array.eigenvectors * eig_array.eigenvalues
    ) @ eig_array.eigenvectors.transpose()
    assert np.allclose(c, eig_c, atol=1e-3)
    assert np.abs(c - eig_c).mean() < 1e-4

    # Check that eigenvectors are just permuted
    permutation = np.rint(scipy_eigenvectors @ eig_array.eigenvectors).astype(int)
    assert np.logical_or(permutation == 0, np.abs(permutation) == 1).all()
    assert (1 == np.count_nonzero(permutation, axis=0)).all()
    assert (1 == np.count_nonzero(permutation, axis=1)).all()

    new_allocation_names = {array.name, tri_array.name, eig_array.name}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


def test_eig_rmw(
    tmp_path: Path,
    vcf_files_by_size_and_chromosome: Mapping[str, Mapping[int | str, VCFFile]],
    tri_paths_by_size_and_chromosome: Mapping[str, Mapping[str | int, Path]],
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())

    c: int | str = 22
    sample_size_label = "small"
    vcf_file = vcf_files_by_size_and_chromosome[sample_size_label][c]
    vcf_file.set_variants_from_cutoffs(
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
    )

    tri_array = Triangular.from_file(
        tri_paths_by_size_and_chromosome[sample_size_label][c], sw, np.float64
    )
    eig_array = Eigendecomposition.from_tri(
        tri_array,
        chromosome=c,
    )
    request.addfinalizer(eig_array.free)

    vcf_zst_path = vcf_file.file_path
    vcf_gz_path = to_bgzip(tmp_path, vcf_zst_path)

    ped_path = tmp_path / f"chr{c}.ped"
    with ped_path.open("wt") as file_handle:
        for s in vcf_file.samples:
            file_handle.write(f"{s} {s} 0 0 1 0\n")

    dat_path = tmp_path / f"chr{c}.dat"
    with dat_path.open("wt") as file_handle:
        file_handle.write("T variable")

    variants_path = tmp_path / f"chr{c}.variants.txt"
    with variants_path.open("wt") as file_handle:
        variants_lines = (
            ":".join(
                map(
                    str,
                    (
                        vcf_file.chromosome,
                        row.position,
                        row.reference_allele,
                        row.alternate_allele,
                    ),
                )
            )
            for row in vcf_file.variants.itertuples()
        )
        file_handle.write("\n".join(variants_lines))

    with chdir(tmp_path):
        filtered_vcf_path = tmp_path / f"chr{c}.filt.vcf.gz"
        check_call(
            [
                bcftools,
                "view",
                "--include",
                f"ID=@{variants_path}",
                "--output-type",
                "z",
                "--output-file",
                str(filtered_vcf_path),
                str(vcf_gz_path),
            ]
        )

        check_call(
            [
                tabix,
                "-p",
                "vcf",
                str(filtered_vcf_path),
            ]
        )

        check_call(
            [
                rmw,
                "--ped",
                str(ped_path),
                "--dat",
                str(dat_path),
                "--vcf",
                str(filtered_vcf_path),
                "--kinGeno",
                "--kinSave",
                "--kinOnly",
                "--kinMaf",
                "0.00",  # We already filtered in the previous step.
                "--dosage",
                "--noPhoneHome",
            ]
        )

    sample_count = vcf_file.sample_count
    kinship = np.zeros((sample_count, sample_count))
    with gzip.open(tmp_path / "Empirical.Kinship.gz", "rt") as file_handle:
        kinship_lines = file_handle.readlines()
        header = kinship_lines.pop(0)
        samples = header.split()
        assert set(samples) == set(vcf_file.samples)
        sample_indices = [vcf_file.samples.index(s) for s in samples]
        for i, line in zip(sample_indices, kinship_lines, strict=True):
            tokens = line.split()
            for j, token in zip(sample_indices, tokens, strict=False):
                kinship[i, j] = float(token)
                kinship[j, i] = float(token)

    eig_c = (
        eig_array.eigenvectors * eig_array.eigenvalues
    ) @ eig_array.eigenvectors.transpose()
    assert np.allclose(eig_c, kinship, atol=1e-6)

    numpy_eigenvalues, numpy_eigenvectors = np.linalg.eigh(kinship)
    permutation = np.rint(
        numpy_eigenvectors.transpose() @ eig_array.eigenvectors
    ).astype(int)
    assert np.logical_or(permutation == 0, np.abs(permutation) == 1).all()
    assert (1 == np.count_nonzero(permutation, axis=0)).all()
    assert (1 == np.count_nonzero(permutation, axis=1)).all()
    assert np.allclose(numpy_eigenvalues[::-1], eig_array.eigenvalues, atol=1e-6)

    new_allocation_names = {eig_array.name}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


def test_eig_mp(
    tri_paths_by_size_and_chromosome: Mapping[str, Mapping[str | int, Path]],
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())

    sample_size_label = "small"

    tri_paths_by_chromosome = tri_paths_by_size_and_chromosome[sample_size_label]
    tri_paths = [tri_paths_by_chromosome[c] for c in chromosomes if c != "X"]

    eig_array = Eigendecomposition.from_files(*tri_paths, sw=sw, num_threads=2)
    request.addfinalizer(eig_array.free)

    assert np.isfinite(eig_array.eigenvalues).all()
    assert np.isfinite(eig_array.eigenvectors).all()

    new_allocation_names = {eig_array.name}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


def test_eig_multiple(
    tri_paths_by_size_and_chromosome: Mapping[str, Mapping[str | int, Path]],
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())
    new_allocation_names: set[str] = set()

    sample_size_label = "small"

    tri_paths_by_chromosome = tri_paths_by_size_and_chromosome[sample_size_label]
    tri_paths = [tri_paths_by_chromosome[c] for c in chromosomes if c != "X"]

    tri_array = Triangular.from_file(tri_paths[0], sw, np.float64)
    request.addfinalizer(tri_array.free)
    new_allocation_names.add(tri_array.name)
    samples = tri_array.samples

    k = int(0.8 * len(samples))
    n = 5
    samples_lists = [sample(samples, k=k) for _ in range(n)]

    eigendecompositions = calc_eigendecompositions(
        *tri_paths,
        sw=sw,
        samples_lists=samples_lists,
        chromosome="X",
        num_threads=cpu_count(),
    )
    for eig in eigendecompositions:
        request.addfinalizer(eig.free)
        new_allocation_names.add(eig.name)

        assert not np.isclose(eig.eigenvalues, 0).all()
        assert not np.isclose(eig.eigenvectors, 0).all()

        assert np.isfinite(eig.eigenvalues).all()
        assert np.isfinite(eig.eigenvectors).all()

    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
