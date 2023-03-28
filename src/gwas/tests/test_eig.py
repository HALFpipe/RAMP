# -*- coding: utf-8 -*-
import gzip
from contextlib import chdir
from subprocess import check_call

import numpy as np
import scipy

from gwas.eig import Eigendecomposition
from gwas.mem.wkspace import SharedWorkspace
from gwas.tri import Triangular, scale
from gwas.utils import MinorAlleleFrequencyCutoff, chromosome_to_int, chromosomes_set
from gwas.vcf import VCFFile

from .utils import rmw, to_bgzip

chromosomes = sorted(chromosomes_set() - {1, "X"}, key=chromosome_to_int)
minor_allele_frequency_cutoff: float = 0.05


def test_eig(vcf_files: list[VCFFile], tri_paths_by_chromosome):
    sw = SharedWorkspace.create()
    tri_arrays = [
        Triangular.from_file(tri_paths_by_chromosome[c], sw) for c in chromosomes
    ]
    eig_array = Eigendecomposition.from_tri(*tri_arrays)

    (sample_count,) = set(v.sample_count for v in vcf_files)

    array = sw.alloc("array", sample_count, 1)
    a = array.to_numpy(include_trailing_free_memory=True)

    n = 0
    for vcf_file in vcf_files:
        with vcf_file:
            variants = vcf_file.read(
                a[:, n:].transpose(),
                predicate=MinorAlleleFrequencyCutoff(
                    minor_allele_frequency_cutoff,
                ),
            )
            n += len(variants)

    array.resize(sample_count, n)
    a = array.to_numpy()
    scale(a.transpose())

    # check eigenvalues
    c = np.cov(a, ddof=0)
    numpy_eigenvalues, numpy_eigenvectors = np.linalg.eigh(c)
    assert np.allclose(
        numpy_eigenvalues[::-1],
        eig_array.eigenvalues,
        atol=1e-3,
        rtol=1e-3,
    )
    assert np.abs(numpy_eigenvalues[::-1] - eig_array.eigenvalues).mean() < 1e-4

    # check reconstructing covariance
    eig_c = (
        eig_array.eigenvectors * eig_array.eigenvalues
    ) @ eig_array.eigenvectors.transpose()
    assert np.allclose(c, eig_c, atol=1e-3)

    # check that eigenvectors are just permuted
    permutation = np.rint(numpy_eigenvectors.transpose() @ eig_array.eigenvectors)
    assert np.all(
        np.isclose(permutation, 0)
        | np.isclose(permutation, 1)
        | np.isclose(permutation, -1)
    )
    assert np.all(1 == np.count_nonzero(permutation, axis=0))
    assert np.all(1 == np.count_nonzero(permutation, axis=1))

    # check that qr is equal
    (numpy_tri,) = scipy.linalg.qr(a.transpose(), mode="r")

    tri_arrays = [
        Triangular.from_file(tri_paths_by_chromosome[c], sw) for c in chromosomes
    ]
    tri_array = sw.merge(*(tri.name for tri in tri_arrays))
    (tri,) = scipy.linalg.qr(tri_array.to_numpy().transpose(), mode="r")
    assert np.allclose(
        tri.transpose() @ tri,
        numpy_tri.transpose() @ numpy_tri,
    )

    # check that svd is equal
    _, scipy_singular_values, _ = scipy.linalg.svd(
        a.transpose(),
        full_matrices=False,
        lapack_driver="gesvd",
    )
    scipy_scaled_eigenvalues = np.square(scipy_singular_values / np.sqrt(a.shape[1]))
    assert np.allclose(
        numpy_eigenvalues[::-1],
        scipy_scaled_eigenvalues,
        rtol=1e-3,
    )
    assert np.allclose(scipy_scaled_eigenvalues, eig_array.eigenvalues)
    assert np.abs(scipy_scaled_eigenvalues - eig_array.eigenvalues).mean() < 1e-14

    sw.close()
    sw.unlink()


def test_eig_rmw(tmp_path, vcf_by_chromosome, tri_paths_by_chromosome):
    sw = SharedWorkspace.create()

    c: int | str = 22
    vcf_file = vcf_by_chromosome[c]

    tri_array = Triangular.from_file(tri_paths_by_chromosome[c], sw)
    eig_array = Eigendecomposition.from_tri(
        tri_array,
        chromosome=c,
    )

    vcf_zst_path = vcf_file.file_path
    vcf_gz_path = to_bgzip(tmp_path, vcf_zst_path)

    ped_path = tmp_path / f"chr{c}.ped"
    with ped_path.open("wt") as file_handle:
        for sample in vcf_file.samples:
            file_handle.write(f"{sample} {sample} 0 0 1 0\n")

    dat_path = tmp_path / f"chr{c}.dat"
    with dat_path.open("wt") as file_handle:
        file_handle.write("T variable")

    with chdir(tmp_path):
        check_call(
            [
                rmw,
                "--ped",
                str(ped_path),
                "--dat",
                str(dat_path),
                "--vcf",
                str(vcf_gz_path),
                "--kinGeno",
                "--kinSave",
                "--kinOnly",
                "--dosage",
                "--noPhoneHome",
            ]
        )

    sample_count = vcf_file.sample_count
    kinship = np.zeros((sample_count, sample_count))
    with gzip.open(tmp_path / "Empirical.Kinship.gz", "rt") as file_handle:
        lines = file_handle.readlines()
        header = lines.pop(0)
        assert header.split() == vcf_file.samples
        for i, line in enumerate(lines):
            tokens = line.split()
            for j, token in enumerate(tokens):
                kinship[i, j] = float(token)
                kinship[j, i] = float(token)

    eig_c = (
        eig_array.eigenvectors * eig_array.eigenvalues
    ) @ eig_array.eigenvectors.transpose()
    assert np.allclose(eig_c, kinship)

    sw.close()
    sw.unlink()
