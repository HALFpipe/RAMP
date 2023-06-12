# -*- coding: utf-8 -*-
import gzip
from contextlib import chdir
from pathlib import Path
from subprocess import check_call
from typing import Mapping

import numpy as np
import pytest
import scipy

from gwas.eig import Eigendecomposition
from gwas.mem.wkspace import SharedWorkspace
from gwas.tri import Triangular, scale
from gwas.utils import chromosome_to_int, chromosomes_set
from gwas.vcf.base import VCFFile

from .utils import bcftools, rmw, tabix, to_bgzip

chromosomes = sorted(chromosomes_set() - {1, "X"}, key=chromosome_to_int)
minor_allele_frequency_cutoff: float = 0.05


@pytest.mark.slow
def test_eig(
    vcf_files_by_size_and_chromosome: Mapping[str, Mapping[int | str, VCFFile]],
    tri_paths_by_size_and_chromosome: Mapping[str, Mapping[str | int, Path]],
    sw: SharedWorkspace,
):
    sampe_size_label = "small"
    vcf_files = list(vcf_files_by_size_and_chromosome[sampe_size_label].values())
    tri_paths_by_chromosome = tri_paths_by_size_and_chromosome[sampe_size_label]

    tri_paths = [tri_paths_by_chromosome[c] for c in chromosomes]
    eig_array = Eigendecomposition.from_files(*tri_paths, sw=sw)

    (sample_count,) = set(v.sample_count for v in vcf_files)

    array = sw.alloc("array", sample_count, 1)
    a = array.to_numpy(include_trailing_free_memory=True)

    variant_count = 0
    for vcf_file in vcf_files:
        if vcf_file.chromosome not in chromosomes:
            continue
        vcf_file.set_samples(set(eig_array.samples))
        assert vcf_file.samples == eig_array.samples
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
    a = array.to_numpy()
    scale(a.transpose())

    # check that svd is equal
    _, scipy_singular_values, scipy_eigenvectors = scipy.linalg.svd(
        a.transpose(),
        full_matrices=False,
    )
    scipy_eigenvalues = np.square(scipy_singular_values) / variant_count
    assert np.allclose(scipy_eigenvalues, eig_array.eigenvalues)
    assert np.abs(scipy_eigenvalues - eig_array.eigenvalues).mean() < 1e-14

    # check reconstructing covariance
    c = np.cov(a, ddof=0)
    eig_c = (
        eig_array.eigenvectors * eig_array.eigenvalues
    ) @ eig_array.eigenvectors.transpose()
    assert np.allclose(c, eig_c, atol=1e-3)
    assert np.abs(c - eig_c).mean() < 1e-4

    # check that eigenvectors are just permuted
    permutation = np.rint(scipy_eigenvectors @ eig_array.eigenvectors).astype(int)
    assert ((permutation == 0) | (np.abs(permutation) == 1)).all()
    assert (1 == np.count_nonzero(permutation, axis=0)).all()
    assert (1 == np.count_nonzero(permutation, axis=1)).all()

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
    assert np.allclose(tri.transpose() @ tri / variant_count, c, atol=1e-3)

    array.free()
    eig_array.free()
    tri_array.free()
    assert len(sw.allocations) == 1


def test_eig_rmw(
    tmp_path: Path,
    vcf_files_by_size_and_chromosome: Mapping[str, Mapping[int | str, VCFFile]],
    tri_paths_by_size_and_chromosome: Mapping[str, Mapping[str | int, Path]],
    sw: SharedWorkspace,
):
    c: int | str = 22
    sampe_size_label = "small"
    vcf_file = vcf_files_by_size_and_chromosome[sampe_size_label][c]
    vcf_file.set_variants_from_cutoffs(
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
    )

    tri_array = Triangular.from_file(
        tri_paths_by_size_and_chromosome[sampe_size_label][c], sw
    )
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

    variants_path = tmp_path / f"chr{c}.variants.txt"
    with variants_path.open("wt") as file_handle:
        file_handle.write(
            "\n".join(
                (
                    ":".join(
                        (
                            str(vcf_file.chromosome),
                            str(row.position),
                            row.reference_allele,
                            row.alternate_allele,
                        )
                    )
                    for row in vcf_file.variants.itertuples()
                )
            )
        )

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
        lines = file_handle.readlines()
        header = lines.pop(0)
        samples = header.split()
        assert set(samples) == set(vcf_file.samples)
        sample_indices = [vcf_file.samples.index(s) for s in samples]
        for i, line in zip(sample_indices, lines):
            tokens = line.split()
            for j, token in zip(sample_indices, tokens):
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
    assert ((permutation == 0) | (np.abs(permutation) == 1)).all()
    assert (1 == np.count_nonzero(permutation, axis=0)).all()
    assert (1 == np.count_nonzero(permutation, axis=1)).all()
    assert np.allclose(numpy_eigenvalues[::-1], eig_array.eigenvalues, atol=1e-6)

    eig_array.free()
    assert len(sw.allocations) == 1
