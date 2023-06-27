# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy import typing as npt

from ._matrix_functions import dgesvdq
from .mem.arr import SharedArray
from .mem.wkspace import SharedWorkspace
from .tri import Triangular
from .utils import chromosomes_set, make_sample_boolean_vectors
from .vcf.base import VCFFile


@dataclass
class Eigendecomposition(SharedArray):
    chromosome: int | str | None
    samples: list[str]
    variant_count: int

    def to_file_name(self) -> str:
        return self.get_file_name(self.chromosome)

    @staticmethod
    def get_file_name(chromosome: int | str | None) -> str:
        if chromosome is None:
            return "eig.txt.gz"
        else:
            return f"no-chr{chromosome}.eig.txt.gz"

    @staticmethod
    def get_prefix(**kwargs) -> str:
        chromosome = kwargs.get("chromosome")
        if chromosome is not None:
            return f"no-chr{chromosome}-eig"
        else:
            return "eig"

    @property
    def sample_count(self) -> int:
        return self.shape[0]

    @property
    def singular_values(self) -> npt.NDArray[np.float64]:
        a = self.to_numpy()
        return a[:, -1]

    @property
    def sqrt_eigenvalues(self) -> npt.NDArray[np.float64]:
        return self.singular_values / np.sqrt(self.variant_count)

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        """Returns the eigenvalues in descending order.

        Returns:
            npt.NDArray[np.float64]: Eigenvalues.
        """
        return np.square(self.sqrt_eigenvalues)

    @property
    def eigenvectors(self) -> npt.NDArray[np.float64]:
        """Return the n-by-n matrix of eigenvectors, where each column is an
        eigenvector.

        Returns:
            npt.NDArray[np.float64]: Eigenvectors.
        """
        a = self.to_numpy()
        return a[:, :-1]

    @classmethod
    def from_arrays(
        cls,
        chromosome: int | str,
        samples: list[str],
        variant_count: int,
        eigenvalues: npt.NDArray,
        eigenvectors: npt.NDArray,
        sw: SharedWorkspace,
    ) -> Self:
        _, sample_count = eigenvectors.shape
        name = cls.get_name(sw, chromosome=chromosome)
        sw.alloc(name, sample_count, sample_count + 1)
        eig = cls(
            name=name,
            samples=samples,
            sw=sw,
            chromosome=chromosome,
            variant_count=variant_count,
        )

        s = eig.singular_values
        v = eig.eigenvectors

        s[:] = np.sqrt(eigenvalues * variant_count)
        v[:] = eigenvectors
        return eig

    @classmethod
    def from_tri(
        cls,
        *arrays: Triangular,
        chromosome: int | str | None = None,
    ) -> Self:
        if len(arrays) == 0:
            raise ValueError

        if chromosome is None:
            # determine which chromosomes we are leaving out
            chromosomes = chromosomes_set()
            for tri in arrays:
                chromosomes -= {tri.chromosome}

            if len(chromosomes) > 1:
                if "X" in chromosomes:
                    # when leaving out an autosome, we usually only
                    # consider the other autosomes, so also leaving
                    # out the X chromosome is valid
                    chromosomes -= {"X"}

            if len(chromosomes) == 1:
                (chromosome,) = chromosomes

        samples = arrays[0].samples
        for tri in arrays:
            if tri.samples != samples:
                raise ValueError(f"Samples do not match: {tri.samples} != {samples}")

        # concatenate triangular matrices
        sw = arrays[0].sw
        tri_array = sw.merge(*(tri.name for tri in arrays))
        tri_array.transpose()

        a = tri_array.to_numpy()

        _, sample_count = a.shape
        variant_count = sum(tri.variant_count for tri in arrays)

        # allocate outputs
        name = cls.get_name(sw, chromosome=chromosome)
        sw.alloc(name, sample_count, sample_count + 1)
        eig = cls(
            name=name,
            samples=samples,
            sw=sw,
            chromosome=chromosome,
            variant_count=variant_count,
        )

        # perform high-precision singular value decomposition
        s = eig.singular_values
        v = eig.eigenvectors
        numrank = dgesvdq(a, s, v)

        # ensure that we have full rank
        if numrank < sample_count:
            raise RuntimeError

        # the contents of the input arrays have been destroyed
        # so we remove them from the workspace
        sw.free(tri_array.name)
        sw.squash()

        # transpose only the singular vectors to get the eigenvectors
        eig.transpose(shape=(sample_count, sample_count))

        return eig

    @classmethod
    def from_files(
        cls,
        *tri_paths: Path,
        sw: SharedWorkspace,
        samples: list[str] | None = None,
        chromosome: int | str | None = None,
    ) -> Self:
        tris: list[Triangular] = list()
        for tri_path in tri_paths:
            tri = Triangular.from_file(tri_path, sw)
            if samples is not None:
                tri.subset_samples(samples)
            tris.append(tri)
        return cls.from_tri(*tris, chromosome=chromosome)


@dataclass
class EigendecompositionCollection:
    chromosome: int | str | None
    samples: list[str]
    sample_boolean_vectors: list[npt.NDArray[np.bool_]]
    eigenvector_arrays: list[SharedArray]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def job_count(self) -> int:
        return len(self.eigenvector_arrays)

    @classmethod
    def from_eigendecompositions(
        cls,
        vcf_file: VCFFile,
        eigs: list[Eigendecomposition],
    ) -> Self:
        sw: SharedWorkspace = eigs[0].sw

        chromosomes = {eig.chromosome for eig in eigs}
        if len(chromosomes) != 1:
            raise ValueError("Eigendecompositions must be from the same chromosome")
        (chromosome,) = chromosomes

        base_samples = [
            sample
            for sample in vcf_file.vcf_samples
            if any(sample in eig.samples for eig in eigs)
        ]
        base_sample_count = len(base_samples)

        sample_boolean_vectors = make_sample_boolean_vectors(
            base_samples, (eig.samples for eig in eigs)
        )
        eigenvector_arrays: list[SharedArray] = list()

        for eig, sample_boolean_vector in zip(eigs, sample_boolean_vectors):
            sample_count = len(eig.samples)

            name = f"expanded-{eig.name}"
            array = sw.alloc(name, base_sample_count, sample_count)

            matrix = array.to_numpy()
            matrix[:] = 0
            matrix[sample_boolean_vector, :] = eig.eigenvectors

            eigenvector_arrays.append(array)

        return cls(
            chromosome,
            base_samples,
            sample_boolean_vectors,
            eigenvector_arrays,
        )
