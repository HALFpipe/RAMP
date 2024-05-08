# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import partial
from math import sqrt
from operator import attrgetter
from pathlib import Path
from typing import Self, Sequence

import numpy as np
from numpy import typing as npt
from tqdm.auto import tqdm

from ._matrix_functions import dgesvdq
from .mem.arr import SharedArray, SharedFloat64Array
from .mem.wkspace import SharedWorkspace
from .tri.base import Triangular
from .utils import (
    IterationOrder,
    chromosomes_set,
    make_pool_or_null_context,
    make_sample_boolean_vectors,
)
from .vcf.base import VCFFile


@dataclass
class Eigendecomposition(SharedFloat64Array):
    chromosome: int | str | None
    samples: list[str]
    variant_count: int

    def to_file_name(self) -> str:
        return self.get_file_name(self.chromosome)

    @classmethod
    def get_file_name(cls, chromosome: int | str | None) -> str:
        if chromosome is None:
            return "eig.txt.gz"
        else:
            return f"no-chr{chromosome}.eig"

    @staticmethod
    def get_prefix(**kwargs: str | int | None) -> str:
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
        sqrt_variant_count = sqrt(self.variant_count)
        return self.singular_values / sqrt_variant_count

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
        eigenvalues: npt.NDArray[np.float64],
        eigenvectors: npt.NDArray[np.float64],
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
        samples: list[str] | None = None,
    ) -> Self:
        if len(arrays) == 0:
            raise ValueError

        sw = arrays[0].sw

        chromosome = cls.get_chromosome(arrays, chromosome)

        if samples is None:
            samples = arrays[0].samples
        else:
            for tri in arrays:
                if samples is not None:
                    tri.subset_samples(samples)
            sw.squash()

        for tri in arrays:
            if tri.samples != samples:
                raise RuntimeError(f"Samples do not match: {tri.samples} != {samples}")

        # Concatenate triangular matrices
        tri_array = SharedFloat64Array.merge(*arrays)
        tri_array.transpose()

        a = tri_array.to_numpy()
        _, sample_count = a.shape
        variant_count = sum(tri.variant_count for tri in arrays)

        # Allocate outputs
        name = cls.get_name(sw, chromosome=chromosome)
        sw.alloc(name, sample_count, sample_count + 1)
        eig = cls(
            name=name,
            samples=samples,
            sw=sw,
            chromosome=chromosome,
            variant_count=variant_count,
        )

        # Perform singular value decomposition
        s = eig.singular_values
        v = eig.eigenvectors
        numrank = dgesvdq(a, s, v)

        # Ensure that we have full rank
        if numrank < sample_count:
            raise RuntimeError

        # The contents of the input arrays have been destroyed
        # so we remove them from the workspace
        sw.free(tri_array.name)
        sw.squash()

        # Transpose only the singular vectors to get the eigenvectors
        eig.transpose(shape=(sample_count, sample_count))

        return eig

    @classmethod
    def get_chromosome(
        cls, arrays: Sequence[Triangular], chromosome: int | str | None = None
    ) -> int | str | None:
        if chromosome is None:
            # Determine which chromosomes we are leaving out
            chromosomes = chromosomes_set()
            for tri in arrays:
                chromosomes -= {tri.chromosome}
            if len(chromosomes) > 1:
                if "X" in chromosomes:
                    # When leaving out an autosome, we usually only
                    # consider the other autosomes, so also leaving
                    # out the X chromosome is valid
                    chromosomes -= {"X"}
            if len(chromosomes) == 1:
                (chromosome,) = chromosomes
        return chromosome

    @classmethod
    def from_files(
        cls,
        *tri_paths: Path,
        sw: SharedWorkspace,
        samples: list[str] | None = None,
        chromosome: int | str | None = None,
        num_threads: int = 1,
    ) -> Self:
        load = partial(Triangular.from_file, sw=sw, dtype=np.float64)
        pool, iterator = make_pool_or_null_context(
            tri_paths,
            load,
            num_threads=num_threads,
            iteration_order=IterationOrder.UNORDERED,
        )
        with pool:
            tri_arrays = list(
                tqdm(
                    iterator,
                    total=len(tri_paths),
                    desc="loading triangular matrices",
                    unit="matrices",
                    leave=False,
                )
            )
        tri_arrays.sort(key=attrgetter("start"))

        return cls.from_tri(*tri_arrays, chromosome=chromosome, samples=samples)


@dataclass
class EigendecompositionCollection:
    chromosome: int | str | None
    samples: list[str]
    sample_boolean_vectors: list[npt.NDArray[np.bool_]]
    eigenvector_arrays: list[SharedFloat64Array]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def job_count(self) -> int:
        return len(self.eigenvector_arrays)

    def free(self) -> None:
        for array in self.eigenvector_arrays:
            array.free()

    @classmethod
    def from_eigendecompositions(
        cls,
        vcf_file: VCFFile,
        eigs: list[Eigendecomposition],
        base_samples: list[str] | None = None,
    ) -> Self:
        sw: SharedWorkspace = eigs[0].sw

        chromosomes = {eig.chromosome for eig in eigs}
        if len(chromosomes) != 1:
            raise ValueError("Eigendecompositions must be from the same chromosome")
        (chromosome,) = chromosomes

        if base_samples is None:
            base_samples = [
                sample
                for sample in vcf_file.vcf_samples
                if any(sample in eig.samples for eig in eigs)
            ]
        base_sample_count = len(base_samples)

        sample_boolean_vectors = make_sample_boolean_vectors(
            base_samples, (eig.samples for eig in eigs)
        )
        eigenvector_arrays: list[SharedFloat64Array] = list()

        for eig, sample_boolean_vector in zip(eigs, sample_boolean_vectors, strict=True):
            sample_count = len(eig.samples)

            prefix = eig.get_prefix(chromosome=eig.chromosome)
            name = SharedArray.get_name(sw, f"expanded-{prefix}")
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
