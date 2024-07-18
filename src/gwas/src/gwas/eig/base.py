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

from .._matrix_functions import dgesvdq
from ..log import logger
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..tri.base import Triangular
from ..utils import (
    IterationOrder,
    chromosomes_set,
    make_pool_or_null_context,
)


@dataclass
class Eigendecomposition(SharedArray):
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

    def set_from_tri_array(self, tri_array: SharedArray) -> None:
        tri_array.transpose()
        a = tri_array.to_numpy()
        _, sample_count = a.shape

        logger.debug("Performing singular value decomposition")
        s = self.singular_values
        v = self.eigenvectors
        numrank = dgesvdq(a, s, v)

        # Ensure that we have full rank
        if numrank < sample_count:
            raise RuntimeError

        # The contents of the input arrays have been destroyed
        # so we remove them from the workspace
        tri_array.free()

        logger.debug("Transposing the right singular vectors to get the eigenvectors")
        self.transpose(shape=(sample_count, sample_count))

    @classmethod
    def empty(
        cls,
        chromosome: int | str,
        samples: list[str],
        variant_count: int,
        sw: SharedWorkspace,
    ) -> Self:
        sample_count = len(samples)
        name = cls.get_name(sw, chromosome=chromosome)
        sw.alloc(name, sample_count, sample_count + 1)
        return cls(
            name=name,
            samples=samples,
            sw=sw,
            chromosome=chromosome,
            variant_count=variant_count,
        )

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
        eig = cls.empty(chromosome, samples, variant_count, sw)

        s = eig.singular_values
        v = eig.eigenvectors

        s[:] = np.sqrt(eigenvalues * variant_count)
        v[:] = eigenvectors
        return eig

    @classmethod
    def from_tri_array(
        cls,
        tri_array: SharedArray,
        samples: list[str],
        variant_count: int,
        chromosome: int | str,
    ) -> Self:
        tri_array = tri_array
        sw = tri_array.sw

        # Allocate outputs
        eig = cls.empty(
            samples=samples,
            sw=sw,
            chromosome=chromosome,
            variant_count=variant_count,
        )
        eig.set_from_tri_array(tri_array)
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

        chromosome = cls.get_chromosome(arrays, chromosome)

        if samples is None:
            samples = arrays[0].samples
        else:
            for tri in arrays:
                if samples is not None:
                    tri.subset_samples(samples)

        for tri in arrays:
            if tri.samples != samples:
                raise RuntimeError(f"Samples do not match: {tri.samples} != {samples}")

        variant_count = sum(tri.variant_count for tri in arrays)

        sw = arrays[0].sw
        sw.squash({tri.name for tri in arrays})

        # Concatenate triangular matrices
        tri_array = SharedArray.merge(*arrays)
        return cls.from_tri_array(
            tri_array, samples, variant_count, chromosome=chromosome
        )

    @classmethod
    def get_chromosome(
        cls, arrays: Sequence[Triangular], chromosome: int | str | None = None
    ) -> int | str:
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
        if isinstance(chromosome, (int, str)):
            return chromosome
        raise ValueError(f"Invalid chromosome: {chromosome}")

    @classmethod
    def from_files(
        cls,
        *tri_paths: Path,
        sw: SharedWorkspace,
        samples: list[str] | None = None,
        chromosome: int | str | None = None,
        num_threads: int = 1,
    ) -> Self:
        sw.squash()
        tri_arrays = load_tri_arrays(tri_paths, sw, num_threads=num_threads)
        return cls.from_tri(*tri_arrays, chromosome=chromosome, samples=samples)


def load_tri_arrays(
    tri_paths: Sequence[Path], sw: SharedWorkspace, num_threads: int = 1
) -> list[Triangular]:
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
    logger.debug(f"Loaded {len(tri_arrays)} triangular matrices: {tri_arrays}")
    return tri_arrays
