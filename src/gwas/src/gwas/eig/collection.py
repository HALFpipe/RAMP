from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy import typing as npt

from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..utils import global_lock, make_sample_boolean_vectors
from ..vcf.base import VCFFile
from .base import Eigendecomposition


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
        eigenvector_arrays: list[SharedArray] = list()

        for eig, sample_boolean_vector in zip(eigs, sample_boolean_vectors, strict=True):
            sample_count = len(eig.samples)

            prefix = eig.get_prefix(chromosome=eig.chromosome)
            with global_lock:
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
