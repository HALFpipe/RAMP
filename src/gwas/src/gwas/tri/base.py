# -*- coding: utf-8 -*-
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event, Lock

import numpy as np
from numpy import typing as npt

from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..utils import SharedState
from ..vcf.base import VCFFile


def is_lower_triangular(a: npt.NDArray):
    return np.allclose(np.triu(a, k=1), 0)


@dataclass
class Triangular(SharedArray):
    chromosome: int | str
    samples: list[str]
    variant_count: int

    minor_allele_frequency_cutoff: float
    r_squared_cutoff: float

    @property
    def sample_count(self) -> int:
        return self.shape[0]

    def to_file_name(self) -> str:
        return self.get_file_name(self.chromosome)

    def subset_samples(self, samples: list[str], pivoting: bool = True) -> None:
        """Reduce the lower triangular matrix to a subset of samples.
        Golub and Van Loan (1996) section 12.5.2 implements this by first removing the
        specified columns from the matrix, and uses standard methods to make it
        triangular again.

        Args:
            samples (list[str]): The samples to keep.
        """
        if samples == self.samples:
            # Nothing to do.
            return

        from .tsqr import TallSkinnyQR

        new_sample_indices = [self.samples.index(sample) for sample in samples]

        # Remove samples
        self.transpose()
        array = self.to_numpy()
        array[:, : len(samples)] = array[:, new_sample_indices]
        self.resize(self.sample_count, len(samples))

        TallSkinnyQR.triangularize(self, pivoting=pivoting)

        self.samples = samples

    @staticmethod
    def get_file_name(chromosome: int | str) -> str:
        return f"chr{chromosome}.tri.txt.gz"

    @staticmethod
    def get_prefix(**kwargs) -> str:
        chromosome = kwargs.get("chromosome")
        if chromosome is not None:
            return f"chr{chromosome}-tri"
        else:
            return "tri"

    @classmethod
    def from_vcf(
        cls,
        vcf_file: VCFFile,
        sw: SharedWorkspace,
    ) -> Triangular | None:
        from .tsqr import TallSkinnyQR

        tsqr = TallSkinnyQR(
            vcf_file,
            sw,
        )
        return tsqr.map_reduce()


@dataclass
class TaskSyncCollection(SharedState):
    # Indicates that can run another task.
    can_run: Event = field(default_factory=mp.Event)
    # Ensures that only one multithreaded workload can run at a time.
    multithreading_lock: Lock = field(default_factory=mp.Lock)
