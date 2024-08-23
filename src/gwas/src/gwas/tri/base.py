from dataclasses import dataclass, field
from multiprocessing.synchronize import Event, Semaphore
from typing import Any

import numpy as np
from numpy import typing as npt

from ..log import multiprocessing_context
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..utils.multiprocessing import SharedState
from ..vcf.base import VCFFile


def is_lower_triangular(a: npt.NDArray[Any]) -> bool:
    return np.allclose(np.triu(a, k=1), 0)


@dataclass
class Triangular(SharedArray):
    chromosome: int | str | None
    samples: list[str]
    variant_count: int

    minor_allele_frequency_cutoff: float
    r_squared_cutoff: float

    @property
    def sample_count(self) -> int:
        return self.shape[0]

    def to_file_name(self) -> str:
        return self.get_file_name(self.chromosome)

    def get_sample_indices(self, samples: list[str]) -> np.ndarray:
        return np.array(
            [self.samples.index(sample) for sample in samples], dtype=np.uint32
        )

    def subset_samples(self, samples: list[str]) -> None:
        """Reduce the lower triangular matrix to a subset of samples.
        Golub and Van Loan (1996) section 12.5.2 implements this by first removing the
        specified rows from the matrix.
        Since we are just using them for SVD, we just remove the rows and leave
        the matrix non-triangular.

        Args:
            samples (list[str]): The samples to keep.
        """
        if samples == self.samples:
            # Nothing to do
            return
        self.compress(self.get_sample_indices(samples))
        self.samples = samples

    @classmethod
    def get_file_name(cls, chromosome: int | str | None) -> str:
        if chromosome is None:
            return "tri"
        return f"chr{chromosome}.tri"

    @staticmethod
    def get_prefix(**kwargs: str | int | None) -> str:
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
    ) -> "Triangular | None":
        from .tsqr import TallSkinnyQR

        tsqr = TallSkinnyQR(
            vcf_file,
            sw,
        )
        return tsqr.map_reduce()


@dataclass
class TaskSyncCollection(SharedState):
    processes: int = field(kw_only=True)
    # Indicates that can run another task.
    can_run: Event = field(default_factory=multiprocessing_context.Event)
    # Ensures that only one multithreaded workload can run at a time.
    multithreading_semaphore: Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.multithreading_semaphore = multiprocessing_context.BoundedSemaphore(
            value=self.processes
        )
