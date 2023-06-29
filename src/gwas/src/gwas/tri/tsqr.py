# -*- coding: utf-8 -*-
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import numpy as np
from numpy import typing as npt

from ..log import logger
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..vcf.base import VCFFile
from .base import TaskSyncCollection, Triangular


def scale(b: npt.NDArray):
    # Calculate variant properties
    mean = b.mean(axis=1)
    minor_allele_frequency = mean / 2

    # apply scaling
    b -= mean[:, np.newaxis]
    standard_deviation = np.sqrt(
        2 * minor_allele_frequency * (1 - minor_allele_frequency)
    )
    b /= standard_deviation[:, np.newaxis]


@dataclass
class TallSkinnyQR:
    vcf_file: VCFFile
    sw: SharedWorkspace

    t: TaskSyncCollection | None = None
    variant_indices: npt.NDArray[np.uint32] | None = None

    @staticmethod
    def triangularize(shared_array: SharedArray, pivoting: bool = True) -> None:
        """Triangularize the given array to a lower triangular matrix"""
        _, sample_count = shared_array.shape
        # Triangularize to upper triangle
        if pivoting:
            pivot = shared_array.triangularize(pivoting=True)
            # Apply the inverse pivot to the columns
            shared_array.apply_inverse_pivot(pivot)
        else:
            shared_array.triangularize(pivoting=False)

        # Transpose and reshape to lower triangle
        shared_array.transpose()
        shared_array.resize(sample_count, sample_count)
        logger.debug(f"Triangularized to a dimension of {sample_count}")

    def map(self) -> Triangular:
        """Triangularize as much of the VCF file as fits into the shared
        workspace. The result is an m-by-m lower triangular matrix with
        the given name.

        Parameters
        ----------
        vcf_file : VCFFile
            vcf_file
        sw : SharedWorkspace
            sw
        minor_allele_frequency_cutoff : float
            minor_allele_frequency_cutoff

        Returns
        -------
        SharedArray | None

        """

        sample_count = self.vcf_file.sample_count
        variant_count = self.vcf_file.variant_count

        name = Triangular.get_name(self.sw, chromosome=self.vcf_file.chromosome)
        shared_array = self.sw.alloc(name, sample_count, variant_count)

        if self.variant_indices is not None:
            if self.variant_indices.size == 0:
                if self.t is not None:
                    # We have enough space to start another task in parallel
                    self.t.can_run.set()

        # Read dosages from the VCF file
        array = shared_array.to_numpy()
        logger.debug(
            f"Mapping {array.shape[1]} variants from "
            f'"{self.vcf_file.file_path.name}" into "{shared_array.name}"'
        )
        self.vcf_file.read(array.transpose())

        # Transpose, reshape and scale the data
        shared_array.resize(sample_count, variant_count)
        shared_array.transpose()
        array = shared_array.to_numpy()
        scale(array)

        multithreading_lock: ContextManager = nullcontext()
        if self.t is not None:
            multithreading_lock = self.t.multithreading_lock

        # Triangularize to lower triangle
        with multithreading_lock:
            self.triangularize(shared_array)

        return Triangular(
            name=name,
            sw=self.sw,
            chromosome=self.vcf_file.chromosome,
            samples=self.vcf_file.samples,
            variant_count=variant_count,
            minor_allele_frequency_cutoff=self.vcf_file.minor_allele_frequency_cutoff,
            r_squared_cutoff=self.vcf_file.r_squared_cutoff,
        )

    @classmethod
    def reduce(cls, *shared_arrays: Triangular) -> Triangular:
        if len(shared_arrays) == 0:
            raise ValueError

        if len(shared_arrays) == 1:
            (shared_array,) = shared_arrays
            return shared_array

        logger.debug(f"Reducing {len(shared_arrays)} chunks")

        names = [shared_array.name for shared_array in shared_arrays]

        sw = shared_arrays[0].sw
        reduce_array = sw.merge(*names)
        reduce_array.transpose()
        # Triangularize to lower triangle
        cls.triangularize(reduce_array)

        # Get metadata.
        chromosome_set = set(a.chromosome for a in shared_arrays)
        if len(chromosome_set) == 1:
            (chromosome,) = chromosome_set
        else:
            raise ValueError

        cutoffs = sorted(a.minor_allele_frequency_cutoff for a in shared_arrays)
        if np.isclose(min(cutoffs), max(cutoffs)):
            minor_allele_frequency_cutoff = cutoffs[0]
        else:
            raise ValueError

        cutoffs = sorted(a.r_squared_cutoff for a in shared_arrays)
        if np.isclose(min(cutoffs), max(cutoffs)):
            r_squared_cutoff = cutoffs[0]
        else:
            raise ValueError

        variant_count = sum(a.variant_count for a in shared_arrays)

        return Triangular(
            name=reduce_array.name,
            sw=sw,
            chromosome=chromosome,
            samples=shared_arrays[0].samples,
            variant_count=variant_count,
            minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
            r_squared_cutoff=r_squared_cutoff,
        )

    @property
    def variant_count(self) -> int:
        sample_count = self.vcf_file.sample_count
        variant_count = self.sw.unallocated_size // (
            np.float64().itemsize * sample_count
        )
        if variant_count < sample_count:
            raise MemoryError(
                f"There is only space for {variant_count:d} columns of the matrix, but "
                f"we need at least {sample_count:d} columns."
            )
        if variant_count >= self.vcf_file.variant_count:
            # We can fit the entire VCF file into memory.
            logger.debug(
                f"There is space for {variant_count:d} columns of the matrix, but "
                f"we only need {self.vcf_file.variant_count:d}."
            )
            variant_count = self.vcf_file.variant_count
        return variant_count

    def map_reduce(self) -> Triangular | None:
        self.variant_indices = self.vcf_file.variant_indices.copy()

        arrays: list[Triangular] = list()
        with self.vcf_file:
            while self.variant_indices.size > 0:
                try:
                    variant_count = self.variant_count
                    variant_indices = self.variant_indices[:variant_count]
                    self.variant_indices = self.variant_indices[variant_count:]
                    self.vcf_file.variant_indices = variant_indices

                    arrays.append(self.map())
                except MemoryError:
                    arrays = [self.reduce(*arrays)]

        return self.reduce(*arrays)
