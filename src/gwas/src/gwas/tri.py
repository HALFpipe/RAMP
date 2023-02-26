# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from .log import logger
from .mem.arr import SharedArray
from .mem.wkspace import SharedWorkspace
from .utils import MinorAlleleFrequencyCutoff
from .vcf import VCFFile


@dataclass
class Triangular(SharedArray):
    chromosome: int | str
    variant_count: int
    minor_allele_frequency_cutoff: float = 0.05

    def to_file_name(self) -> str:
        return f"chr{self.chromosome}.tri.txt.gz"

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
        minor_allele_frequency_cutoff: float = 0.05,
    ) -> Triangular | None:
        tsqr = TallSkinnyQR(
            vcf_file,
            sw,
            minor_allele_frequency_cutoff,
        )
        return tsqr.map_reduce()


def scale(b: npt.NDArray):
    # calculate variant properties
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
    minor_allele_frequency_cutoff: float = 0.05

    def map(self) -> Triangular | None:
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

        name = Triangular.get_name(self.sw, chromosome=self.vcf_file.chromosome)
        array = self.sw.alloc(name, sample_count, sample_count)

        # read dosages
        a = array.to_numpy(include_trailing_free_memory=True)

        logger.info(
            f"Mapping up to {a.shape[1]} variants from "
            f'"{self.vcf_file.file_path.name}" into "{name}"'
        )

        variants = self.vcf_file.read(
            a.transpose(),
            predicate=MinorAlleleFrequencyCutoff(
                self.minor_allele_frequency_cutoff,
            ),
        )
        variant_count = len(variants)

        if variant_count == 0:
            self.sw.free(name)
            return None

        # transpose and reshape
        array.resize(sample_count, variant_count)
        array.transpose()
        b = array.to_numpy()

        scale(b)

        # triangularize to upper triangle
        array.triangularize()

        # transpose and reshape to lower triangle
        array.transpose()
        array.resize(sample_count, sample_count)

        return Triangular(
            name=name,
            sw=self.sw,
            chromosome=self.vcf_file.chromosome,
            variant_count=variant_count,
            minor_allele_frequency_cutoff=self.minor_allele_frequency_cutoff,
        )

    @staticmethod
    def reduce(*arrays: Triangular) -> Triangular:
        if len(arrays) == 0:
            raise ValueError

        if len(arrays) == 1:
            (array,) = arrays
            return array

        logger.info(f"Reducing {len(arrays)} chunks")

        names = [a.name for a in arrays]

        sw = arrays[0].sw
        array = sw.merge(*names)

        # triangularize to upper triangle
        array.transpose()
        array.triangularize()

        # transpose and reshape
        array.transpose()

        m = array.shape[0]
        array.resize(m, m)

        # get metadata
        chromosome_set = set(a.chromosome for a in arrays)
        if len(chromosome_set) == 1:
            (chromosome,) = chromosome_set
        else:
            raise ValueError

        cutoffs = sorted(a.minor_allele_frequency_cutoff for a in arrays)
        if np.isclose(min(cutoffs), max(cutoffs)):
            minor_allele_frequency_cutoff = cutoffs[0]
        else:
            raise ValueError

        variant_count = sum(a.variant_count for a in arrays)

        return Triangular(
            name=array.name,
            sw=sw,
            chromosome=chromosome,
            variant_count=variant_count,
            minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
        )

    def map_reduce(self) -> Triangular | None:
        arrays: list[Triangular] = list()
        with self.vcf_file:
            while True:
                array: Triangular | None = None
                try:
                    array = self.map()
                except MemoryError:
                    arrays = [self.reduce(*arrays)]

                if array is None:
                    break

                arrays.append(array)

        return self.reduce(*arrays)
