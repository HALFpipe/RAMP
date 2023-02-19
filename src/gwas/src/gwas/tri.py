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
    ) -> Triangular:
        array, variant_count = tri_map_reduce(vcf_file, sw)

        return cls(
            name=array.name,
            sw=sw,
            chromosome=vcf_file.chromosome,
            variant_count=variant_count,
            minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
        )


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


def tri_map(
    vcf_file: VCFFile,
    sw: SharedWorkspace,
    minor_allele_frequency_cutoff: float = 0.05,
) -> tuple[SharedArray | None, int]:
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
    m = vcf_file.sample_count

    name = Triangular.get_name(sw, chromosome=vcf_file.chromosome)
    array = sw.alloc(name, m, m)

    # read dosages
    a = array.to_numpy(include_trailing_free_memory=True)

    logger.info(
        f"Mapping up to {a.shape[1]} variants from "
        f'"{vcf_file.file_path.name}" into "{name}"'
    )

    variants = vcf_file.read(
        a.transpose(),
        predicate=MinorAlleleFrequencyCutoff(
            minor_allele_frequency_cutoff,
        ),
    )
    variant_count = len(variants)

    if variant_count == 0:
        return None, variant_count

    # transpose and reshape
    array.resize(m, variant_count)
    array.transpose()
    b = array.to_numpy()

    scale(b)

    # triangularize to upper triangle
    array.triangularize()

    # transpose and reshape to lower triangle
    array.transpose()
    array.resize(m, m)

    return array, variant_count


def tri_reduce(*arrays: SharedArray) -> SharedArray:
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

    return array


def tri_map_reduce(
    vcf_file: VCFFile,
    sw: SharedWorkspace,
):
    variant_count: int = 0
    arrays: list[SharedArray] = list()
    with vcf_file:
        while True:
            array: SharedArray | None = None
            try:
                array, n = tri_map(vcf_file, sw)
                variant_count += n
            except MemoryError:
                tri_reduce(*arrays)

            if array is None:
                break

            arrays.append(array)

    return tri_reduce(*arrays), variant_count
