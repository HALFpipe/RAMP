# -*- coding: utf-8 -*-
from argparse import Namespace
from typing import Callable, NamedTuple

import numpy as np
import scipy.stats
from numpy import typing as npt

from .base import SampleID
from .populations import plot_populations, populations, super_populations


class MDS(NamedTuple):
    samples: list[SampleID]
    sample_components: npt.NDArray[np.float64]
    reference_components: dict[str, npt.NDArray[np.float64]]


def classify_samples_by_mds(
    arguments: Namespace,
    mds: MDS,
    sample_classes: dict[str, dict[str, set[SampleID]]],
) -> None:
    samples, sample_components, reference_components = mds

    cutoff = scipy.stats.norm.logpdf(arguments.population_standard_deviations).item()
    sample_populations = sample_classes["population"]

    for p, c in reference_components.items():
        if p in arguments.ignore_population:
            continue
        # Fit a multivariate normal distribution to the reference components
        # with maximum likelihood estimation.
        mean = np.mean(c, axis=0)
        covariance = np.cov(c.transpose())
        full_multivariate_normal = scipy.stats.multivariate_normal(
            mean=mean,
            cov=covariance,  # type: ignore
        )

        # Determine which samples are in the populations.
        sample_likelihoods = full_multivariate_normal.logpdf(sample_components)
        in_population = sample_likelihoods > cutoff
        (sample_indices,) = np.nonzero(in_population)

        for i in sample_indices:
            sample_populations[p].add(samples[i])

        if arguments.by_super_population:
            # Merge into super populations.
            for p, q in super_populations.items():
                if p not in sample_populations:
                    continue
                sample_populations[q].update(sample_populations.pop(p))

    # Also delete super populations
    for p in list(sample_populations.keys()):
        if p in arguments.ignore_population:
            del sample_populations[p]

    if len(sample_populations) > 0:
        # Plot the samples.
        plot_populations(
            cutoff,
            samples,
            reference_components,
            sample_components,
            sample_populations,
        )


def parse_mds(arguments: Namespace, rename_sample: Callable[[str], str]) -> MDS:
    header: list[str] | None = None
    with open(arguments.mds) as file_handle:
        for line in file_handle:
            header = line.strip().split(",")
            break

    if header is None:
        raise ValueError(f'No header found for "{arguments.mds}"')

    types = [object, object, int]
    component_count = len(header) - len(types)
    types += [float] * component_count
    dtype = np.dtype(list(zip(header, types, strict=True)))

    matrix = np.loadtxt(arguments.mds, skiprows=1, delimiter=",", dtype=dtype)
    is_sample = ~np.isin(matrix["FID"], populations)
    samples: list[SampleID] = [
        SampleID(fid, iid)
        for fid, iid in zip(matrix["FID"], matrix["IID"], strict=True)
        if fid not in populations
    ]

    components = np.vstack([matrix[c] for c in header[-component_count:]]).transpose()
    if arguments.component_count is not None:
        components = components[:, : arguments.component_count]
        component_count = arguments.component_count

    sample_components = components[is_sample, :]

    # Classify the samples by population.
    reference_components: dict[str, npt.NDArray[np.float64]] = dict()

    for p in populations:
        reference_components[p] = components[matrix["FID"] == p, :]
    return MDS(
        samples,
        sample_components,
        reference_components,
    )
