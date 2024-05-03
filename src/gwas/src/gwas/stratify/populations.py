# -*- coding: utf-8 -*-

from itertools import product
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist
from numpy import typing as npt
from scipy.stats import multivariate_normal

from .base import SampleID

populations = [
    "ASW",
    "CEU",
    "CHB",
    "CHD",
    "GIH",
    "JPT",
    "LWK",
    "MEX",
    "MKK",
    "TSI",
    "YRI",
]

population_descriptions = {
    "ASW": "African ancestry in Southwest USA",
    "CEU": "Utah residents with N/W European ancestry",
    "CHB": "Han Chinese in Beijing, China",
    "CHD": "Chinese in Metropolitan Denver, Colorado",
    "GIH": "Gujarati Indians in Houston, Texas",
    "JPT": "Japanese in Tokyo, Japan",
    "LWK": "Luhya in Webuye, Kenya",
    "MEX": "Mexican ancestry in Los Angeles, California",
    "MKK": "Maasai in Kinyawa, Kenya",
    "TSI": "Toscani in Italia",
    "YRI": "Yoruba in Ibadan, Nigeria",
    "EUR": "European",
    "AFR": "African",
    "EAS": "East Asian",
    "AMR": "Ad Mixed American",
    "SAS": "South Asian",
}

super_populations = {
    "ASW": "AFR",
    "CEU": "EUR",
    "CHB": "EAS",
    "CHD": "EAS",
    "GIH": "SAS",
    "JPT": "EAS",
    "LWK": "AFR",
    "MEX": "AMR",
    "MKK": "AFR",
    "TSI": "EUR",
    "YRI": "AFR",
}


def plot_populations(
    cutoff: float,
    samples: list[SampleID],
    reference_components: dict[str, npt.NDArray[np.float64]],
    sample_components: npt.NDArray[np.float64],
    sample_populations: dict[str, set[SampleID]],
) -> None:
    """Save a pairplot of the populations.

    Args:
        cutoff (float): The cutoff of the log-pdf of the multivariate normal.
        samples (list[SampleID]): The samples as tuples of FID and IID.
        reference_components (dict[str, npt.NDArray[np.float64]]): The MDS components for
            the reference.
        sample_components (npt.NDArray[np.float64]): The MDs components for the samples.
        sample_populations (dict[str, set[SampleID]]): The samples classified by
            population.
    """
    import matplotlib
    from matplotlib import pyplot as plt

    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    component_count = sample_components.shape[1]
    k = component_count - 1

    # Populations
    figure, axes = plt.subplots(
        ncols=k,
        nrows=k,
        layout="constrained",
        figsize=(6 * k, 6 * k),
        dpi=300,
    )
    gridspec = axes[1][0].get_gridspec()
    for ax in axes[-1][:-1]:
        ax.remove()
    ax_bottom_left = figure.add_subplot(gridspec[-1, :-1])
    ax_bottom_left.axis("off")

    populations_in_sample = {p for p, s in sample_populations.items() if len(s) > 0}
    if populations_in_sample.issubset(populations):
        populations_to_plot: list[str] = populations
    else:
        populations_to_plot = sorted(set(super_populations.values()))

    color_palette, sample_colors = calc_sample_colors(
        samples, sample_populations, populations_to_plot
    )

    artists: Sequence[Artist | None] = list()
    for c1, c2 in product(range(component_count), repeat=2):
        if c2 >= c1:
            continue

        if (c1 - 1) != c2:
            ax = axes[c1 - 1][c2]
            ax.axis("off")

        ax = axes[c2][c1 - 1]
        ax.scatter(
            sample_components[:, c1],
            sample_components[:, c2],
            c=sample_colors,
            alpha=0.1,
        )
        ax.set_xlabel(f"Component {c1 + 1:d}")
        ax.set_ylabel(f"Component {c2 + 1:d}")

        # Prepare the contour plots.
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x, y = np.meshgrid(
            np.linspace(x_min, x_max, num=128),
            np.linspace(y_min, y_max, num=128),
        )
        coordinates = np.dstack((x, y))

        z: dict[str, npt.NDArray[np.float64]] = calc_z(
            reference_components, sample_populations, c1, c2, coordinates
        )
        prune_z(populations_to_plot, z)

        artists = list()
        for j, super_population in enumerate(populations_to_plot):
            if super_population not in z:
                artists.append(None)
                continue
            contour_set = ax.contour(
                x, y, z[super_population], levels=[cutoff], colors=[color_palette[j]]
            )
            (artist,), _ = contour_set.legend_elements()
            if artist is None:
                continue
            artists.append(artist)

    population_labels = [
        f"{population} ({population_descriptions[population]})"
        for population in populations_to_plot
    ]
    ax_bottom_left.legend(
        handles=artists,
        labels=population_labels,
        fontsize="xx-large",
    )

    figure.savefig("populations.pdf")
    plt.close("all")


Color = str | tuple[float, float, float]


def calc_sample_colors(
    samples: list[SampleID],
    sample_populations: dict[str, set[SampleID]],
    populations_to_plot: list[str],
) -> tuple[Sequence[Color], Sequence[Color]]:
    import seaborn as sns

    color_palette: Sequence[Color] = sns.color_palette("husl", len(populations_to_plot))
    if not isinstance(color_palette, list):
        raise ValueError
    sample_colors: list[Color] = list()
    for sample in samples:
        matches = {p for p in populations_to_plot if sample in sample_populations[p]}
        if len(matches) == 1:
            (p,) = matches
            sample_colors.append(color_palette[populations_to_plot.index(p)])
        else:
            sample_colors.append("black")
    return color_palette, sample_colors


def calc_z(
    reference_components: dict[str, npt.NDArray[np.float64]],
    sample_populations: dict[str, set[SampleID]],
    c1: int,
    c2: int,
    coordinates: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    z: dict[str, npt.NDArray[np.float64]] = dict()
    for population in populations:
        super_population = super_populations[population]
        if (
            len(sample_populations[population]) == 0
            and len(sample_populations[super_population]) == 0
        ):
            continue

            # Plot contours.
        c = reference_components[population][:, [c1, c2]]
        mean = np.mean(c, axis=0)
        covariance = np.cov(c.transpose())
        marginal_multivariate_normal = multivariate_normal(
            mean=mean,
            cov=covariance,
        )
        z[population] = marginal_multivariate_normal.logpdf(coordinates)
    return z


def prune_z(
    populations_to_plot: list[str], z: dict[str, npt.NDArray[np.float64]]
) -> None:
    if populations_to_plot != populations:
        for population, super_population in super_populations.items():
            if population not in z:
                continue
            if super_population not in z:
                z[super_population] = z.pop(population)
            else:
                z[super_population] = np.minimum(z[super_population], z.pop(population))
