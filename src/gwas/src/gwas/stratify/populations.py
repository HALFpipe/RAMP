# -*- coding: utf-8 -*-

from itertools import product

import numpy as np
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
    reference_components: dict[str, npt.NDArray],
    sample_components: npt.NDArray,
    sample_populations: dict[str, set[SampleID]],
) -> None:
    """Save a pairplot of the populations.

    Args:
        cutoff (float): The cutoff of the log-pdf of the multivariate normal.
        samples (list[SampleID]): The samples as tuples of FID and IID.
        reference_components (dict[str, npt.NDArray]): The MDS components for the
            reference.
        sample_components (npt.NDArray): The MDs components for the samples.
        sample_populations (dict[str, set[SampleID]]): The samples classified by
            population.
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

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
    gridspec = axes[1][0].get_gridspec()  # type: ignore
    for ax in axes[-1][:-1]:
        ax.remove()
    ax_bottom_left = figure.add_subplot(gridspec[-1, :-1])
    ax_bottom_left.axis("off")

    populations_in_sample = {p for p, s in sample_populations.items() if len(s) > 0}
    if populations_in_sample.issubset(populations):
        populations_to_plot = populations
    else:
        populations_to_plot = sorted(set(super_populations.values()))

    color_palette = sns.color_palette("husl", len(populations_to_plot))
    if not isinstance(color_palette, list):
        raise ValueError
    sample_colors = list()
    for i, sample in enumerate(samples):
        matches = {p for p in populations_to_plot if sample in sample_populations[p]}
        if len(matches) == 1:
            (p,) = matches
            sample_colors.append(color_palette[populations_to_plot.index(p)])
        else:
            sample_colors.append("black")

    artists: list = list()
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

        z: dict[str, npt.NDArray] = dict()
        for j, p in enumerate(populations):
            q = super_populations[p]
            if len(sample_populations[p]) == 0 and len(sample_populations[q]) == 0:
                continue

            # Plot contours.
            c = reference_components[p][:, [c1, c2]]
            mean = np.mean(c, axis=0)
            covariance = np.cov(c.transpose())
            marginal_multivariate_normal = multivariate_normal(
                mean=mean, cov=covariance
            )
            z[p] = marginal_multivariate_normal.logpdf(coordinates)

        if populations_to_plot != populations:
            for p, q in super_populations.items():
                if p not in z:
                    continue
                if q not in z:
                    z[q] = z.pop(p)
                else:
                    z[q] = np.minimum(z[q], z.pop(p))

        artists = list()
        for j, q in enumerate(populations_to_plot):
            if q not in z:
                artists.append(None)
                continue
            contour_set = ax.contour(
                x, y, z[q], levels=[cutoff], colors=[color_palette[j]]
            )
            (artist,), _ = contour_set.legend_elements()
            artists.append(artist)

    population_labels = [
        f"{p} ({population_descriptions[p]})" for p in populations_to_plot
    ]
    ax_bottom_left.legend(
        handles=artists,
        labels=population_labels,
        fontsize="xx-large",
    )

    figure.savefig("populations.pdf")
    plt.close("all")
