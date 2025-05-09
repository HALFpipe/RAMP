from dataclasses import dataclass

import matplotlib
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from numpy import typing as npt
from upath import UPath

from gwas.mem.arr import SharedArray

from ..hg19 import (
    genome_wide_segments,
    offset,
    suggestive_log_p_value,
    suggestive_segments,
    x_ticks,
)
from ..log import logger
from ..utils.genetics import chromosomes_list
from .get import PlotJob

matplotlib.rcParams["font.family"] = "DejaVu Sans"

light_grey = "#888888"
dark_grey = "#444444"


def get_file_path(output_directory: UPath, label: str) -> UPath:
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = f"{label}_plot.png"
    file_path = output_directory / filename
    return file_path


def plot_manhattan(
    chromosome_int: npt.NDArray[np.uint8],
    position: npt.NDArray[np.int64],
    log_p_value: npt.NDArray[np.float64],
    axes: Axes,
) -> None:
    cmap = LinearSegmentedColormap.from_list(
        name="Custom cmap", colors=[light_grey, dark_grey] * 13, N=26
    )

    bounds = np.arange(26) + 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    axes.scatter(
        position,
        -log_p_value,
        c=chromosome_int,
        edgecolors="none",
        marker=".",
        alpha=0.8,
        cmap=cmap,
        norm=norm,
    )
    axes.set_xlabel("Chromosome")
    axes.set_ylabel(r"$-log_{10}{(p)}$")
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(map(str, chromosomes_list()))

    axes.set_xlim(0, offset[-1])
    log_p_value_limit = (-log_p_value).max()
    log_p_value_limit = max(log_p_value_limit, suggestive_log_p_value)
    axes.set_ylim(0, log_p_value_limit * 1.2)

    axes.add_collection(
        LineCollection(suggestive_segments, color=light_grey, linestyle="dashed")
    )
    axes.add_collection(LineCollection(genome_wide_segments, color=light_grey))


def plot_q_q(
    p_value: npt.NDArray[np.float64], log_p_value: npt.NDArray[np.float64], axes: Axes
) -> None:
    if not np.all(p_value >= 0):
        raise ValueError("p-values must be positive")
    if not np.all(p_value <= 1):
        raise ValueError("p-values must be less than or equal to 1")
    if not np.all(log_p_value <= 1):
        raise ValueError("log p-values must be less than or equal to 1")

    n = log_p_value.size

    # Get uniform distribution quantiles like in R's ppoints function
    expected_p_value = (np.arange(n) + 0.5) / n
    expected_log_p_value = -np.log10(expected_p_value)

    axes.scatter(
        expected_log_p_value,
        -np.sort(log_p_value),
        facecolors=dark_grey,
        edgecolors="none",
        marker=".",
        alpha=0.8,
    )
    axes.set_xlabel(r"Expected $-log_{10}{(p)}$")
    axes.set_ylabel(r"Observed $-log_{10}{(p)}$")

    expected_median = scipy.stats.chi2.ppf(0.5, 1)
    observed_median = np.median(scipy.stats.norm.ppf(1 - p_value / 2) ** 2)
    lambda_value = observed_median / expected_median
    axes.set_title(f"$\\lambda = {lambda_value:.3f}$")

    log_p_value_limit = (-log_p_value).max()
    log_p_value_limit = max(log_p_value_limit, expected_log_p_value.max())
    axes.set_xlim(0, log_p_value_limit * 1.1)
    axes.set_ylim(0, log_p_value_limit * 1.1)
    axes.set_aspect("equal", adjustable="datalim")

    line = np.linspace(0.0, log_p_value_limit * 1.1)
    axes.plot(line, line, c=light_grey)


@dataclass
class PlotGenerator:
    chromosome_array: SharedArray[np.uint8]
    position_array: SharedArray[np.int64]
    p_value_array: SharedArray[np.float64]
    mask_array: SharedArray[np.bool_]

    output_directory: UPath

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        phenotype_count, variant_count = self.mask_array.shape

        expected_shape: tuple[int, ...] = (phenotype_count * 2, variant_count)
        if self.p_value_array.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape} for p-value array "
                f"but got {self.p_value_array.shape}"
            )

        expected_shape = (variant_count,)
        if self.chromosome_array.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape} for chromosome array "
                f"but got {self.position_array.shape}"
            )
        if self.position_array.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape} for position array "
                f"but got {self.position_array.shape}"
            )

    def plot(self, job: PlotJob) -> None:
        logger.debug(f"Plotting {job.name}")
        try:
            self.validate()

            chromosome_int = self.chromosome_array.to_numpy()
            position = self.position_array.to_numpy()
            data = self.p_value_array.to_numpy().transpose()
            mask = self.mask_array.to_numpy().transpose()

            p_value = data[:, ::2]
            log_p_value = data[:, 1::2]

            mask = mask[:, job.phenotype_index]
            p_value = p_value[:, job.phenotype_index]
            log_p_value = log_p_value[:, job.phenotype_index]

            chromosome_int = chromosome_int[mask]
            position = position[mask]
            p_value = p_value[mask]
            log_p_value = log_p_value[mask]

            plot(
                self.output_directory,
                job.name,
                chromosome_int,
                position,
                p_value,
                log_p_value,
            )
        except Exception as e:
            logger.error(f'Error while plotting "{job.name}"', exc_info=e)


def plot(
    output_directory: UPath,
    name: str,
    chromosome_int: npt.NDArray[np.uint8],
    position: npt.NDArray[np.int64],
    p_value: npt.NDArray[np.float64],
    log_p_value: npt.NDArray[np.float64],
):
    plot_path = get_file_path(output_directory, name)
    figure, axes_array = plt.subplots(
        nrows=1,
        ncols=2,
        width_ratios=(2, 1),
        figsize=(18, 8),
        constrained_layout=True,
    )
    if not isinstance(axes_array, np.ndarray):
        raise ValueError("Expected axes_array to be a numpy array")
    (manhattan_axes, qq_axes) = axes_array
    figure.suptitle(name)

    plot_manhattan(chromosome_int, position, log_p_value, manhattan_axes)
    plot_q_q(p_value, log_p_value, qq_axes)

    figure.savefig(plot_path)
    plt.close(figure)
