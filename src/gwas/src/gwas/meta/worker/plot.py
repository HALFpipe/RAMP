from dataclasses import fields
from operator import methodcaller
from typing import Any

import numpy as np
import polars as pl
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from upath import UPath

from ...hg19 import offset
from ...pheno import VariableSummary
from ...plot.make import dark_grey, light_grey, plot_manhattan, plot_q_q
from ...utils.genetics import chromosome_to_int
from ..base import Job
from .ldsc import LDSCOutputCollection, ValueSE


def plot_pre_meta(
    job: Job, summaries: dict[str, dict[str, dict[str, float]]], output_path: UPath
) -> None:
    figure, axes = plt.subplot_mosaic(
        [["forest", "r2", "se-n"]], figsize=(12, 5), layout="constrained"
    )
    figure.suptitle(job.name.replace("_", " "), wrap=True)

    axes["forest"].set_title("Forest plot")
    plot_forest(job, axes["forest"])

    plot_r_squared(job, summaries, axes["r2"])
    axes["r2"].set_title("$R^2$ plot")
    axes["r2"].sharey(axes["forest"])
    axes["r2"].set_ylabel("")
    for label in axes["r2"].get_yticklabels():
        label.set_visible(False)

    plot_median_effect_size(job, summaries, axes["se-n"])
    axes["se-n"].set_title("SE-N plot")

    figure.savefig(output_path)
    plt.close(figure)


def format_value_and_se(x: ValueSE | str) -> str:
    return f"= {x.value} ± {x.se}" if isinstance(x, ValueSE) else x


def plot_post_meta(
    job: Job,
    summaries: dict[str, dict[str, dict[str, float]]],
    data_frame: pl.DataFrame,
    reference_population: str,
    ldsc: LDSCOutputCollection,
    output_path: UPath,
) -> None:
    split = data_frame["marker_name"].str.split(":")
    chromosome_int = (
        split.list.get(0)
        .map_elements(chromosome_to_int, return_dtype=pl.UInt8)
        .to_numpy()
    )
    position_in_chromosome = split.list.get(1).cast(pl.UInt32).to_numpy()
    position = offset[chromosome_int - 1] + position_in_chromosome
    p_value = data_frame["p_value"].to_numpy()
    log_p_value = data_frame["p_value"].log10().to_numpy()

    sample_count = data_frame["sample_count"].max()
    if not isinstance(sample_count, int):
        raise ValueError(f'Sample count has invalid type "{type(sample_count)}"')

    figure, axes = plt.subplot_mosaic(
        [
            ["manhattan", "manhattan", "manhattan", "manhattan", "qq"],
            ["forest", "forest", "se-n", "eaf", "ldsc"],
        ],
        figsize=(18, 10),
        layout="constrained",
    )
    figure.suptitle(f"{job.name.replace('_', ' ')} (n = {sample_count})")

    subfigures: dict[str, SubFigure] = dict()
    for key, _axes in axes.items():
        _axes.remove()  # clear for subfigure
        subplotspec = _axes.get_subplotspec()
        if subplotspec is None:
            raise ValueError(f'Subplotspec is "{subplotspec}" for "{key}"')
        subfigures[key] = figure.add_subfigure(subplotspec)

    subfigure = subfigures["manhattan"]
    subfigure.suptitle("Manhattan plot")
    _axes = subfigure.subplots(1, 1)
    plot_manhattan(chromosome_int, position, log_p_value, _axes)

    subfigure = subfigures["qq"]
    subfigure.suptitle("Q-Q plot")
    _axes = subfigure.subplots(1, 1)
    plot_q_q(p_value, log_p_value, _axes)

    subfigure = subfigures["forest"]
    _axes_dict = subfigure.subplot_mosaic([["forest", "r2"]], sharey=True)
    plot_forest(job, _axes_dict["forest"])
    _axes_dict["forest"].set_title("Forest plot")
    plot_r_squared(job, summaries, _axes_dict["r2"])
    _axes_dict["r2"].set_title("$R^2$ plot")
    _axes_dict["r2"].set_ylabel("")
    for label in _axes_dict["r2"].get_yticklabels():
        label.set_visible(False)

    subfigure = subfigures["se-n"]
    subfigure.suptitle("SE-N plot")
    _axes = subfigure.subplots(1, 1)
    plot_median_effect_size(job, summaries, _axes)

    subfigure = subfigures["eaf"]
    subfigure.suptitle("EAF plot")
    _axes = subfigure.subplots(1, 1)
    plot_allele_frequencies(data_frame, reference_population, _axes)

    subfigure = subfigures["ldsc"]
    subfigure.suptitle("LD Score regression")
    _axes = subfigure.subplots(1, 1)
    text = f"""munge-sumstats:
Mean $\\chi^2$ = {ldsc.munge_sumstats.mean_chisq}
$\\lambda_{{GC}}$ = {ldsc.munge_sumstats.lambda_gc}
Max $\\chi^2$ = {ldsc.munge_sumstats.max_chisq}
"""
    for key, value in ldsc.data.items():
        text += f"""ldsc {key}:
Mean $\\chi^2$ = {value.mean_chisq}
$\\lambda_{{GC}}$ = {value.lambda_gc}
Intercept {format_value_and_se(value.intercept)}
Ratio {format_value_and_se(value.ratio)}
$h^2$ {format_value_and_se(value.total_observed_scale_h2)}
"""
    _axes.text(
        0,
        1,
        text,
        horizontalalignment="left",
        verticalalignment="top",
        transform=_axes.transAxes,
    )
    _axes.set_axis_off()

    figure.savefig(output_path)
    plt.close(figure)


def plot_allele_frequencies(
    data_frame: pl.DataFrame,
    reference_population: str,
    axes: Axes,
) -> None:
    split = pl.col("marker_name").str.splitn(":", n=4)
    reference_allele1 = split.struct[2].str.to_lowercase()
    reference_allele2 = split.struct[3].str.to_lowercase()

    allele1_metal = pl.col("allele1")
    allele2_metal = pl.col("allele2")

    equal_alleles_expr = (reference_allele1 == allele1_metal) & (
        reference_allele2 == allele2_metal
    )
    swapped_alleles_expr = (reference_allele1 == allele2_metal) & (
        reference_allele2 == allele1_metal
    )

    if data_frame.filter(~(equal_alleles_expr ^ swapped_alleles_expr)).height != 0:
        raise ValueError

    data_frame = data_frame.with_columns(swapped_alleles_expr.alias("swapped_alleles"))

    freq1 = data_frame["freq1"]
    swapped_alleles = data_frame["swapped_alleles"]
    data_frame = data_frame.with_columns(
        (1 - freq1).zip_with(swapped_alleles, freq1).alias("alternate_allele_frequency")
    )

    axes.scatter(
        data_frame["reference_alternate_allele_frequency"],
        data_frame["alternate_allele_frequency"],
        facecolors=dark_grey,
        edgecolors="none",
        marker=".",
        alpha=0.8,
    )

    axes.set_xlabel(f"Alternate allele frequency from {reference_population} reference")
    axes.set_ylabel("Observed alternate allele frequency")


def plot_median_effect_size(
    job: Job, summaries: dict[str, dict[str, dict[str, float]]], axes: Axes
) -> None:
    data_frame = pl.from_dicts(
        (
            dict(
                study=name,
                sample_count=ji.sample_count,
                median_standard_error=summaries[name]["standard_error"]["median"],
            )
            for name, ji in job.inputs.items()
        )
    )

    # drop missing values
    is_finite = pl.col("sample_count", "median_standard_error").is_finite()
    data_frame = data_frame.filter(pl.all_horizontal(is_finite))

    x = data_frame["sample_count"].sqrt()
    y = 1 / data_frame["median_standard_error"]

    axes.scatter(x, y, facecolors=dark_grey)

    b, a = np.polyfit(x, y, deg=1)
    m = max(*map(methodcaller("max"), (x, y)))
    line = np.linspace(0.0, m * 1.1)
    axes.plot(line, b * line + a, c=light_grey)

    adjust_text(
        [
            axes.text(_x, _y, study, bbox=dict(facecolor="white", edgecolor="white"))
            for study, _x, _y in zip(data_frame["study"], x, y, strict=True)
        ],
        expand=(2, 2),
        arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
    )

    axes.set_xlabel(r"$\sqrt{N}$")
    axes.set_ylabel(r"$1\ /\ \text{Median standard error}$")


def plot_forest(job: Job, axes: Axes) -> None:
    data_frame = pl.from_dicts(
        dict(
            study=name,
            sample_count=ji.sample_count,
            **{field.name: getattr(ji, field.name) for field in fields(VariableSummary)},
        )
        for name, ji in job.inputs.items()
    )
    _boxplot(data_frame, axes)
    axes.set_xlabel("Phenotype")


def plot_r_squared(
    job: Job, summaries: dict[str, dict[str, dict[str, float]]], axes: Axes
) -> None:
    data_frame = pl.from_dicts(
        dict(
            study=name,
            sample_count=ji.sample_count,
            **{
                field.name: summaries[name]["r_squared"][field.name]
                for field in fields(VariableSummary)
            },
        )
        for name, ji in job.inputs.items()
    )
    _boxplot(data_frame, axes)
    axes.set_xlabel("$R^2$")


def _boxplot(data_frame: pl.DataFrame, axes: Axes) -> None:
    data_frame = data_frame.sort("sample_count", descending=False)
    axes.bxp(
        [
            dict(
                med=row["median"],
                q1=row["median"],
                q3=row["median"],
                whislo=row["lower_quartile"],
                whishi=row["upper_quartile"],
                mean=row["mean"],
                fliers=[row["minimum"], row["maximum"]],
            )
            for row in data_frame.iter_rows(named=True)
        ],
        showfliers=True,
        showbox=False,
        vert=False,
    )

    axes.set_ylabel("Study")
    studies = [
        f"""{row["study"]}\n(n = {row["sample_count"]})"""
        for row in data_frame.iter_rows(named=True)
    ]
    axes.set_yticks(range(1, len(studies) + 1), labels=studies)

    minimum: Any = data_frame["minimum"].min()
    maximum: Any = data_frame["maximum"].max()
    r = maximum - minimum
    axes.set_xlim(minimum - 0.05 * r, maximum + 0.05 * r)
    axes.axvline(0, color=light_grey)
