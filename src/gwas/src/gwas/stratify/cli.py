# -*- coding: utf-8 -*-
import logging
import re
from argparse import Action, ArgumentError, ArgumentParser, Namespace
from collections import defaultdict
from itertools import product
from typing import Any, Sequence

import numpy as np
import scipy
from numpy import typing as npt

from ..log import logger
from ..pheno import read_and_combine
from .base import SampleID
from .populations import plot_populations, populations, super_populations


def write_table(
    prefix: str,
    samples: list[SampleID],
    columns: list[str],
    array: npt.NDArray,
    output_sample_ids: str = "merge_both_with_underscore",
) -> None:
    import pandas as pd

    index = [s.to_str(method=output_sample_ids) for s in samples]
    data_frame = pd.DataFrame(array, index=index, columns=columns)
    data_frame.to_csv(f"{prefix}.tsv", sep="\t", index=True)


class GroupAction(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        by_list: list[str] | None = getattr(namespace, "by", None)
        if by_list is None:
            raise ArgumentError(self, "Must specify `--by` before `--group`")
        if not isinstance(values, (list, tuple)):
            raise ArgumentError(
                self, "Must specify name, lower bound and upper bound for `--group`"
            )

        by = by_list[-1]
        name, lower_bound, upper_bound = values
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
        values = (by, name, lower_bound, upper_bound)

        items = getattr(namespace, self.dest, None)
        if items is None:
            items = list()
        items.append(values)
        setattr(namespace, self.dest, items)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)8s] %(funcName)s: "
            "%(message)s (%(filename)s:%(lineno)s)"
        ),
    )

    argument_parser = ArgumentParser()
    argument_parser.add_argument("--mds", metavar="CSV", required=True)
    argument_parser.add_argument("--covariates", nargs="+")
    argument_parser.add_argument("--phenotypes", nargs="+")

    argument_parser.add_argument("--minimum-sample-size", type=int, default=50)

    group = argument_parser.add_mutually_exclusive_group()
    group.add_argument("--by-population", action="store_true", default=False)
    group.add_argument("--by-super-population", action="store_true", default=False)
    argument_parser.add_argument("--ignore-population", nargs="+", default=list())
    argument_parser.add_argument(
        "--main-population", required=False, choices=populations, default="CEU"
    )
    argument_parser.add_argument(
        "--main-super-population",
        required=False,
        choices=super_populations.keys(),
        default="EUR",
    )
    argument_parser.add_argument("--component-count", type=int)
    argument_parser.add_argument(
        "--population-standard-deviations",
        "--population-std-devs",
        required=False,
        type=float,
        default=6,
    )

    argument_parser.add_argument(
        "--by",
        type=str,
        action="append",
    )
    argument_parser.add_argument(
        "--group",
        type=str,
        action=GroupAction,
        nargs=3,
        metavar=("NAME", "LOWER_BOUND", "UPPER_BOUND"),
    )

    argument_parser.add_argument(
        "--rename-samples",
        type=str,
        nargs=2,
        action="append",
        metavar=("FROM", "TO"),
        default=list(),
    )
    argument_parser.add_argument(
        "--rename-samples-from-file",
        type=str,
        nargs=3,
        action="append",
        metavar=("FROM_COLUMN", "TO_COLUMN", "CSV_PATH"),
        default=list(),
    )
    argument_parser.add_argument("--normalize-case", action="store_true", default=False)
    argument_parser.add_argument(
        "--normalize-special", action="store_true", default=False
    )
    argument_parser.add_argument(
        "--output-sample-ids",
        required=False,
        choices=["fid", "iid", "merge_both_with_underscore"],
        default="merge_both_with_underscore",
    )

    argument_parser.add_argument("--debug", action="store_true", default=False)

    arguments = argument_parser.parse_args()

    try:
        run(arguments)
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()


def run(arguments: Namespace) -> None:
    header: list[str] | None = None

    sample_mapping: dict[str, str] = dict()

    def rename_sample(sample: str) -> str:
        sample = sample.strip()
        for old, new in arguments.rename_samples:
            sample = re.sub(old, new, sample)
        if arguments.normalize_case:
            sample = sample.lower()
        if arguments.normalize_special:
            sample = "".join(filter(str.isalnum, sample))
        sample = sample_mapping.get(sample, sample)
        return sample

    for from_column, to_column, path in arguments.rename_samples_from_file:
        with open(path) as file_handle:
            for line in file_handle:
                header = line.strip().split(",")
                break

        if header is None:
            raise ValueError(f'No header found for "{path}"')
        from_index = header.index(from_column)
        to_index = header.index(to_column)

        with open(path) as file_handle:
            for line in file_handle:
                tokens = line.strip().split(",")
                from_token = rename_sample(tokens[from_index])
                sample_mapping[from_token] = tokens[to_index]

    # Parse the MDS file.
    with open(arguments.mds) as file_handle:
        for line in file_handle:
            header = line.strip().split(",")
            break

    if header is None:
        raise ValueError(f'No header found for "{arguments.mds}"')

    types = [object, object, int]
    component_count = len(header) - len(types)
    types += [float] * component_count
    dtype = np.dtype(list(zip(header, types)))

    matrix = np.loadtxt(arguments.mds, skiprows=1, delimiter=",", dtype=dtype)
    is_sample = ~np.isin(matrix["FID"], populations)
    samples: list[SampleID] = [
        SampleID(rename_sample(fid), rename_sample(iid))
        for fid, iid in zip(matrix["FID"], matrix["IID"])
        if fid not in populations
    ]

    sample_classes: dict[str, dict[str, set[SampleID]]] = defaultdict(
        lambda: defaultdict(set)
    )
    cutoff = scipy.stats.norm.logpdf(arguments.population_standard_deviations)

    components = np.vstack([matrix[c] for c in header[-component_count:]]).transpose()
    if arguments.component_count is not None:
        components = components[:, : arguments.component_count]
        component_count = arguments.component_count

    sample_components = components[is_sample, :]

    # Classify the samples by population.
    reference_components: dict[str, npt.NDArray] = dict()

    sample_populations = sample_classes["population"]
    for p in populations:
        reference_components[p] = components[matrix["FID"] == p, :]

    for p, c in reference_components.items():
        if p in arguments.ignore_population:
            continue
        # Fit a multivariate normal distribution to the reference components
        # with maximum likelihood estimation.
        mean = np.mean(c, axis=0)
        covariance = np.cov(c.transpose())
        full_multivariate_normal = scipy.stats.multivariate_normal(
            mean=mean, cov=covariance
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

    if len(sample_populations) > 0:
        # Plot the samples.
        plot_populations(
            cutoff,
            samples,
            reference_components,
            sample_components,
            sample_populations,
        )

    if arguments.phenotypes is None or arguments.covariates is None:
        return

    # Read and rename phenotypes and covariates.
    removed_samples: set[str] = set()

    def rename_samples(
        array_samples: list[str], array: npt.NDArray
    ) -> tuple[list[SampleID], npt.NDArray]:
        nonlocal removed_samples

        iid = [rename_sample(sample.iid) for sample in samples]
        underscored = [rename_sample(f"{fid}_{iid}") for fid, iid in samples]
        sample_indices = np.full(len(array_samples), -1, dtype=int)
        for i, sample in enumerate(array_samples):
            sample = rename_sample(sample)
            sample = rename_sample(sample)  # Run again to apply normalization.
            if sample in iid:
                sample_indices[i] = iid.index(sample)
            elif sample in underscored:
                sample_indices[i] = underscored.index(sample)

        removed_samples |= {c for c, i in zip(array_samples, sample_indices) if i == -1}
        array_samples_new = [samples[i] for i in sample_indices if i >= 0]
        array = array[sample_indices >= 0, :]

        return array_samples_new, array

    covariate_samples_base, covariate_names, covariate_array = read_and_combine(
        arguments.covariates
    )
    phenotype_samples_base, phenotype_names_base, phenotype_array = read_and_combine(
        arguments.phenotypes
    )
    covariate_samples, covariate_array = rename_samples(
        covariate_samples_base, covariate_array
    )
    phenotype_samples, phenotype_array = rename_samples(
        phenotype_samples_base, phenotype_array
    )

    if len(removed_samples) > 0:
        logger.warning(
            f"The following {len(removed_samples)} samples "
            "were not found in the MDS file: %s",
            ", ".join(sorted(removed_samples)),
        )

    # Classify the samples by groups.
    for by, name, lower_bound, upper_bound in arguments.group:
        covariate_vector = covariate_array[:, covariate_names.index(by)]

        in_group = (covariate_vector >= lower_bound) & (covariate_vector < upper_bound)
        (sample_indices,) = np.nonzero(in_group)

        for i in sample_indices:
            sample_classes[by][name].add(covariate_samples[i])

    # Convert defaultdicts to dicts.
    sample_classes = {k: dict(v) for k, v in sample_classes.items()}

    # Remove zero-length classes.
    for _, class_dict in sample_classes.items():
        keys = list(class_dict.keys())
        for k in keys:
            if len(class_dict[k]) == 0:
                del class_dict[k]

    # Create mixed sample classes.
    for _, class_dict in sample_classes.items():
        if len(class_dict) == 1:
            continue
        mixed = set.union(*class_dict.values())
        class_dict["mixed"] = mixed

    # Create non-main ancestry group.
    sample_populations = sample_classes["population"]
    if arguments.by_population:
        main = arguments.main_population
    elif arguments.by_super_population:
        main = arguments.main_super_population
    else:
        raise ValueError
    if main in sample_populations and len(sample_populations) > 1:
        mixed = set.union(*sample_populations.values())
        non_main = mixed - sample_populations[main]
        sample_populations[f"non{main}"] = non_main

    # Write the sample classes to files.
    variables = sorted(sample_classes.keys())
    values = [sorted(sample_classes[v].keys()) for v in variables]

    phenotype_names_new: list[str] = list()
    phenotype_arrays: list[npt.NDArray] = list()

    for value_tuple in product(*values):
        class_samples = set.intersection(
            *(
                sample_classes[variable][value]
                for variable, value in zip(variables, value_tuple)
            )
        )

        if len(class_samples) < arguments.minimum_sample_size:
            continue

        # Remove samples not in the class.
        class_phenotype_array = phenotype_array.copy()
        for i, sample in enumerate(phenotype_samples):
            if sample not in class_samples:
                class_phenotype_array[i, :] = np.nan
        phenotype_arrays.append(class_phenotype_array)

        prefix = "_".join(
            f"{variable}-{value}" for variable, value in zip(variables, value_tuple)
        )

        phenotype_names_new.extend(f"{prefix}_{name}" for name in phenotype_names_base)

    phenotype_array_new = np.hstack(phenotype_arrays)
    write_table(
        "phenotypes",
        phenotype_samples,
        phenotype_names_new,
        phenotype_array_new,
        output_sample_ids=arguments.output_sample_ids,
    )

    write_table(
        "covariates",
        covariate_samples,
        covariate_names,
        covariate_array,
        output_sample_ids=arguments.output_sample_ids,
    )
