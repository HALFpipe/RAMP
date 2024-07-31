from argparse import Namespace
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt

from ..log import logger
from ..pheno import read_and_combine
from .base import SampleID
from .mds import classify_samples_by_mds, parse_mds
from .rename import SampleRenamer


def write_table(
    prefix: str,
    samples: list[SampleID],
    columns: list[str],
    array: npt.NDArray[np.float64],
    output_sample_ids: str = "merge_both_with_underscore",
) -> None:
    index = [s.to_str(method=output_sample_ids) for s in samples]
    data_frame = pd.DataFrame(array, index=index, columns=columns)
    csv_kwargs: dict[str, Any] = dict(sep="\t", index=True, na_rep="n/a")
    data_frame.to_csv(f"{prefix}.tsv", **csv_kwargs)


def stratify(arguments: Namespace, output_directory: Path) -> None:
    sample_classes: dict[str, dict[str, set[SampleID]]] = defaultdict(
        lambda: defaultdict(set)
    )
    sample_renamer = SampleRenamer(arguments)

    # Parse the MDS file
    mds = parse_mds(arguments, sample_renamer.rename_sample)
    classify_samples_by_mds(arguments, mds, sample_classes)

    if arguments.phenotypes is None or arguments.covariates is None:
        logger.warning("No phenotypes or covariates specified")
        return

    # Read and rename phenotypes and covariates
    covariate_samples_base, covariate_names, covariate_array = read_and_combine(
        arguments.covariates
    )
    phenotype_samples_base, phenotype_names_base, phenotype_array = read_and_combine(
        arguments.phenotypes
    )
    covariate_samples, covariate_array = sample_renamer.rename_samples(
        mds.samples, covariate_samples_base, covariate_array
    )
    phenotype_samples, phenotype_array = sample_renamer.rename_samples(
        mds.samples, phenotype_samples_base, phenotype_array
    )

    if len(sample_renamer.removed_samples) > 0:
        removed_samples_str = ", ".join(sorted(sample_renamer.removed_samples))
        logger.warning(
            f"The following {len(sample_renamer.removed_samples)} samples "
            f"were not found in the MDS file: {removed_samples_str}",
        )

    if arguments.add_components_to_covariates:
        covariate_sample_indices = [
            mds.samples.index(sample) for sample in covariate_samples
        ]
        covariate_array = np.hstack(
            [covariate_array, mds.sample_components[covariate_sample_indices, :]]
        )
        component_count = mds.sample_components.shape[1]
        covariate_names.extend(
            f"mds_component_{i + 1:02d}" for i in range(component_count)
        )

    # Classify the samples by groups
    classify_samples_by_group(
        arguments,
        covariate_samples,
        covariate_names,
        covariate_array,
        sample_classes,
    )

    # Apply rules to classes
    finalize_sample_classes(arguments, sample_classes)

    # Write the sample classes to files.
    phenotype_names_new, phenotype_array_new = apply_classes_to_phenotypes(
        arguments,
        sample_classes,
        phenotype_names_base,
        phenotype_array,
        phenotype_samples,
    )

    if len(phenotype_samples) == 0:
        raise ValueError(
            "No samples remaining. Please check that the sample IDs in the "
            "phenotypes and covariates files are correctly matched to the "
            "sample IDs in the MDS file and genotypes"
        )
    if len(phenotype_names_new) == 0:
        raise ValueError("No phenotypes remaining")
    if len(covariate_names) == 0:
        raise ValueError("No covariates remaining")

    write_table(
        str(output_directory / "stratified_phenotypes"),
        phenotype_samples,
        phenotype_names_new,
        phenotype_array_new,
        output_sample_ids=arguments.output_sample_ids,
    )

    write_table(
        str(output_directory / "stratified_covariates"),
        covariate_samples,
        covariate_names,
        covariate_array,
        output_sample_ids=arguments.output_sample_ids,
    )


def classify_samples_by_group(
    arguments: Namespace,
    covariate_samples: list[SampleID],
    covariate_names: list[str],
    covariate_array: npt.NDArray[np.float64],
    sample_classes: dict[str, dict[str, set[SampleID]]],
) -> None:
    for by, name, lower_bound, upper_bound in arguments.group:
        covariate_vector = covariate_array[:, covariate_names.index(by)]

        in_group = (covariate_vector >= lower_bound) & (covariate_vector < upper_bound)
        (sample_indices,) = np.nonzero(in_group)

        for i in sample_indices:
            sample_classes[by][name].add(covariate_samples[i])


def prune_classes(
    sample_classes: dict[str, dict[str, set[SampleID]]], minimum_class_size: int
) -> None:
    for by, class_dict in sample_classes.items():
        keys = list(class_dict.keys())
        for k in keys:
            if len(class_dict[k]) < minimum_class_size:
                class_info = {by: k}
                logger.info(
                    f"Will not output {class_info} because it has "
                    f"only {len(class_dict[k])} samples, which is less than "
                    f"{minimum_class_size}"
                )
                del class_dict[k]


def finalize_sample_classes(
    arguments: Namespace,
    sample_classes: dict[str, dict[str, set[SampleID]]],
) -> None:
    # Convert defaultdicts to dicts
    for k in sample_classes.keys():
        v = sample_classes[k]
        sample_classes[k] = dict(v)

    # Remove small subsamples
    prune_classes(sample_classes, arguments.minimum_subsample_size)

    # Create mixed sample classes
    for _, class_dict in sample_classes.items():
        if len(class_dict) == 1:
            continue
        mixed = set.union(*class_dict.values())
        class_dict["mixed"] = mixed

    # Remove small samples
    prune_classes(sample_classes, arguments.minimum_sample_size)

    # Create non-main ancestry group
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


def apply_classes_to_phenotypes(
    arguments: Namespace,
    sample_classes: dict[str, dict[str, set[SampleID]]],
    phenotype_names_base: list[str],
    phenotype_array: npt.NDArray[np.float64],
    phenotype_samples: list[SampleID],
) -> tuple[list[str], npt.NDArray[np.float64]]:
    variables = sorted(sample_classes.keys())
    values = [sorted(sample_classes[v].keys()) for v in variables]

    phenotype_names_new: list[str] = list()
    phenotype_arrays: list[npt.NDArray[np.float64]] = list()

    for value_tuple in product(*values):
        class_samples = set.intersection(
            *(
                sample_classes[variable][value]
                for variable, value in zip(variables, value_tuple, strict=True)
            )
        )

        class_info = dict(zip(variables, value_tuple, strict=True))
        if len(class_samples) < arguments.minimum_sample_size:
            logger.info(
                f"Will not output {class_info} because it has "
                f"only {len(class_samples)} samples, which is less than "
                f"{arguments.minimum_sample_size}"
            )
            continue
        else:
            logger.info(
                f"Will output group {class_info} with " f"{len(class_samples)} samples"
            )

        # Remove samples not in the class.
        class_phenotype_array = phenotype_array.copy()
        for i, sample in enumerate(phenotype_samples):
            if sample not in class_samples:
                class_phenotype_array[i, :] = np.nan
        phenotype_arrays.append(class_phenotype_array)

        prefix = "_".join(
            f"{variable}-{value}"
            for variable, value in zip(variables, value_tuple, strict=True)
        )

        phenotype_names_new.extend(f"{prefix}_{name}" for name in phenotype_names_base)

    phenotype_array_new = np.hstack(phenotype_arrays)
    return phenotype_names_new, phenotype_array_new
