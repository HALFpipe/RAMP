from functools import partial

import numpy as np
from tqdm import tqdm

from .log import logger
from .mem.arr import SharedArray
from .mem.data_frame import SharedSeries
from .pheno import VariableCollection
from .utils.multiprocessing import (
    get_global_lock,
    make_pool_or_null_context,
)
from .utils.numpy import make_sample_boolean_vectors
from .vcf.base import VCFFile


def apply(
    genotypes_array: SharedArray[np.float64],
    alternate_allele_frequency_arrays: list[SharedArray[np.float64]],
    sample_boolean_array: SharedArray[np.bool_],
    shape: tuple[int, int],
    variant_slice: slice,
    i: int,
) -> None:
    genotypes_matrix = genotypes_array.to_numpy(shape=shape)
    alternate_allele_frequency = alternate_allele_frequency_arrays[i].to_numpy()
    sample_boolean_vector = sample_boolean_array.to_numpy()[:, i, np.newaxis]

    mean = alternate_allele_frequency[variant_slice]
    logger.debug(f"Calculating sum for variants {variant_slice}")
    np.sum(genotypes_matrix, axis=0, where=sample_boolean_vector, out=mean)
    sample_count = sample_boolean_vector.sum()
    logger.debug(f"Dividing by sample count {sample_count} to get mean")
    if sample_count:
        np.divide(mean, 2 * sample_count, out=mean)


def validate_samples(variable_collections, base_samples):
    for variable_collection in variable_collections:
        if not set(variable_collection.samples) <= set(base_samples):
            difference = set(variable_collection.samples) - set(base_samples)
            raise ValueError(
                f"Variable collection contains additional samples {difference} "
                "that are not selected to be read"
            )


def make_sample_boolean_array(
    variable_collections: list[VariableCollection],
    base_samples: list[str],
) -> SharedArray[np.bool_]:
    sw = variable_collections[0].sw
    sample_boolean_vectors = make_sample_boolean_vectors(
        base_samples,
        (variable_collection.samples for variable_collection in variable_collections),
    )

    all_samples_boolean_vector = np.ones(len(base_samples), dtype=np.bool_)
    sample_boolean_vectors.insert(0, all_samples_boolean_vector)

    sample_boolean_matrix = np.stack(sample_boolean_vectors).transpose()
    sample_boolean_array: SharedArray[np.bool_] = SharedArray.from_numpy(
        sample_boolean_matrix, sw
    )

    return sample_boolean_array


def calc_mean(
    vcf_file: VCFFile,
    variable_collections: list[VariableCollection],
    num_threads: int,
) -> bool:
    if len(variable_collections) == 0:
        raise ValueError("No variable collections provided")

    columns = ["alternate_allele_frequency"]
    for variable_collection in variable_collections:
        name = variable_collection.name
        if name is None:
            raise ValueError("Variable collection has no name")
        columns.append(f"{variable_collection.name}_alternate_allele_frequency")
    if vcf_file.allele_frequency_columns == columns:
        return False

    sw = variable_collections[0].sw

    vcf_variant_count = vcf_file.vcf_variant_count
    variant_indices = np.arange(vcf_variant_count, dtype=np.uint32)

    base_samples = vcf_file.samples
    validate_samples(variable_collections, base_samples)

    sample_boolean_array = make_sample_boolean_array(variable_collections, base_samples)
    _, missing_value_pattern_count = sample_boolean_array.shape

    with get_global_lock():
        alternate_allele_frequency_arrays: list[SharedArray[np.float64]] = list()
        for _ in range(missing_value_pattern_count):
            name = SharedArray.get_name(sw, prefix="alternate-allele-frequency")
            array = sw.alloc(name, vcf_variant_count)
            alternate_allele_frequency_arrays.append(array)

        per_variant_size = np.float64().itemsize * len(base_samples)
        variant_count = sw.unallocated_size // per_variant_size

        name = SharedArray.get_name(sw, prefix="genotypes")
        genotypes_array = sw.alloc(name, len(base_samples), variant_count)

    iterable = list(range(missing_value_pattern_count))

    progress_bar = tqdm(
        total=vcf_variant_count,
        unit="variants",
        leave=False,
    )
    with vcf_file, progress_bar:
        variant_offset = 0
        while len(variant_indices) > 0:
            # Read the genotypes
            vcf_file.variant_indices = variant_indices[:variant_count]
            shape = (len(base_samples), vcf_file.variant_count)
            genotypes_matrix = genotypes_array.to_numpy(shape=shape)
            vcf_file.read(genotypes_matrix.transpose())

            # Calculate the alternate allele frequency
            end = variant_offset + vcf_file.variant_count
            variant_slice = slice(variant_offset, end)

            callable = partial(
                apply,
                genotypes_array,
                alternate_allele_frequency_arrays,
                sample_boolean_array,
                shape,
                variant_slice,
            )
            pool, iterator = make_pool_or_null_context(iterable, callable, num_threads)
            with pool:
                for _ in iterator:
                    pass

            # Remove already read variant indices
            progress_bar.update(vcf_file.variant_count)
            variant_offset += vcf_file.variant_count
            variant_indices = variant_indices[variant_count:]

    genotypes_array.free()
    sample_boolean_array.free()

    # Remove old allele frequency columns
    shared_vcf_variants = vcf_file.shared_vcf_variants
    for c in shared_vcf_variants.columns:
        if c.name in vcf_file.allele_frequency_columns:
            c.free()
    shared_vcf_variants.columns = [
        c
        for c in shared_vcf_variants.columns
        if c.name not in vcf_file.allele_frequency_columns
    ]

    # Add new allele frequency columns
    shared_vcf_variants.columns.extend(
        SharedSeries(name=column, values=values)
        for column, values in zip(
            columns, alternate_allele_frequency_arrays, strict=False
        )
    )

    vcf_file.allele_frequency_columns = columns

    return True
