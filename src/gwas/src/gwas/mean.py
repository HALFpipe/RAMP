from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from .mem.arr import SharedArray
from .mem.data_frame import SharedDataFrame
from .pheno import VariableCollection
from .utils import global_lock, make_pool_or_null_context, make_sample_boolean_vectors
from .vcf.base import VCFFile


def func(
    genotypes_array: SharedArray[np.float64],
    shape: tuple[int, int],
    alternate_allele_frequency_array: SharedArray[np.float64],
    variant_slice: slice,
    sample_boolean_array: SharedArray[np.bool_],
    i: int,
) -> None:
    genotypes_matrix = genotypes_array.to_numpy(shape=shape)
    alternate_allele_frequency_matrix = alternate_allele_frequency_array.to_numpy()
    mean = alternate_allele_frequency_matrix[variant_slice, i]
    sample_boolean_vector = sample_boolean_array.to_numpy()[:, i]
    genotypes_matrix.mean(axis=0, where=sample_boolean_vector[:, np.newaxis], out=mean)
    mean /= 2


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
):
    sw = variable_collections[0].sw
    sample_boolean_vectors = make_sample_boolean_vectors(
        base_samples,
        (variable_collection.samples for variable_collection in variable_collections),
    )

    all_samples_boolean_vector = np.ones(len(base_samples), dtype=np.bool_)
    sample_boolean_vectors.insert(0, all_samples_boolean_vector)

    sample_boolean_matrix = np.stack(sample_boolean_vectors).transpose()
    sample_boolean_array = SharedArray.from_numpy(sample_boolean_matrix, sw)

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

    variant_indices = vcf_file.variant_indices
    vcf_variant_count = len(variant_indices)

    base_samples = vcf_file.samples
    validate_samples(variable_collections, base_samples)

    sample_boolean_array = make_sample_boolean_array(variable_collections, base_samples)
    _, missing_value_pattern_count = sample_boolean_array.shape

    with global_lock:
        name = SharedArray.get_name(sw, "alternate-allele-frequency")
        alternate_allele_frequency_array = sw.alloc(
            name, vcf_variant_count, missing_value_pattern_count
        )

        per_variant_size = np.float64().itemsize * len(base_samples)
        variant_count = sw.unallocated_size // per_variant_size

        name = SharedArray.get_name(sw, "genotypes")
        genotypes_array = sw.alloc(name, len(base_samples), variant_count)

    progress_bar = tqdm(
        total=variant_count,
        unit="variants",
        desc="calculating allele frequencies",
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
            variant_slice = slice(
                variant_offset, variant_offset + vcf_file.variant_count
            )

            callable = partial(
                func,
                genotypes_array,
                shape,
                alternate_allele_frequency_array,
                variant_slice,
                sample_boolean_array,
            )
            pool, iterator = make_pool_or_null_context(
                range(missing_value_pattern_count), callable, num_threads
            )
            with pool:
                for _ in iterator:
                    pass

            # Remove already read variant indices
            progress_bar.update(vcf_file.variant_count)
            variant_offset += vcf_file.variant_count
            variant_indices = variant_indices[variant_count:]

    genotypes_array.free()
    sample_boolean_array.free()

    data_frame = pd.DataFrame(
        alternate_allele_frequency_array.to_numpy(), columns=columns
    )
    shared_data_frame = SharedDataFrame.from_pandas(data_frame, sw)
    alternate_allele_frequency_array.free()

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
    shared_vcf_variants.columns.extend(shared_data_frame.columns)

    vcf_file.allele_frequency_columns = columns

    return True
