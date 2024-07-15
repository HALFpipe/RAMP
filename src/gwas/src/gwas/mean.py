# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

from .mem.arr import SharedArray
from .pheno import VariableCollection
from .utils import make_sample_boolean_vectors
from .vcf.base import VCFFile


def calc_mean(
    vcf_file: VCFFile,
    variable_collections: list[VariableCollection],
) -> bool:
    if len(variable_collections) == 0:
        raise ValueError("No variable collections provided")

    columns = [
        f"{variable_collection.name}_alternate_allele_frequency"
        for variable_collection in variable_collections
    ]
    columns.insert(0, "alternate_allele_frequency")
    if vcf_file.allele_frequency_columns == columns:
        return False

    sw = variable_collections[0].sw

    variant_indices = vcf_file.variant_indices.copy()
    vcf_variant_count = len(variant_indices)

    base_samples = vcf_file.samples
    for variable_collection in variable_collections:
        if not set(variable_collection.samples) <= set(base_samples):
            difference = set(variable_collection.samples) - set(base_samples)
            raise ValueError(
                f"Variable collection contains additional samples {difference} "
                "that are not selected to be read"
            )
    sample_count = vcf_file.sample_count

    sample_boolean_vectors = make_sample_boolean_vectors(
        base_samples,
        (variable_collection.samples for variable_collection in variable_collections),
    )
    sample_boolean_vectors.insert(0, np.ones(sample_count, dtype=np.bool_))
    missing_value_pattern_count = len(sample_boolean_vectors)

    name = SharedArray.get_name(sw, "alternate-allele-frequency")
    alternate_allele_frequency_array = sw.alloc(
        name, vcf_variant_count, missing_value_pattern_count
    )
    alternate_allele_frequency_matrix = alternate_allele_frequency_array.to_numpy()

    per_variant_size = np.float64().itemsize * sample_count
    variant_count = sw.unallocated_size // per_variant_size

    name = SharedArray.get_name(sw, "genotypes")
    genotypes_array = sw.alloc(name, sample_count, variant_count)

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
            genotypes_matrix = genotypes_array.to_numpy(
                shape=(sample_count, vcf_file.variant_count)
            )
            vcf_file.read(genotypes_matrix.transpose())

            # Calculate the alternate allele frequency
            variant_slice = slice(
                variant_offset, variant_offset + vcf_file.variant_count
            )
            for i, sample_boolean_vector in enumerate(sample_boolean_vectors):
                mean = genotypes_matrix.mean(
                    axis=0, where=sample_boolean_vector[:, np.newaxis]
                )
                alternate_allele_frequency_matrix[variant_slice, i] = mean / 2

            # Remove already read variant indices
            progress_bar.update(vcf_file.variant_count)
            variant_offset += vcf_file.variant_count
            variant_indices = variant_indices[variant_count:]

    alternate_allele_frequency_frame = pd.DataFrame(
        alternate_allele_frequency_matrix, columns=columns
    )
    vcf_variants = vcf_file.vcf_variants
    vcf_variants = vcf_variants.drop(
        axis="columns", labels=vcf_file.allele_frequency_columns
    )
    vcf_variants = pd.concat(
        [vcf_variants, alternate_allele_frequency_frame],
        axis="columns",
    )
    vcf_file.vcf_variants = vcf_variants
    vcf_file.allele_frequency_columns = columns

    alternate_allele_frequency_array.free()
    genotypes_array.free()

    return True
