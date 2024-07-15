# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from ..compression.arr.base import CompressionMethod, FileArray, FileArrayWriter
from ..compression.pipe import CompressedTextWriter
from ..eig.base import Eigendecomposition
from ..eig.calc import calc_eigendecompositions
from ..log import logger
from ..mem.arr import SharedFloat64Array
from ..mem.wkspace import SharedWorkspace
from ..null_model.calc import calc_null_model_collections
from ..pheno import VariableCollection
from ..summary import SummaryCollection
from ..utils import chromosome_to_int
from ..vcf.base import VCFFile
from .run import calc_score


@dataclass
class JobCollection:
    vcf_file: VCFFile
    chromosomes: Sequence[int | str]
    tri_paths_by_chromosome: Mapping[int | str, Path]
    null_model_method: str
    output_directory: Path
    compression_method: CompressionMethod
    num_threads: int

    variable_collection_chunks: list[list[VariableCollection]]
    summary_collection: SummaryCollection = field(init=False)
    stat_file_array: FileArrayWriter[np.float64] = field(init=False)

    sw: SharedWorkspace = field(init=False)

    def __post_init__(self) -> None:
        if len(self.variable_collection_chunks) == 0:
            raise ValueError("No phenotypes to analyze")
        self.sw = next(
            chain.from_iterable(self.variable_collection_chunks)
        ).phenotypes.sw
        # Create an array proxy
        self.stat_file_array = FileArray.create(
            self.file_path,
            (self.vcf_file.variant_count, self.phenotype_count * 2),
            np.float64,
            compression_method=self.compression_method,
            num_threads=self.num_threads,
        )
        # Set column names
        phenotype_names = [
            f"{phenotype_name}_stat-{stat}"
            for variable_collections in self.variable_collection_chunks
            for vc in variable_collections
            for phenotype_name in vc.phenotype_names
            for stat in ["u", "v"]
        ]
        self.stat_file_array.set_axis_metadata(1, pd.Series(phenotype_names))
        # Set row metadata
        self.stat_file_array.set_axis_metadata(
            0, self.vcf_file.vcf_variants.iloc[self.vcf_file.variant_indices]
        )
        # Try to load an existing summary collection.
        chunks_path = self.file_path.with_suffix(".yaml.gz")
        if chunks_path.is_file():
            try:
                summary_collection = SummaryCollection.from_file(chunks_path)
                summary_collection.validate(self.variable_collection_chunks)
                self.summary_collection = summary_collection
                return
            except ValueError as e:
                logger.warning(
                    f'Failed to load summary collection from "{chunks_path}"',
                    exc_info=e,
                )
        # Create a new summary collection.
        self.summary_collection = SummaryCollection.from_variable_collection_chunks(
            self.variable_collection_chunks
        )
        self.dump()

    def dump(self) -> None:
        value = asdict(self.summary_collection)
        with CompressedTextWriter(
            self.file_path.with_suffix(".metadata.yaml.gz")
        ) as file_handle:
            yaml.dump(value, file_handle, sort_keys=False, width=np.inf)

    def get_eigendecompositions(
        self, chromosome: int | str, variable_collections: list[VariableCollection]
    ) -> list[Eigendecomposition]:
        # Leave out current chromosome from calculation.
        other_chromosomes = sorted(
            set(self.chromosomes) - {chromosome, "X"}, key=chromosome_to_int
        )
        tri_paths = [self.tri_paths_by_chromosome[c] for c in other_chromosomes]
        samples_lists: list[list[str]] = [vc.samples for vc in variable_collections]
        # Calculate eigendecomposition and free tris.
        return calc_eigendecompositions(
            *tri_paths,
            sw=self.sw,
            samples_lists=samples_lists,
            chromosome=chromosome,
            num_threads=self.num_threads,
        )

    def run(self) -> None:
        phenotype_offset: int = 0
        for variable_collections, summaries in zip(
            self.variable_collection_chunks,
            self.summary_collection.chunks.values(),
            strict=True,
        ):
            eigendecompositions = self.get_eigendecompositions(
                self.chromosome, variable_collections
            )
            inverse_variance_arrays: list[SharedFloat64Array] = list()
            scaled_residuals_arrays: list[SharedFloat64Array] = list()

            null_model_collections = calc_null_model_collections(
                eigendecompositions,
                variable_collections,
                method=self.null_model_method,
                num_threads=self.num_threads,
            )
            for nm, summary in zip(
                null_model_collections,
                summaries.values(),
                strict=True,
            ):
                summary.put_null_model_collection(nm)
                # Extract the matrices we actually need from the null model
                (
                    inverse_variance_array,
                    scaled_residuals_array,
                ) = nm.get_arrays_for_score_calc()
                inverse_variance_arrays.append(inverse_variance_array)
                scaled_residuals_arrays.append(scaled_residuals_array)
                # Free the nm.
                nm.free()

            # Write null model results to disk
            self.dump()

            calc_score(
                self.vcf_file,
                eigendecompositions,
                inverse_variance_arrays,
                scaled_residuals_arrays,
                self.stat_file_array,
                phenotype_offset,
            )
            for array in [*inverse_variance_arrays, *scaled_residuals_arrays]:
                array.free()

            # Update status
            for summary in summaries.values():
                summary.status = "score_complete"
            self.dump()

            phenotype_offset += sum(vc.phenotype_count for vc in variable_collections)

    @property
    def chromosome(self) -> int | str:
        return self.vcf_file.chromosome

    @property
    def phenotype_count(self) -> int:
        return sum(
            vc.phenotype_count
            for vc in chain.from_iterable(self.variable_collection_chunks)
        )

    @property
    def file_path(self) -> Path:
        return self.output_directory / f"chr{self.chromosome}.score"
