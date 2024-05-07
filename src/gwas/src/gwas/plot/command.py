# -*- coding: utf-8 -*-
from argparse import Namespace
from pathlib import Path

from tqdm.auto import tqdm

from ..mem.wkspace import SharedWorkspace
from ..utils import IterationOrder, make_pool_or_null_context
from .get import DataLoader
from .make import PlotGenerator, get_file_path
from .resolve import resolve_score_files


def plot(arguments: Namespace, output_directory: Path, sw: SharedWorkspace) -> None:
    input_directory = Path(arguments.input_directory)
    num_threads = arguments.num_threads

    phenotype_names: list[str] = []
    with Path(arguments.phenotype_list).open("rt") as file_handle:
        for line in file_handle:
            phenotype_names.append(line.strip())

    phenotype_names = [
        phenotype_name
        for phenotype_name in phenotype_names
        if not get_file_path(output_directory, phenotype_name).exists()
    ]

    phenotypes, score_files, variant_metadata = resolve_score_files(
        input_directory, phenotype_names
    )
    data_loader = DataLoader(
        phenotypes=phenotypes,
        score_files=score_files,
        variant_metadata=variant_metadata,
        sw=sw,
        num_threads=num_threads,
        minor_allele_frequency_cutoff=arguments.minor_allele_frequency_cutoff,
        r_squared_cutoff=arguments.r_squared_cutoff,
    )
    plot_generator = PlotGenerator(
        chromosome_array=data_loader.chromosome_array,
        position_array=data_loader.position_array,
        p_value_array=data_loader.data_array,
        mask_array=data_loader.mask_array,
        output_directory=output_directory,
    )

    with tqdm(
        total=len(phenotypes),
        desc="plotting",
        unit="phenotypes",
    ) as progress_bar:
        for chunk in data_loader.run():
            pool, iterator = make_pool_or_null_context(
                chunk,
                plot_generator.plot,
                num_threads=num_threads,
                iteration_order=IterationOrder.UNORDERED,
            )
            with pool:
                for _ in iterator:
                    progress_bar.update(1)

    data_loader.free()
