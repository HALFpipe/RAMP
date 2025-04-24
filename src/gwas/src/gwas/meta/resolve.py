import os
from argparse import Namespace
from dataclasses import dataclass, fields
from datetime import datetime
from functools import partial
from itertools import chain
from operator import attrgetter
from typing import Self

from IPython.lib.pretty import pretty
from tqdm.auto import tqdm
from upath import UPath

from ..compression.cache import get_last_modified, load_from_cache, save_to_cache
from ..log import logger
from ..pheno import VariableSummary
from ..summary import SummaryCollection
from ..utils.hash import hex_digest
from ..utils.multiprocessing import make_pool_or_null_context
from .alternatives import alternative
from .index import Index, parse


@dataclass(frozen=True, eq=True, kw_only=True)
class IndexArguments:
    input_directory: UPath | None

    study_paths: list[UPath]
    replace_phenotype: list[list[str]]
    remove: list[list[str]]
    replace_key: list[list[str]]
    drop_key: list[str]
    replace_value: list[list[str]]
    alternatives: list[list[str]]

    @classmethod
    def from_arguments(cls, arguments: Namespace) -> Self:
        return cls(
            input_directory=arguments.input_directory,
            study_paths=list(arguments.input_directory.glob("*_summary_statistics/")),
            replace_phenotype=arguments.replace_phenotype or list(),
            remove=arguments.remove or list(),
            replace_key=arguments.replace_key or list(),
            drop_key=arguments.drop_key or list(),
            replace_value=arguments.replace_value or list(),
            alternatives=arguments.alternative or list(),
        )


@dataclass(frozen=True, eq=True)
class IndexJob:
    tags: tuple[tuple[str, str], ...]
    metadata_path: UPath
    last_modified: datetime


def load_index_phenotypes(
    cache_path: UPath, ij: IndexJob, ia: IndexArguments
) -> list[tuple[str, str, dict[str, str]]]:
    replace_phenotype = ia.replace_phenotype
    ia_tuple = (replace_phenotype,)

    metadata_path = ij.metadata_path
    summary_collection = SummaryCollection.from_file(metadata_path)

    tags = dict(ij.tags)
    name = tags["ds"]
    cache_key = f"index-{name}-{hex_digest(ia_tuple)}-{hex_digest(ij)}"

    phenotypes: list[tuple[str, str, dict[str, str]]] | None = load_from_cache(
        cache_path, cache_key
    )
    if phenotypes is not None:
        return phenotypes

    prefix = "_".join(f"{key}-{value}" for key, value in ij.tags)

    phenotypes = list()
    for chunk in summary_collection.chunks.values():
        for variable_collection_name, variable_collection in chunk.items():
            for phenotype, summary in variable_collection.phenotypes.items():
                phenotype = f"{prefix}_{phenotype}"
                phenotype_name = phenotype
                for substring, replacement in replace_phenotype:
                    phenotype = phenotype.replace(substring, replacement)

                path = ij.metadata_path.parent
                extra_tags = dict(
                    vc=variable_collection_name,
                    path=str(path),
                    prefix=prefix,
                    # descriptive statistics
                    **{
                        key: f"{getattr(summary, key)}"
                        for key in map(attrgetter("name"), fields(VariableSummary))
                    },
                    sample_count=f"{variable_collection.sample_count}",
                )
                phenotypes.append((phenotype, phenotype_name, extra_tags))

    save_to_cache(cache_path, cache_key, phenotypes, int(os.environ["OMP_NUM_THREADS"]))

    return phenotypes


def create_index(
    arguments: Namespace, cache_path: UPath
) -> tuple[IndexArguments, Index]:
    ia = IndexArguments.from_arguments(arguments)
    index_jobs: list[IndexJob] = list()
    for study_path in tqdm(ia.study_paths, unit="studies"):
        tags = tuple(
            (key, value)
            for key, value in parse(study_path.name)
            if key not in {"suffix"}
        )
        name = dict(tags)["ds"]

        metadata_paths = sorted(study_path.rglob("chr1.metadata.yaml.gz"))
        if len(metadata_paths) == 0:
            logger.warning(
                f'No metadata files found in "{study_path}". Skipping "{name}"'
            )
            continue

        logger.info(f"Adding {study_path=} with {tags=}")

        for metadata_path in metadata_paths:
            last_modified = get_last_modified(metadata_path)
            index_jobs.append(IndexJob(tags, metadata_path, last_modified))

    cache_key = f"index-{hex_digest(ia)}-{hex_digest(index_jobs)}"

    index: Index | None = load_from_cache(cache_path, cache_key)
    if index is not None:
        return ia, index

    pool, iterator = make_pool_or_null_context(
        index_jobs,
        partial(load_index_phenotypes, cache_path, ia=ia),
        arguments.num_threads,
    )

    index = Index()
    with pool:
        for phenotype, phenotype_name, extra_tags in chain.from_iterable(
            tqdm(iterator, unit="files", total=len(index_jobs))
        ):
            index.put(phenotype, phenotype_name=phenotype_name, **extra_tags)

    update_index(ia, index, arguments.num_threads)

    save_to_cache(cache_path, cache_key, index, arguments.num_threads)

    return ia, index


def update_index(ia: IndexArguments, index: Index, num_threads: int) -> None:
    for pairs in tqdm(ia.remove, unit="removals"):
        query = dict(pair.split("=") for pair in pairs)
        for phenotype in index.get_phenotypes(**query):
            index.remove(phenotype)

    for key, replacement in tqdm(ia.replace_key, unit="key replacements"):
        index.replace_key(key, replacement)

    for key in tqdm(ia.drop_key, unit="key drops"):
        index.drop_key(key)

    for key, value, replacement in tqdm(ia.replace_value, unit="value replacements"):
        index.replace_value(key, value, replacement)

    for key in index.phenotypes_by_tags.keys():
        values = sorted(index.phenotypes_by_tags[key].keys())
        logger.debug(
            f'raw "{key}" has {len(values)} values: {pretty(values, max_seq_length=64)}'
        )

    for key, value, alt in tqdm(ia.alternatives, unit="alternatives", position=0):
        alternative(index, key, value, alt, num_threads=num_threads)
