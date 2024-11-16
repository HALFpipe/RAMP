from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from functools import partial
from operator import attrgetter, itemgetter
from typing import Any, Iterator, Mapping

import yaml
from IPython.lib.pretty import pretty
from more_itertools import consume
from tqdm.auto import tqdm
from upath import UPath

from ..compression.pipe import CompressedTextWriter
from ..log import logger
from ..pheno import VariableSummary
from ..utils.genetics import chromosomes_list
from ..utils.multiprocessing import make_pool_or_null_context
from .base import Job, JobInput
from .index import Index
from .resolve import create_index


def prepare(arguments: Namespace, output_directory: UPath) -> None:
    logger.info("Arguments are %s", pretty(vars(arguments)))

    cache_path = output_directory / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    ia, index = create_index(arguments, cache_path)

    for key in index.phenotypes_by_tags.keys():
        if key in index.ignore_keys:
            continue

        values = sorted(index.phenotypes_by_tags[key].items(), key=itemgetter(0))
        message: list[str] = list()
        for i, (value, phenotypes) in enumerate(values):
            studies = sorted(
                set(
                    index.tags_by_phenotypes[phenotype]["study"]
                    for phenotype in phenotypes
                )
            )
            message.append(
                f'"{key}={value}" is available for {len(studies)} studies: {studies}'
            )
            if i > 64:
                message.append("...")
                break
        logger.info("\n".join(message))

    group_by = arguments.group_by
    groups = index.get_tag_groups(group_by)

    logger.debug(f"Grouping by {pretty(group_by)} resulted in {len(groups)} groups")

    job_path = output_directory / "jobs"
    job_path.mkdir(parents=True, exist_ok=True)

    iterable = make_job_iterator(index, groups, ia.data_directory)
    callable = partial(write_group, job_path=job_path, num_threads=1)

    pool, iterator = make_pool_or_null_context(iterable, callable, arguments.num_threads)
    with pool:
        consume(iterator)


def make_job_iterator(
    index: Index, groups: list[Mapping[str, str | None]], data_directory: UPath | None
) -> Iterator[tuple[Mapping[str, str | None], Job]]:
    paths = index.get_tag_values("path")
    score_paths, covariance_paths = get_paths(paths, data_directory)

    seen: set[frozenset[str]] = set()
    for group in tqdm(groups, unit=" " + "phenotypes", unit_scale=True):
        name = index.format(**{k: v for k, v in group.items() if v is not None})
        phenotypes = frozenset(index.get_phenotypes(**group))

        if len(phenotypes) < 3:
            continue
        if phenotypes in seen:
            continue
        seen.add(phenotypes)

        study_counter = Counter(index.tags_by_phenotypes[p]["study"] for p in phenotypes)
        if set(study_counter.values()) != {1}:
            duplicate_phenotypes_by_study = {
                study: {
                    p
                    for p in phenotypes
                    if index.tags_by_phenotypes[p]["study"] == study
                }
                for study, count in study_counter.items()
                if count > 1
            }
            raise ValueError(
                f'Duplicate studies in group "{name}": '
                f"{pretty(duplicate_phenotypes_by_study)}. "
                "Please adjust the `--group-by` argument"
            )

        inputs: dict[str, JobInput] = dict()
        for phenotype in phenotypes:
            tags = index.tags_by_phenotypes[phenotype]
            study = tags["study"]
            path = tags["path"]
            covariance_path: UPath | str | None = covariance_paths.get(path)
            if covariance_path is not None:
                covariance_path = str(covariance_path)

            study_prefix = f"study-{study}_"
            if phenotype.startswith(study_prefix):
                phenotype = phenotype.removeprefix(study_prefix)
            else:
                raise ValueError

            inputs[study] = JobInput(
                phenotype=phenotype,
                variable_collection_name=tags["vc"],
                score_paths=list(map(str, score_paths[path])),
                covariance_path=covariance_path,
                sample_count=int(tags["sample_count"]),
                **{
                    key: float(tags[key])
                    for key in map(attrgetter("name"), fields(VariableSummary))
                },
            )

        yield group, Job(name=name, inputs=inputs)


def write_group(
    g: tuple[Mapping[str, str | None], Job],
    job_path: UPath,
    num_threads: int,
) -> None:
    group, job = g

    p = job_path
    for key in ["population", "age", "feature", "taskcontrast"]:
        value = group.get(key)
        if value is None:
            continue
        p /= f"{key}-{value}"
    p.mkdir(parents=True, exist_ok=True)

    data: Any = asdict(job)
    with CompressedTextWriter(
        p / f"{job.name}.yaml.gz", num_threads=num_threads
    ) as file_handle:
        yaml.dump(
            data=data,
            stream=file_handle,
            Dumper=yaml.CDumper,
            width=2**16 - 1,
            sort_keys=False,
        )


def get_paths(
    paths: set[str], data_directory: UPath | None
) -> tuple[dict[str, list[UPath]], dict[str, UPath]]:
    score_paths: dict[str, list[UPath]] = defaultdict(list)
    covariance_paths: dict[str, UPath] = dict()
    for path_str in paths:
        path = UPath(path_str)

        for chromosome in chromosomes_list():
            score_file_path = path / f"chr{chromosome}.score.parquet"
            if not score_file_path.is_file():
                continue
            if data_directory is not None:
                score_file_path = score_file_path.relative_to(data_directory)
            score_paths[path_str].append(score_file_path)

        covariance_path = path / "covariance.parquet"
        if not covariance_path.is_file():
            continue
        if data_directory is not None:
            covariance_path = covariance_path.relative_to(data_directory)
        covariance_paths[path_str] = covariance_path

    return score_paths, covariance_paths
