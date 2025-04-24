from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from filecmp import cmp
from functools import cmp_to_key, partial
from operator import attrgetter, itemgetter
from pprint import pformat
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Mapping

import yaml
from IPython.lib.pretty import pretty
from more_itertools import consume
from pyarrow import parquet as pq
from tqdm.auto import tqdm
from upath import UPath

from ..compression.pipe import CompressedTextWriter
from ..log import logger
from ..pheno import VariableSummary
from ..utils.genetics import chromosomes_list
from ..utils.multiprocessing import make_pool_or_null_context
from .base import Job, JobInput
from .groups import check_groups
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
                    index.tags_by_phenotypes[phenotype]["ds"] for phenotype in phenotypes
                )
            )
            message.append(
                f'"{key}={value}" is available for {len(studies)} studies: {studies}'
            )
            if i > 64:
                message.append("...")
                break
        logger.info("\n".join(message))

    group_by: list[str] = arguments.group_by
    groups = index.get_tag_groups(group_by)

    logger.debug(f"Grouping by {pretty(group_by)} resulted in {len(groups)} groups")

    job_path = output_directory / "jobs"
    job_path.mkdir(parents=True, exist_ok=True)

    iterable = make_job_iterator(
        index, groups, group_by, ia.input_directory, num_threads=arguments.num_threads
    )
    callable = partial(write_group, job_path=job_path, group_by=group_by, num_threads=1)

    pool, iterator = make_pool_or_null_context(iterable, callable, arguments.num_threads)
    with pool:
        consume(iterator)


def resolve_ties(
    index: Index,
    group_sets: set[frozenset[tuple[str, str | None]]],
) -> frozenset[tuple[str, str | None]]:
    if len(group_sets) == 1:
        (group_set,) = group_sets
        return group_set

    def cmp(
        group_set_a: frozenset[tuple[str, str | None]],
        group_set_b: frozenset[tuple[str, str | None]],
    ) -> int:
        group_a = dict(group_set_a)
        group_b = dict(group_set_b)
        score = 0

        for key, value_a in group_a.items():
            value_b = group_b.get(key)
            if value_b is None or value_a == value_b:
                continue
            alternatives = index.alternatives.get(key)
            if not alternatives:
                continue
            if value_b in alternatives.get(value_a, set()):
                score -= 1
            elif value_a in alternatives.get(value_b, set()):
                score += 1

        return score

    sorted_group_sets = sorted(group_sets, key=cmp_to_key(cmp))
    if cmp(sorted_group_sets[0], sorted_group_sets[1]) == 0:
        raise ValueError(
            f"Multiple groups refer to the same set of phenotypes: {pformat(group_sets)}"
        )
    return sorted_group_sets[0]


def make_job_iterator(
    index: Index,
    groups: list[Mapping[str, str | None]],
    group_by: list[str],
    input_directory: UPath | None,
    num_threads: int = 1,
) -> Iterator[tuple[Mapping[str, str | None], Job]]:
    paths = index.get_tag_values("path")
    score_paths, covariance_paths, alternate_allele_frequency_columns = get_paths(
        paths, input_directory
    )

    groups_by_phenotypes = check_groups(index, groups, num_threads=num_threads)

    for phenotypes, group_sets in tqdm(
        groups_by_phenotypes.items(), unit=" " + "phenotypes", unit_scale=True
    ):
        group_set = resolve_ties(index, group_sets)
        group = dict(group_set)

        name = index.format(
            (
                (key, group[key])
                for key in group_by
                if key in group and group[key] is not None
            )
        )

        study_counter = Counter(index.tags_by_phenotypes[p]["ds"] for p in phenotypes)
        if set(study_counter.values()) != {1}:
            duplicate_phenotypes_by_study = {
                study: {
                    p for p in phenotypes if index.tags_by_phenotypes[p]["ds"] == study
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
            try:
                tags = index.tags_by_phenotypes[phenotype]
                ds = tags["ds"]
                path = tags["path"]
                covariance_path: UPath | str | None = covariance_paths.get(path)
                if covariance_path is not None:
                    covariance_path = str(covariance_path)

                prefix = f"{tags['prefix']}_"
                if phenotype.startswith(prefix):
                    phenotype = phenotype.removeprefix(prefix)
                else:
                    raise ValueError(
                        f'Phenotype "{phenotype}" does not start with "{prefix}"'
                    )

                a = f"{tags['vc']}_alternate_allele_frequency"
                if a not in alternate_allele_frequency_columns[path]:
                    a = a.replace("vc", "variableCollection")
                    if a not in alternate_allele_frequency_columns[path]:
                        raise ValueError(
                            f'Alternate allele frequency column "{a}" '
                            f"not found in {path}"
                        )

                inputs[ds] = JobInput(
                    phenotype=phenotype,
                    alternate_allele_frequency_column_name=a,
                    score_paths=list(map(str, score_paths[path])),
                    covariance_path=covariance_path,
                    sample_count=int(tags["sample_count"]),
                    **{
                        key: float(tags[key])
                        for key in map(attrgetter("name"), fields(VariableSummary))
                    },
                )
            except ValueError as error:
                logger.warning(
                    f"Failed to parse phenotype {phenotype} in group {name}: {error}"
                )

        yield group, Job(name=name, inputs=inputs)


def write_group(
    g: tuple[Mapping[str, str | None], Job],
    job_path: UPath,
    group_by: list[str],
    num_threads: int,
) -> None:
    group, job = g

    p = job_path
    for key in group_by[:6]:
        value = group.get(key)
        if value is None:
            continue
        p /= f"{key}-{value}"
    p.mkdir(parents=True, exist_ok=True)

    data: Any = asdict(job)

    output_path = p / f"{job.name}.yaml.gz"

    with TemporaryDirectory() as temporary_directory:
        temporary_path = UPath(temporary_directory) / output_path.name
        with CompressedTextWriter(
            temporary_path, num_threads=num_threads
        ) as file_handle:
            yaml.dump(
                data=data,
                stream=file_handle,
                Dumper=yaml.CDumper,
                width=2**16 - 1,
                sort_keys=False,
            )
        if not output_path.is_file() or not cmp(
            output_path, temporary_path, shallow=False
        ):
            copyfile(temporary_path, output_path)


def get_alternate_allele_frequency_columns(path: UPath) -> set[str] | None:
    try:
        parquet_file = pq.ParquetFile(path)
        return set(
            column_name
            for column_name in parquet_file.schema.names
            if column_name.endswith("_alternate_allele_frequency")
        )
    except Exception:
        logger.warning(f'Score file "{path}" is not readable')
        return None


def get_paths(
    paths: set[str], input_directory: UPath | None
) -> tuple[dict[str, list[UPath]], dict[str, UPath], dict[str, set[str]]]:
    score_paths: dict[str, list[UPath]] = defaultdict(list)
    covariance_paths: dict[str, UPath] = dict()
    alternate_allele_frequency_columns: dict[str, set[str]] = defaultdict(set)
    for path_str in paths:
        path = UPath(path_str)

        missing_chromosomes: set[str | int] = set()
        for chromosome in chromosomes_list():
            score_file_path = path / f"chr{chromosome}.score.parquet"
            if not score_file_path.is_file():
                missing_chromosomes.add(chromosome)
                continue

            a = get_alternate_allele_frequency_columns(score_file_path)
            if a is None:
                missing_chromosomes.add(chromosome)
                continue
            alternate_allele_frequency_columns[path_str].update(a)

            if input_directory is not None:
                score_file_path = score_file_path.relative_to(input_directory)
            score_paths[path_str].append(score_file_path)

        if missing_chromosomes and missing_chromosomes != {"X"}:
            logger.warning(f'Path "{path}" is missing chromosomes {missing_chromosomes}')

        covariance_path = path / "covariance.parquet"
        if not covariance_path.is_file():
            continue
        if input_directory is not None:
            covariance_path = covariance_path.relative_to(input_directory)
        covariance_paths[path_str] = covariance_path

    return score_paths, covariance_paths, alternate_allele_frequency_columns
