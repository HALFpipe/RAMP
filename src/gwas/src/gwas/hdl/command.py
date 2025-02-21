import os
from argparse import Namespace
from dataclasses import fields
from functools import partial
from operator import attrgetter
from typing import Any

from more_itertools import consume
from upath import UPath

from ..compression.cache import get_last_modified, load_from_cache, save_to_cache
from ..log import logger
from ..mem.wkspace import SharedWorkspace
from ..utils.hash import hex_digest
from ..utils.jax import setup_jax
from ..utils.multiprocessing import make_pool_or_null_context
from .calc import HDL
from .load import Data, load
from .ml import eig_count as eig_count

suffix = ".sumstats.gz"


def load_field(
    sw: SharedWorkspace, path: UPath, cache_key: str, key: str
) -> tuple[str, Any]:
    return key, load_from_cache(path, f"{cache_key}-{key}", sw)


def load_fields(
    sw: SharedWorkspace, arguments: Namespace, cache_key: str
) -> Data | None:
    keys = [field.name for field in fields(Data)]
    callable = partial(load_field, sw, arguments.output_path, cache_key)
    pool, iterator = make_pool_or_null_context(
        keys, callable, num_threads=arguments.num_threads
    )
    with pool:
        data_fields = dict(iterator)
    if all(value is not None for value in data_fields.values()):
        return Data(**data_fields)
    else:
        return None


def save_field(path: UPath, cache_key: str, item: tuple[str, Any]) -> None:
    key, value = item
    save_to_cache(
        path,
        f"{cache_key}-{key}",
        value,
        num_threads=int(os.environ["OMP_NUM_THREADS"]),
        compression_level=11,
    )


def save_fields(arguments: Namespace, cache_key: str, data: Data) -> None:
    callable = partial(save_field, arguments.output_path, cache_key)
    items = [(field.name, getattr(data, field.name)) for field in fields(Data)]
    pool, iterator = make_pool_or_null_context(
        items, callable, num_threads=arguments.num_threads
    )
    with pool:
        consume(iterator)


def hdl(sw: SharedWorkspace, arguments: Namespace) -> None:
    ld_path: UPath = arguments.ld_path
    input_path: UPath = arguments.input_path

    setup_jax()

    sumstats_paths = sorted(
        input_path.rglob(f"*genomic-sem{suffix}"), key=attrgetter("stem")
    )
    phenotypes = [sumstats_path.name.split(".")[0] for sumstats_path in sumstats_paths]
    last_modified = [
        get_last_modified(sumstats_path) for sumstats_path in sumstats_paths
    ]

    cache_key = f"data-{hex_digest([phenotypes, last_modified])}"
    data = load_fields(sw, arguments, cache_key)
    if data is None:
        data = load(sw, ld_path, sumstats_paths, arguments.num_threads)
        save_fields(arguments, cache_key, data)

    logger.debug(f"Loaded {len(sumstats_paths)} sumstats files")
    hdl = HDL(phenotypes, data, arguments.output_path, arguments.num_threads)

    hdl.calc_piecewise()
    hdl.calc_jackknife()
