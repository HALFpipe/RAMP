from dataclasses import dataclass
from functools import partial
from itertools import batched, combinations
from typing import Callable, Iterator, Literal, NamedTuple, Sequence, TypeVar
from uuid import uuid4

import jax
import jax.experimental
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from jax import numpy as jnp
from jaxtyping import Array, Float64, Integer
from numpy import typing as npt
from tqdm.auto import tqdm
from upath import UPath

from ..log import logger
from ..mem.arr import SharedArray
from ..utils.hash import hex_digest
from ..utils.multiprocessing import make_pool_or_null_context
from .base import Reference, Sample1, Sample2
from .load import Data
from .ml import eig_count as eig_count
from .ml import estimate1, estimate2

fields: list[pa.Field] = [
    pa.field("phenotype1", pa.string(), nullable=False),
    pa.field("phenotype2", pa.string(), nullable=False),
    pa.field("piecewise_index", pa.uint8(), nullable=True),
    pa.field("jackknife_index", pa.uint8(), nullable=True),
    pa.field("slope", pa.float64(), nullable=False),
    pa.field("intercept", pa.float64(), nullable=False),
]
schema = pa.schema(fields)


def get_phenotypes1(
    phenotypes: list[str],
    paths: list[UPath],
    piece_index: int | None,
    type: Literal["piecewise", "jackknife"],
) -> set[str]:
    logger.debug(
        f"Getting phenotypes for {type} variance estimates at piece index {piece_index}"
    )

    if piece_index is not None:
        piece_filter = pl.col(f"{type}_index").eq(piece_index)
    else:
        piece_filter = pl.col(f"{type}_index").is_null()

    existing_phenotypes: set[str] = set()
    for path in paths:
        if not path.stem.endswith(f"{type}1"):
            continue
        existing_phenotypes.update(
            pl.scan_parquet(path)
            .filter(
                pl.col("phenotype1").eq(pl.col("phenotype2")),
                piece_filter,
            )
            .select(pl.col("phenotype1"))
            .collect()["phenotype1"]
            .to_list()
        )
    return set(phenotypes) - existing_phenotypes


def get_phenotypes2(
    phenotypes: list[str],
    paths: list[UPath],
    piece_index: int | None,
    type: Literal["piecewise", "jackknife"],
) -> set[tuple[str, str]]:
    logger.debug(
        f"Getting phenotypes for {type} covariance estimates "
        f"at piece index {piece_index}"
    )

    if piece_index is not None:
        piece_filter = pl.col(f"{type}_index").eq(piece_index)
    else:
        piece_filter = pl.col(f"{type}_index").is_null()

    existing: set[tuple[str, str]] = set()
    for path in paths:
        if not path.stem.endswith(f"{type}2"):
            continue
        existing_frame = (
            pl.scan_parquet(path)
            .filter(
                pl.col("phenotype1").ne(pl.col("phenotype2")),
                pl.col("phenotype1").is_in(phenotypes),
                pl.col("phenotype2").is_in(phenotypes),
                piece_filter,
            )
            .select(pl.col("phenotype1"), pl.col("phenotype2"))
            .collect()
        )
        existing1 = existing_frame["phenotype1"].to_list()
        existing2 = existing_frame["phenotype2"].to_list()
        existing.update(zip(existing1, existing2, strict=True))

    return set(combinations(phenotypes, 2)) - existing


@dataclass(frozen=True, slots=True)
class Job1:
    type: Literal["piecewise", "jackknife"]
    piece_index: int
    phenotype_index: int
    initial_params_index: int | None = None


@dataclass(frozen=True, slots=True)
class Job2:
    type: Literal["piecewise", "jackknife"]
    piece_index: int
    phenotype_index1: int
    phenotype_index2: int
    initial_params_index: int | None = None


def get_indices(
    data: Data, piece_indices: npt.NDArray[np.uint8]
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    indices = np.cumsum(data.snp_count_array.to_numpy())[:-1]
    pieces = np.split(np.arange(data.snp_count), indices)
    pieces = [p for i, p in enumerate(pieces) if i in piece_indices]
    snp_indices = np.concatenate(pieces)

    indices = np.cumsum(data.eig_count_array.to_numpy())[:-1]
    pieces = np.split(np.arange(data.eig_count), indices)
    pieces = [p for i, p in enumerate(pieces) if i in piece_indices]
    eig_indices = np.concatenate(pieces)
    return snp_indices, eig_indices


def get_reference(
    data: Data,
    piece_indices: npt.NDArray[np.uint8],
    snp_indices: npt.NDArray[np.uint32],
    eig_indices: npt.NDArray[np.uint32],
) -> Reference:
    return Reference(
        snp_count=jnp.array(data.snp_count_array[piece_indices].sum()),
        eig_count=jnp.array(data.eig_count_array[piece_indices].sum()),
        ld_scores=jnp.array(data.ld_score_array[snp_indices]),
        eigenvalues=jnp.array(data.eigenvalue_array[eig_indices]),
    )


def get_sample1(
    data: Data,
    phenotype_index: int,
    snp_indices: npt.NDArray[np.uint32],
    eig_indices: npt.NDArray[np.uint32],
):
    return Sample1(
        marginal_effects=jnp.array(
            data.marginal_effect_array[phenotype_index, snp_indices]
        ),
        rotated_effects=jnp.array(
            data.rotated_effect_array[phenotype_index, eig_indices]
        ),
        median_sample_count=jnp.array(data.median_sample_count_array[phenotype_index]),
        min_sample_count=jnp.array(data.min_sample_count_array[phenotype_index]),
    )


class Result(NamedTuple):
    phenotype_index1: np.uint32
    phenotype_index2: np.uint32
    piecewise_index: np.uint8
    piecewise_mask: np.bool_
    jackknife_index: np.uint8
    jackknife_mask: np.bool_
    slope: np.float64
    intercept: np.float64


def get1(
    data: Data,
    initial_params_array: SharedArray[np.float64] | None,
    job: Job1,
    n_ref: Integer[Array, ""] | None = None,
    limit: Float64[Array, ""] | None = None,
) -> Result:
    match job.type:
        case "piecewise":
            piecewise_index: np.uint8 = np.uint8(job.piece_index)
            piecewise_mask: np.bool_ = np.bool_(False)
            jackknife_index: np.uint8 = np.uint8(0)
            jackknife_mask: np.bool_ = np.bool_(True)
            piece_indices = np.array([job.piece_index])
        case "jackknife":
            piecewise_index = np.uint8(0)
            piecewise_mask = np.bool_(True)
            jackknife_index = np.uint8(max(job.piece_index, 0))
            jackknife_mask = np.bool_(job.piece_index < 0)
            piece_indices = np.setdiff1d(np.arange(data.piece_count), [job.piece_index])
        case _:
            raise ValueError(f"Invalid type: {job.type}")

    initial_params: Float64[Array, " 2"] | None = None
    if job.initial_params_index is not None and initial_params_array is not None:
        initial_params = jnp.array(initial_params_array[job.initial_params_index])

    key = jax.random.key(0)
    snp_indices, eig_indices = get_indices(data, piece_indices)
    r = get_reference(data, piece_indices, snp_indices, eig_indices)
    s = get_sample1(data, job.phenotype_index, snp_indices, eig_indices)
    _, params = estimate1(
        s, r, key, initial_params=initial_params, n_ref=n_ref, limit=limit
    )
    slope, intercept = params

    return Result(
        np.uint32(job.phenotype_index),
        np.uint32(job.phenotype_index),
        piecewise_index,
        piecewise_mask,
        jackknife_index,
        jackknife_mask,
        np.float64(slope.item()),
        np.float64(intercept.item()),
    )


def get2(
    data: Data,
    params_array: SharedArray[np.float64],
    initial_params_array: SharedArray[np.float64] | None,
    job: Job2,
    n_ref: Integer[Array, ""] | None = None,
    limit: Float64[Array, ""] | None = None,
) -> Result:
    match job.type:
        case "piecewise":
            k = job.piece_index
            piecewise_index: np.uint8 = np.uint8(job.piece_index)
            piecewise_mask: np.bool_ = np.bool_(False)
            jackknife_index: np.uint8 = np.uint8(0)
            jackknife_mask: np.bool_ = np.bool_(True)
            piece_indices = np.array([job.piece_index])
        case "jackknife":
            k = job.piece_index + 1
            piecewise_index = np.uint8(0)
            piecewise_mask = np.bool_(True)
            jackknife_index = np.uint8(max(job.piece_index, 0))
            jackknife_mask = np.bool_(job.piece_index < 0)
            piece_indices = np.setdiff1d(np.arange(data.piece_count), [job.piece_index])
        case _:
            raise ValueError(f"Invalid type: {job.type}")

    initial_params: Float64[Array, " 2"] | None = None
    if job.initial_params_index is not None and initial_params_array is not None:
        initial_params = jnp.array(initial_params_array[job.initial_params_index])

    params1 = params_array[job.phenotype_index1, k]
    params2 = params_array[job.phenotype_index2, k]

    key = jax.random.key(0)
    snp_indices, eig_indices = get_indices(data, piece_indices)
    r = get_reference(data, piece_indices, snp_indices, eig_indices)
    correlation = data.correlation_array[job.phenotype_index1, job.phenotype_index2]
    s = Sample2(
        sample1=get_sample1(data, job.phenotype_index1, snp_indices, eig_indices),
        sample2=get_sample1(data, job.phenotype_index2, snp_indices, eig_indices),
        params1=jnp.array(params1),
        params2=jnp.array(params2),
        correlation=jnp.array(correlation),
    )
    _, params = estimate2(
        s, r, key, initial_params=initial_params, n_ref=n_ref, limit=limit
    )
    slope, intercept = params

    return Result(
        np.uint32(job.phenotype_index1),
        np.uint32(job.phenotype_index2),
        piecewise_index,
        piecewise_mask,
        jackknife_index,
        jackknife_mask,
        np.float64(slope.item()),
        np.float64(intercept.item()),
    )


def get_estimates(
    paths: list[UPath], phenotypes: list[str], type: Literal["piecewise", "jackknife"]
) -> pl.DataFrame:
    other_type = dict(piecewise="jackknife", jackknife="piecewise")[type]
    data_frame = pl.concat(
        [
            pl.scan_parquet(path)
            .filter(
                pl.col("phenotype1").eq(pl.col("phenotype2")),
                pl.col("phenotype1").is_in(phenotypes),
                pl.col(f"{other_type}_index").is_null(),
            )
            .select(
                pl.col("phenotype1").alias("phenotype"),
                pl.col(f"{type}_index"),
                pl.col("slope"),
                pl.col("intercept"),
            )
            for path in paths
            if path.stem.endswith(f"{type}1")
        ]
    ).collect()
    return data_frame


J = TypeVar("J", Job1, Job2)


@dataclass
class HDL:
    phenotypes: list[str]
    data: Data
    path: UPath

    num_threads: int = 1

    def get_paths(self) -> list[UPath]:
        return list(
            filter(
                lambda path: path.stat().st_size > 8,
                self.path.glob("chunk-*.parquet"),
            )
        )

    def get_chunk_path(self, k: str) -> UPath:
        prefix = "chunk-"
        suffix = ".parquet"
        digest = hex_digest([self.phenotypes, uuid4().hex])
        return self.path / f"{prefix}{digest}-{k}{suffix}"

    def write(
        self,
        parquet_writer: pq.ParquetWriter,
        results: Sequence[Result],
    ) -> None:
        (
            phenotype_index1,
            phenotype_index2,
            piecewise_index,
            piecewise_mask,
            jackknife_index,
            jackknife_mask,
            slope,
            intercept,
        ) = map(
            np.array,
            zip(*results, strict=True),
        )

        phenotype1 = [self.phenotypes[i] for i in phenotype_index1]
        phenotype2 = [self.phenotypes[i] for i in phenotype_index2]

        record_batch = pa.RecordBatch.from_arrays(
            arrays=[
                pa.array(phenotype1),
                pa.array(phenotype2),
                pa.array(piecewise_index, mask=piecewise_mask),
                pa.array(jackknife_index, mask=jackknife_mask),
                pa.array(np.array(slope).ravel()),
                pa.array(np.array(intercept).ravel()),
            ],
            schema=schema,
        )
        parquet_writer.write_batch(record_batch)

    def run(
        self,
        jobs: Iterator[J],
        k: str,
        size: int,
        callable: Callable[[J], Result],
    ) -> None:
        logger.debug(f"Running {size} jobs")
        chunksize = 1 << 7
        pool, iterator = make_pool_or_null_context(
            jobs,
            callable,
            num_threads=self.num_threads,
            size=size,
            chunksize=chunksize if size > chunksize else None,
            is_jax=True,
        )
        with pool:
            for results in batched(
                tqdm(iterator, unit="jobs", total=size),
                n=10_000_000,
            ):
                with pq.ParquetWriter(
                    self.get_chunk_path(k),
                    schema,
                    compression="zstd",
                    compression_level=9,
                ) as parquet_writer:
                    for r in batched(results, n=1_000_000):
                        self.write(parquet_writer, r)

    def calc_piecewise(self) -> None:
        self.calc_piecewise1()
        self.calc_piecewise2()

    def calc_jackknife(self) -> None:
        self.calc_jackknife1()
        self.calc_jackknife2()

    def calc_piecewise1(self) -> None:
        piece_count = self.data.piece_count
        type: Literal["piecewise", "jackknife"] = "piecewise"

        def generate_jobs() -> Iterator[Job1]:
            for piece_index in range(piece_count):
                phenotypes = get_phenotypes1(
                    self.phenotypes, self.get_paths(), piece_index, type
                )
                for phenotype in phenotypes:
                    yield Job1(type, piece_index, self.phenotypes.index(phenotype))

        logger.debug(
            f"Calculating {type} estimates for {len(self.phenotypes)} phenotypes"
        )

        callable = partial(get1, self.data, None)
        jobs = generate_jobs()
        self.run(
            k=f"{type}1",
            jobs=jobs,
            callable=callable,
            size=piece_count * len(self.phenotypes),
        )

    def calc_jackknife1(self) -> None:
        sw = self.data.marginal_effect_array.sw
        piece_count = self.data.piece_count
        type: Literal["piecewise", "jackknife"] = "jackknife"

        paths = self.get_paths()
        piecewise1_frame = get_estimates(paths, self.phenotypes, "piecewise")
        initial_slope = (
            piecewise1_frame.group_by("phenotype")
            .agg(pl.col("slope").sum())
            .to_pandas()
            .set_index("phenotype")
            .loc[self.phenotypes]
            .to_numpy()
        )
        initial_params = np.column_stack([initial_slope, np.ones_like(initial_slope)])
        initial_params_array = SharedArray.from_numpy(initial_params, sw)

        def generate_jobs() -> Iterator[Job1]:
            for piece_index in range(-1, piece_count):
                phenotypes = get_phenotypes1(self.phenotypes, paths, piece_index, type)
                for phenotype in phenotypes:
                    phenotype_index = self.phenotypes.index(phenotype)
                    yield Job1(
                        type,
                        piece_index,
                        phenotype_index,
                        phenotype_index,
                    )

        callable = partial(get1, self.data, initial_params_array)
        jobs = generate_jobs()
        self.run(
            k=f"{type}1",
            jobs=jobs,
            callable=callable,
            size=(piece_count + 1) * len(self.phenotypes),
        )

        initial_params_array.free()

    def get_params_array(
        self,
        paths: list[UPath],
        phenotypes: list[str],
        type: Literal["piecewise", "jackknife"],
    ) -> tuple[dict[str, int], SharedArray[np.float64]]:
        piece_count = self.data.piece_count
        phenotype_count = len(phenotypes)
        indices_by_phenotype: dict[str, int] = {
            phenotype: i for i, phenotype in enumerate(self.phenotypes)
        }

        params_frame = get_estimates(paths, phenotypes, type).fill_null(-1).to_pandas()
        params_frame["phenotype_index"] = params_frame["phenotype"].map(
            indices_by_phenotype
        )
        if type == "jackknife":
            params_frame[f"{type}_index"] += 1
            piece_count += 1
        params_frame = params_frame.reset_index().set_index(
            ["phenotype_index", f"{type}_index"]
        )[["slope", "intercept"]]

        sw = self.data.marginal_effect_array.sw
        params_array = sw.alloc("params", phenotype_count, piece_count, 2)
        params_array[:] = np.nan  # type: ignore[assignment]
        params_array[*zip(*params_frame.index, strict=True)] = params_frame.to_numpy()

        if not np.all(np.isfinite(params_array.to_numpy())):
            raise ValueError("Invalid params")

        return indices_by_phenotype, params_array

    def calc_piecewise2(self) -> None:
        piece_count = self.data.piece_count
        phenotype_count = len(self.phenotypes)
        pairs_count = phenotype_count * (phenotype_count - 1) // 2
        type: Literal["piecewise", "jackknife"] = "piecewise"

        paths = self.get_paths()

        indices_by_phenotype, params_array = self.get_params_array(
            paths, self.phenotypes, "piecewise"
        )

        def generate_jobs() -> Iterator[Job2]:
            for piece_index in range(piece_count):
                phenotypes = get_phenotypes2(self.phenotypes, paths, piece_index, type)
                for phenotype1, phenotype2 in phenotypes:
                    yield Job2(
                        type,
                        piece_index,
                        indices_by_phenotype[phenotype1],
                        indices_by_phenotype[phenotype2],
                    )

        callable = partial(get2, self.data, params_array, None)
        jobs = generate_jobs()
        self.run(
            k=f"{type}2", jobs=jobs, callable=callable, size=piece_count * pairs_count
        )

        params_array.free()

    def calc_jackknife2(self) -> None:
        sw = self.data.marginal_effect_array.sw
        piece_count = self.data.piece_count
        phenotype_count = len(self.phenotypes)
        pairs_count = phenotype_count * (phenotype_count - 1) // 2
        type: Literal["piecewise", "jackknife"] = "jackknife"

        paths = self.get_paths()

        indices_by_phenotype, params_array = self.get_params_array(
            paths, self.phenotypes, "jackknife"
        )

        pair_expr = pl.concat_str(
            pl.col("phenotype1"), pl.col("phenotype2"), separator=":"
        ).alias("pair")
        piecewise2_frame = (
            pl.concat(
                [
                    pl.scan_parquet(path)
                    .filter(
                        pl.col("piecewise_index").is_not_null(),
                        pl.col("jackknife_index").is_null(),
                    )
                    .select(pair_expr, pl.col("slope"))
                    for path in paths
                    if path.stem.endswith("piecewise2")
                ]
            )
            .group_by("pair")
            .agg(pl.col("slope").sum())
            .collect()
            .to_pandas()
        )

        initial_slope = piecewise2_frame["slope"].to_numpy()
        initial_params = np.column_stack([initial_slope, np.ones_like(initial_slope)])
        initial_params_array = SharedArray.from_numpy(initial_params, sw)

        initial_params_indices: dict[str, int] = {
            pair: i for i, pair in enumerate(piecewise2_frame["pair"])
        }

        def generate_jobs() -> Iterator[Job2]:
            for piece_index in range(-1, piece_count):
                phenotypes = get_phenotypes2(self.phenotypes, paths, piece_count, type)
                for phenotype1, phenotype2 in phenotypes:
                    pair = f"{phenotype1}:{phenotype2}"
                    initial_params_index = initial_params_indices[pair]
                    yield Job2(
                        type,
                        piece_index,
                        indices_by_phenotype[phenotype1],
                        indices_by_phenotype[phenotype2],
                        initial_params_index,
                    )

        callable = partial(get2, self.data, params_array, initial_params_array)
        jobs = generate_jobs()
        self.run(
            k=f"{type}2",
            jobs=jobs,
            callable=callable,
            size=(piece_count + 1) * pairs_count,
        )

        params_array.free()
        initial_params_array.free()
