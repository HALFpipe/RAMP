from dataclasses import dataclass
from functools import partial
from itertools import batched, product
from typing import Callable, Iterator, Literal, Sequence, TypeVar

import jax
import jax.experimental
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from jax import numpy as jnp
from jaxtyping import Array, Float64, Integer
from numpy import typing as npt
from tqdm.auto import tqdm
from upath import UPath

from ..log import logger
from ..utils.multiprocessing import make_pool_or_null_context
from .base import Reference, Sample1, Sample2
from .load import Data
from .ml import eig_count as eig_count
from .ml import estimate1, estimate2

fields: list[pa.Field] = [
    pa.field("phenotype_index1", pa.uint32(), nullable=False),
    pa.field("phenotype_index2", pa.uint32(), nullable=False),
    pa.field("piecewise_index", pa.uint8(), nullable=True),
    pa.field("jackknife_index", pa.uint8(), nullable=True),
    pa.field("slope", pa.float64(), nullable=False),
    pa.field("intercept", pa.float64(), nullable=False),
]
schema = pa.schema(fields)


def get_indices1(
    phenotype_count: int, dataset: ds.Dataset, type: Literal["piecewise", "jackknife"]
) -> list[int]:
    existing = (
        pl.scan_pyarrow_dataset(dataset)
        .filter(
            pl.col("phenotype_index1").eq(pl.col("phenotype_index2")),
            pl.col(f"{type}_index").is_not_null(),
        )
        .select(pl.col("phenotype_index1"))
        .collect()
        .to_numpy()
        .ravel()
    )
    index = np.setdiff1d(np.arange(phenotype_count), existing)

    return list(index)


def get_indices2(
    phenotype_count: int, dataset: ds.Dataset, type: Literal["piecewise", "jackknife"]
) -> tuple[list[int], list[int]]:
    existing = (
        pl.scan_pyarrow_dataset(dataset)
        .filter(
            pl.col("phenotype_index1").ne(pl.col("phenotype_index2")),
            pl.col(f"{type}_index").is_not_null(),
        )
        .select(pl.col("phenotype_index1"), pl.col("phenotype_index2"))
        .collect()
    )
    existing1 = existing["phenotype_index1"].to_numpy()
    existing2 = existing["phenotype_index2"].to_numpy()

    existing1d = np.ravel_multi_index(
        (existing1, existing2), (phenotype_count, phenotype_count)
    )

    index1, index2 = np.triu_indices(phenotype_count, k=1)
    index1d = np.ravel_multi_index((index1, index2), (phenotype_count, phenotype_count))
    index1d = np.setdiff1d(index1d, existing1d)
    index1, index2 = np.unravel_index(index1d, (phenotype_count, phenotype_count))

    return list(index1), list(index2)


@dataclass
class Job1:
    piece_indices: npt.NDArray[np.uint8]
    phenotype_index: int
    initial_params: npt.NDArray[np.float64] | None = None


@dataclass
class Job2:
    piece_indices: npt.NDArray[np.uint8]
    phenotype_index1: int
    phenotype_index2: int
    params1: npt.NDArray[np.float64]
    params2: npt.NDArray[np.float64]
    initial_params: npt.NDArray[np.float64] | None = None


def get_indices(
    data: Data, job: Job1 | Job2
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    indices = np.cumsum(data.snp_count_array.to_numpy())[:-1]
    pieces = np.split(np.arange(data.snp_count), indices)
    pieces = [p for i, p in enumerate(pieces) if i in job.piece_indices]
    snp_indices = np.concatenate(pieces)

    indices = np.cumsum(data.eig_count_array.to_numpy())[:-1]
    pieces = np.split(np.arange(data.eig_count), indices)
    pieces = [p for i, p in enumerate(pieces) if i in job.piece_indices]
    eig_indices = np.concatenate(pieces)
    return snp_indices, eig_indices


def get_reference(
    data: Data,
    job: Job1 | Job2,
    snp_indices: npt.NDArray[np.uint32],
    eig_indices: npt.NDArray[np.uint32],
) -> Reference:
    return Reference(
        snp_count=jnp.array(data.snp_count_array[job.piece_indices].sum()),
        eig_count=jnp.array(data.eig_count_array[job.piece_indices].sum()),
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


@dataclass
class Result:
    piece_indices: npt.NDArray[np.uint8]
    params: npt.NDArray[np.float64]

    def get_indices(self, piece_count: int) -> tuple[int, bool, int, bool]:
        is_piecewise = self.piece_indices.size == 1
        piecewise_index = self.piece_indices.item() if is_piecewise else 0
        is_jackknife = self.piece_indices.size == piece_count - 1
        jackknife_index = int(
            np.setdiff1d(np.arange(piece_count), self.piece_indices).item()
            if is_jackknife
            else 0
        )
        return (
            piecewise_index,
            not is_piecewise,
            jackknife_index,
            not is_jackknife,
        )


@dataclass
class Result1(Result):
    phenotype_index: int

    @property
    def phenotype_index1(self) -> int:
        return self.phenotype_index

    @property
    def phenotype_index2(self) -> int:
        return self.phenotype_index


def get1(
    data: Data,
    job: Job1,
    n_ref: Integer[Array, ""] | None = None,
    limit: Float64[Array, ""] | None = None,
) -> Result1:
    key = jax.random.key(0)
    snp_indices, eig_indices = get_indices(data, job)
    r = get_reference(data, job, snp_indices, eig_indices)
    s = get_sample1(data, job.phenotype_index, snp_indices, eig_indices)
    _, params = estimate1(
        s, r, key, initial_params=job.initial_params, n_ref=n_ref, limit=limit
    )
    return Result1(
        job.piece_indices,
        np.array(params),
        job.phenotype_index,
    )


@dataclass
class Result2(Result):
    phenotype_index1: int
    phenotype_index2: int


def get2(
    data: Data,
    job: Job2,
    n_ref: Integer[Array, ""] | None = None,
    limit: Float64[Array, ""] | None = None,
) -> Result2:
    key = jax.random.key(0)
    snp_indices, eig_indices = get_indices(data, job)
    r = get_reference(data, job, snp_indices, eig_indices)
    correlation = data.correlation_array[job.phenotype_index1, job.phenotype_index2]
    s = Sample2(
        sample1=get_sample1(data, job.phenotype_index1, snp_indices, eig_indices),
        sample2=get_sample1(data, job.phenotype_index2, snp_indices, eig_indices),
        params1=jnp.array(job.params1),
        params2=jnp.array(job.params2),
        correlation=jnp.array(correlation),
    )
    _, params = estimate2(
        s, r, key, initial_params=job.initial_params, n_ref=n_ref, limit=limit
    )
    return Result2(
        job.piece_indices,
        np.array(params),
        job.phenotype_index1,
        job.phenotype_index2,
    )


def get_estimates(
    dataset: ds.Dataset, index: list[int], type: Literal["piecewise", "jackknife"]
) -> pl.DataFrame:
    other_type = dict(piecewise="jackknife", jackknife="piecewise")[type]
    return (
        pl.scan_pyarrow_dataset(dataset)
        .filter(
            pl.col("phenotype_index1").eq(pl.col("phenotype_index2")),
            pl.col("phenotype_index1").is_in(np.array(index)),
            pl.col(f"{other_type}_index").is_null(),
        )
        .select(
            pl.col("phenotype_index1").alias("phenotype_index"),
            pl.col(f"{type}_index"),
            pl.col("slope"),
            pl.col("intercept"),
        )
        .collect()
    )


J = TypeVar("J", Job1, Job2)


@dataclass
class HDL:
    data: Data
    path: UPath

    num_threads: int = 1

    def get_chunk_path(self) -> UPath:
        prefix = "chunk-"
        suffix = ".parquet"
        k = max(
            (
                int(path.stem.removeprefix(prefix))
                for path in self.path.glob(f"{prefix}*{suffix}")
            ),
            default=-1,
        )
        return self.path / f"{prefix}{k + 1:03d}{suffix}"

    def write(
        self, parquet_writer: pq.ParquetWriter, results: Sequence[Result1 | Result2]
    ) -> None:
        phenotype_index1 = np.fromiter(
            (r.phenotype_index1 for r in results), dtype=np.uint32
        )
        phenotype_index2 = np.fromiter(
            (r.phenotype_index2 for r in results), dtype=np.uint32
        )

        piecewise_index, piecewise_mask, jackknife_index, jackknife_mask = map(
            np.array,
            zip(
                *(result1.get_indices(self.data.piece_count) for result1 in results),
                strict=True,
            ),
        )

        slope = np.fromiter((r.params[0].item() for r in results), dtype=np.float64)
        intercept = np.fromiter((r.params[1].item() for r in results), dtype=np.float64)
        record_batch = pa.RecordBatch.from_arrays(
            arrays=[
                pa.array(phenotype_index1),
                pa.array(phenotype_index2),
                pa.array(piecewise_index, mask=piecewise_mask),
                pa.array(jackknife_index, mask=jackknife_mask),
                pa.array(np.array(slope).ravel()),
                pa.array(np.array(intercept).ravel()),
            ],
            schema=schema,
        )
        parquet_writer.write_batch(record_batch)

    def run(
        self, jobs: Iterator[J], size: int, callable: Callable[[J], Result1 | Result2]
    ) -> None:
        logger.debug(f"Running {size} jobs")
        pool, iterator = make_pool_or_null_context(
            jobs,
            callable,
            num_threads=self.num_threads,
            size=size,
            chunksize=None,
            is_jax=True,
        )
        with pq.ParquetWriter(self.get_chunk_path(), schema) as parquet_writer, pool:
            for results in batched(tqdm(iterator, unit="jobs", total=size), n=10_000):
                self.write(parquet_writer, results)

    def calc_piecewise(self) -> None:
        self.calc_piecewise1()
        self.calc_piecewise2()

    def calc_jackknife(self) -> None:
        self.calc_jackknife1()
        self.calc_jackknife2()

    def calc_piecewise1(self) -> None:
        phenotype_count = self.data.phenotype_count
        piece_count = self.data.piece_count

        dataset = ds.dataset(self.path, schema=schema, format="parquet")
        index = get_indices1(phenotype_count, dataset, "piecewise")
        logger.debug(f"Calculating piecewise estimates for {len(index)} phenotypes")
        jobs = (
            Job1(np.array([piece_index]), phenotype_index)
            for piece_index, phenotype_index in product(range(piece_count), index)
        )

        callable = partial(get1, self.data)
        self.run(jobs=jobs, callable=callable, size=piece_count * len(index))

    def calc_jackknife1(self) -> None:
        phenotype_count = self.data.phenotype_count
        piece_count = self.data.piece_count

        dataset = ds.dataset(self.path, schema=schema, format="parquet")
        index = get_indices1(phenotype_count, dataset, "jackknife")

        piecewise1_frame = get_estimates(dataset, index, "piecewise")
        initial_slope = (
            piecewise1_frame.group_by("phenotype_index")
            .agg(pl.col("slope").sum())
            .to_pandas()
            .set_index("phenotype_index")
            .loc[np.array(index)]
            .to_numpy()
        )
        initial_params = np.column_stack([initial_slope, np.ones_like(initial_slope)])

        jobs = (
            Job1(
                piece_indices=np.setdiff1d(np.arange(piece_count), piece_index),
                phenotype_index=phenotype_index,
                initial_params=initial_params[phenotype_index],
            )
            for piece_index, phenotype_index in product(range(-1, piece_count), index)
        )

        callable = partial(get1, self.data)
        self.run(jobs=jobs, callable=callable, size=(piece_count + 1) * len(index))

    def calc_piecewise2(self) -> None:
        phenotype_count = self.data.phenotype_count
        piece_count = self.data.piece_count

        dataset = ds.dataset(self.path, schema=schema, format="parquet")
        index1, index2 = get_indices2(phenotype_count, dataset, "piecewise")

        piecewise1_frame = (
            get_estimates(dataset, list(set(index1) | set(index2)), "piecewise")
            .to_pandas()
            .set_index(["phenotype_index", "piecewise_index"])
        )
        jobs = (
            Job2(
                np.array([piece_index]),
                phenotype_index1,
                phenotype_index2,
                piecewise1_frame.loc[(phenotype_index1, piece_index)].to_numpy(),  # type: ignore[arg-type,union-attr]
                piecewise1_frame.loc[(phenotype_index2, piece_index)].to_numpy(),  # type: ignore[arg-type,union-attr]
            )
            for piece_index, (phenotype_index1, phenotype_index2) in product(
                range(piece_count), zip(index1, index2, strict=True)
            )
        )

        callable = partial(get2, self.data)
        self.run(jobs=jobs, callable=callable, size=piece_count * len(index1))

    def calc_jackknife2(self) -> None:
        phenotype_count = self.data.phenotype_count
        piece_count = self.data.piece_count

        dataset = ds.dataset(self.path, schema=schema, format="parquet")
        index1, index2 = get_indices2(phenotype_count, dataset, "jackknife")
        jackknife1_frame = (
            get_estimates(dataset, list(set(index1) | set(index2)), "jackknife")
            .fill_null(-1)
            .to_pandas()
            .set_index(["phenotype_index", "jackknife_index"])
        )

        index_expr = pl.col("phenotype_index1") * phenotype_count + pl.col(
            "phenotype_index2"
        )
        index = np.array(index1) * phenotype_count + np.array(index2)
        piecewise2_frame = (
            pl.scan_pyarrow_dataset(dataset)
            .filter(index_expr.is_in(index), pl.col("jackknife_index").is_null())
            .select(index_expr.alias("index"), pl.col("slope"))
            .collect()
        )
        initial_slope = (
            piecewise2_frame.group_by("index")
            .agg(pl.col("slope").sum())
            .to_pandas()
            .set_index("index")
            .loc[index]
            .to_numpy()
        )
        initial_params = np.column_stack([initial_slope, np.ones_like(initial_slope)])

        jobs = (
            Job2(
                np.setdiff1d(np.arange(piece_count), piece_index),
                phenotype_index1,
                phenotype_index2,
                jackknife1_frame.loc[(phenotype_index1, piece_index)].to_numpy(),  # type: ignore[arg-type,union-attr]
                jackknife1_frame.loc[(phenotype_index2, piece_index)].to_numpy(),  # type: ignore[arg-type,union-attr]
                initial_params=initial_params[k],
            )
            for piece_index, (k, (phenotype_index1, phenotype_index2)) in product(
                range(-1, piece_count), enumerate(zip(index1, index2, strict=True))
            )
        )

        callable = partial(get2, self.data)
        self.run(jobs=jobs, callable=callable, size=(piece_count + 1) * len(index1))
