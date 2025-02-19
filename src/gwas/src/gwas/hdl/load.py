from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
from more_itertools import consume
from tqdm.auto import tqdm
from upath import UPath

from ..covar import calc_covariance
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..utils.multiprocessing import IterationOrder, make_pool_or_null_context
from .data import load_pieces


@dataclass(kw_only=True)
class Data:
    snp_count_array: SharedArray[np.uint32]
    eig_count_array: SharedArray[np.uint32]
    marginal_effect_array: SharedArray[np.float64]
    ld_score_array: SharedArray[np.float64]
    rotated_effect_array: SharedArray[np.float64]
    eigenvalue_array: SharedArray[np.float64]

    correlation_array: SharedArray[np.float64]
    min_sample_count_array: SharedArray[np.float64]
    median_sample_count_array: SharedArray[np.float64]

    @property
    def phenotype_count(self) -> int:
        return self.marginal_effect_array.shape[0]

    @property
    def piece_count(self) -> int:
        return self.snp_count_array.size

    @property
    def snp_count(self) -> int:
        return self.ld_score_array.size

    @property
    def eig_count(self) -> int:
        return self.eigenvalue_array.size


def load_sumstats(
    reference_frame: pl.DataFrame,
    marginal_effect_array: SharedArray[np.float64],
    z_array: SharedArray[np.float64],
    min_sample_count_array: SharedArray[np.float64],
    median_sample_count_array: SharedArray[np.float64],
    job: tuple[int, UPath],
) -> int:
    phenotype_index, sumstats_path = job

    min_sample_counts = min_sample_count_array.to_numpy()
    median_sample_counts = median_sample_count_array.to_numpy()

    n_expr = pl.col("N").alias("n")
    not_swapped = pl.col("reference_alternate_allele") == pl.col("alternate_allele")
    sign = 2 * not_swapped.cast(pl.Float64) - 1
    z_expr = (pl.col("Z") * sign).alias("z")

    # Equations 2 and 9
    marginal_effect_expr = (z_expr / n_expr.sqrt()).fill_null(0).alias("marginal_effect")

    sumstats_scan = (
        pl.scan_csv(
            sumstats_path,
            separator="\t",
            schema_overrides=dict(N=pl.Float64, A2=pl.Categorical),
        )
        .drop_nulls(subset="Z")
        .select(
            pl.col("SNP"),
            pl.col("Z"),
            pl.col("N"),
            pl.col("A2").alias("alternate_allele"),
        )
    )
    sumstats_frame = (
        reference_frame.lazy()
        .select(pl.col("SNP"), pl.col("A2").alias("reference_alternate_allele"))
        .join(sumstats_scan, how="left", maintain_order="left", on="SNP")
        .select(marginal_effect_expr, z_expr, n_expr)
        .collect()
    )

    z_array[phenotype_index] = sumstats_frame["z"].to_numpy().ravel()
    marginal_effect_array[phenotype_index] = (
        sumstats_frame["marginal_effect"].to_numpy().ravel()
    )
    min_sample_counts[phenotype_index] = sumstats_frame["n"].min()
    median_sample_counts[phenotype_index] = sumstats_frame["n"].median()

    return phenotype_index


def load(
    sw: SharedWorkspace, ld_path: UPath, sumstats_paths: list[UPath], num_threads: int
) -> Data:
    phenotype_count = len(sumstats_paths)
    pieces = load_pieces(sw, ld_path, num_threads)

    snp_counts = np.array([p.ld_score_array.size for p in pieces], dtype=np.uint32)
    eig_counts = np.array([p.eigenvalue_array.size for p in pieces], dtype=np.uint32)

    eigenvalue_array = SharedArray.merge(*(p.eigenvalue_array for p in pieces))
    ld_score_array = SharedArray.merge(*(p.ld_score_array for p in pieces))

    eig_count_array = SharedArray.from_numpy(eig_counts, sw, name="eigenvalue-counts")
    snp_count_array = SharedArray.from_numpy(snp_counts, sw, name="snp_counts")

    min_sample_count_array: SharedArray[np.float64] = sw.alloc(
        "min-sample-counts", phenotype_count, dtype=np.float64
    )
    median_sample_count_array: SharedArray[np.float64] = sw.alloc(
        "median-sample-counts", phenotype_count, dtype=np.float64
    )

    shape = (phenotype_count, snp_counts.sum().item())
    z_array: SharedArray[np.float64] = sw.alloc("z", *shape, dtype=np.float64)
    z_array.to_numpy().fill(np.nan)
    marginal_effect_array: SharedArray[np.float64] = sw.alloc(
        "marginal_effects", *shape, dtype=np.float64
    )

    shape = (phenotype_count, eig_counts.sum().item())
    rotated_effect_array: SharedArray[np.float64] = sw.alloc(
        "rotated_effects", *shape, dtype=np.float64
    )

    reference_frame = pl.concat(piece.reference for piece in pieces)
    pool, iterator = make_pool_or_null_context(
        enumerate(sumstats_paths),
        partial(
            load_sumstats,
            reference_frame,
            marginal_effect_array,
            z_array,
            min_sample_count_array,
            median_sample_count_array,
        ),
        num_threads=num_threads,
        size=len(sumstats_paths),
        chunksize=None,
        iteration_order=IterationOrder.ORDERED,
    )

    snp_indices = np.cumsum(snp_count_array.to_numpy())[:-1]
    eig_indices = np.cumsum(eig_count_array.to_numpy())[:-1]
    with pool:
        consume(tqdm(iterator, unit="files", total=len(sumstats_paths)))

    for p, marginal_effects, rotated_effects in zip(
        pieces,
        np.split(marginal_effect_array.to_numpy(), snp_indices, axis=1),
        np.split(rotated_effect_array.to_numpy(), eig_indices, axis=1),
        strict=True,
    ):
        eigenvectors = p.eigenvector_array.to_numpy()
        np.einsum("se, ps -> pe", eigenvectors, marginal_effects, out=rotated_effects)

    for p in pieces:
        p.eigenvector_array.free()

    z = z_array.to_numpy()
    covariance_array = calc_covariance(sw, z.transpose())

    covariance_matrix = covariance_array.to_numpy()
    factor = np.power(np.diag(covariance_matrix), -0.5)
    np.einsum("ij, i, j -> ij", covariance_matrix, factor, factor, out=covariance_matrix)
    correlation_array = covariance_array

    sw.squash()

    return Data(
        snp_count_array=snp_count_array,
        eig_count_array=eig_count_array,
        marginal_effect_array=marginal_effect_array,
        ld_score_array=ld_score_array,
        rotated_effect_array=rotated_effect_array,
        eigenvalue_array=eigenvalue_array,
        correlation_array=correlation_array,
        min_sample_count_array=min_sample_count_array,
        median_sample_count_array=median_sample_count_array,
    )
