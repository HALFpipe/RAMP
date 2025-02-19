from dataclasses import dataclass
from functools import partial
from itertools import chain

import numpy as np
import polars as pl
from tqdm.auto import tqdm
from upath import UPath

from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..utils.multiprocessing import IterationOrder, make_pool_or_null_context


def load_snps(ld_path: UPath) -> tuple[list[list[int]], list[str]]:
    from rpy2.robjects import r

    (snp_counter_path,) = ld_path.glob("UKB_snp_counter*")
    (snp_list_path,) = ld_path.glob("UKB_snp_list*")

    (element,) = r["load"](str(snp_counter_path))
    snp_counts = [[int(value) for value in vector] for vector in r[element]]
    r["rm"](element)
    (element,) = r["load"](str(snp_list_path))
    snp_list = list(r[element])
    r["rm"](element)
    return snp_counts, snp_list


@dataclass(kw_only=True)
class Job:
    chromosome: int
    piece_index: int
    eigenvalue_array: SharedArray[np.float64]
    ld_score_array: SharedArray[np.float64]


@dataclass(kw_only=True)
class Piece:
    chromosome: int
    piece_index: int
    reference: pl.DataFrame
    eigenvalue_array: SharedArray[np.float64]
    eigenvector_array: SharedArray[np.float64]
    ld_score_array: SharedArray[np.float64]


def load_piece(ld_path: UPath, job: Job) -> Piece:
    from rpy2.robjects import r

    key = f"_chr{job.chromosome}.{job.piece_index + 1}"
    (bim_path,) = ld_path.glob(f"*{key}*.bim")
    reference = pl.read_csv(
        bim_path,
        has_header=False,
        separator="\t",
        columns=["column_2", "column_6"],  # One-based indexing
        new_columns=["SNP", "A2"],
    )

    (rdata_path,) = ld_path.glob(f"*{key}*.rda")
    elements = r["load"](str(rdata_path))
    if set(elements) != {"lam", "V", "LDsc"}:
        raise ValueError(f"Unexpected elements: {elements}")

    sw = job.eigenvalue_array.sw

    _eigenvalues = np.asarray(r["lam"])
    job.eigenvalue_array.resize(*_eigenvalues.shape)
    job.eigenvalue_array[:] = _eigenvalues

    _eigenvectors = np.asarray(r["V"])
    eigenvector_array = SharedArray.from_numpy(
        _eigenvectors, sw, name=f"eigenvectors-{job.chromosome}-{job.piece_index}"
    )

    _ld_scores = np.asarray(r["LDsc"]).copy()
    job.ld_score_array.resize(*_ld_scores.shape)
    job.ld_score_array[:] = _ld_scores

    r["rm"](*elements)

    return Piece(
        chromosome=job.chromosome,
        piece_index=job.piece_index,
        reference=reference,
        eigenvalue_array=job.eigenvalue_array,
        eigenvector_array=eigenvector_array,
        ld_score_array=job.ld_score_array,
    )


def load_pieces(sw: SharedWorkspace, ld_path: UPath, num_threads: int) -> list[Piece]:
    snp_counts, _ = load_snps(ld_path)
    # snp_frame = pl.DataFrame(dict(SNP=snp_list))

    piece_indices = [
        (chromosome, piece_index)
        for chromosome, counts in enumerate(snp_counts, start=1)
        for piece_index in range(len(counts))
    ]
    eigenvalue_arrays = [
        sw.alloc(f"eigenvalues-{chromosome}-{piece_index}", count)
        for (chromosome, piece_index), count in zip(
            piece_indices, chain.from_iterable(snp_counts), strict=True
        )
    ]
    ld_score_arrays = [
        sw.alloc(f"ld-scores-{chromosome}-{piece_index}", count)
        for (chromosome, piece_index), count in zip(
            piece_indices, chain.from_iterable(snp_counts), strict=True
        )
    ]
    jobs = [
        Job(
            chromosome=chromosome,
            piece_index=piece_index,
            eigenvalue_array=eigenvalue_array,
            ld_score_array=ld_score_array,
        )
        for (chromosome, piece_index), eigenvalue_array, ld_score_array in zip(
            piece_indices, eigenvalue_arrays, ld_score_arrays, strict=True
        )
    ]
    pool, piece_iterator = make_pool_or_null_context(
        jobs,
        partial(load_piece, ld_path),
        num_threads=num_threads,
        iteration_order=IterationOrder.ORDERED,
    )
    with pool:
        pieces: list[Piece] = list(tqdm(piece_iterator, total=len(jobs), unit="pieces"))

    sw.squash()
    return pieces
