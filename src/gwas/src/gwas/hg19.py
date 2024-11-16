from itertools import pairwise

import numpy as np
from numpy import typing as npt

# Obtained via the following R script:
# gr.genome = regioneR::getGenomeAndMask(genome="hg19", mask=NA)$genome
# valid.chrs <- paste0("chr", c(seq(1, 22), "X"))
# gr.genome <- GenomeInfoDb::keepSeqlevels(gr.genome, valid.chrs, pruning.mode="coarse")
# end(gr.genome)

chromosome_lengths: npt.NDArray[np.int64] = np.asarray(
    [
        249250621,
        243199373,
        198022430,
        191154276,
        180915260,
        171115067,
        159138663,
        146364022,
        141213431,
        135534747,
        135006516,
        133851895,
        115169878,
        107349540,
        102531392,
        90354753,
        81195210,
        78077248,
        59128983,
        63025520,
        48129895,
        51304566,
        155270560,
    ],
    dtype=np.int64,
)
offset: npt.NDArray[np.int64] = np.concatenate(
    [[0], np.cumsum(chromosome_lengths)], dtype=np.int64
)

x_ticks: npt.NDArray[np.float64] = (offset[:-1] + offset[1:]) / 2


def make_segments(log_p_value: float) -> list[list[tuple[float, float]]]:
    o = offset[-1] / 500
    return [
        [(start + o, log_p_value), (end - o, log_p_value)]
        for start, end in pairwise(offset)
    ]


suggestive_log_p_value = -np.log10(1e-5)
suggestive_segments = make_segments(suggestive_log_p_value)

genome_wide_log_p_value = -np.log10(1e-8)
genome_wide_segments = make_segments(genome_wide_log_p_value)
