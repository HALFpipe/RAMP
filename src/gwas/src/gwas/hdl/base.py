from typing import NamedTuple, TypeVar

from jaxtyping import Array, Float64, Integer

piece_count = TypeVar("piece_count")
phenotype_count = TypeVar("phenotype_count")
eig_count = TypeVar("eig_count")
snp_count = TypeVar("snp_count")


class Sample1(NamedTuple):
    marginal_effects: Float64[Array, " ... snp_count"]
    rotated_effects: Float64[Array, " ... eig_count"]
    median_sample_count: Float64[Array, " ..."]
    min_sample_count: Float64[Array, " ..."]

    @property
    def phenotype_count(self) -> int:
        return self.median_sample_count.size

    @property
    def piece_count(self) -> int:
        return self.marginal_effects.shape[-2]


class Sample2(NamedTuple):
    sample1: Sample1
    sample2: Sample1
    params1: Float64[Array, " ... 2"]
    params2: Float64[Array, " ... 2"]
    correlation: Float64[Array, " ..."]


class Sample(NamedTuple):
    marginal_effects: Float64[Array, " phenotype_count snp_count"]
    rotated_effects: Float64[Array, " phenotype_count eig_count"]
    correlation_matrix: Float64[Array, " phenotype_count phenotype_count"]
    min_sample_count: Integer[Array, " phenotype_count"]
    median_sample_count: Float64[Array, " phenotype_count"]

    @property
    def phenotype_count(self) -> int:
        return self.marginal_effects.shape[-3]

    def select1(self, indices: Integer[Array, " phenotype_count"]) -> Sample1:
        return Sample1(
            marginal_effects=self.marginal_effects[indices],
            rotated_effects=self.rotated_effects[indices],
            median_sample_count=self.median_sample_count[indices],
            min_sample_count=self.min_sample_count[indices],
        )


class Reference(NamedTuple):
    snp_count: Integer[Array, ""]
    eig_count: Integer[Array, ""]
    ld_scores: Float64[Array, " snp_count"]
    eigenvalues: Float64[Array, " eig_count"]
