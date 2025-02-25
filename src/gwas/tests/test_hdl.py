from itertools import product
from operator import attrgetter
from subprocess import run
from typing import Literal

import jax
import numpy as np
import polars as pl
import pyarrow.dataset as ds
import pytest
from jax import numpy as jnp
from numpy import typing as npt
from optax import tree_utils as otu
from rpy2.robjects import default_converter, numpy2ri, r
from tqdm.auto import tqdm
from upath import UPath

from gwas.hdl.base import Reference, Sample1, Sample2
from gwas.hdl.calc import HDL, schema
from gwas.hdl.command import suffix
from gwas.hdl.load import Data, load
from gwas.hdl.ml import (
    estimate1,
    estimate2,
    minus_two_log_likelihood1,
    minus_two_log_likelihood2,
    scale_eigenvalues1,
)
from gwas.hdl.try_hard import try_hard
from gwas.mem.wkspace import SharedWorkspace
from gwas.tools import r_genomicsem
from gwas.utils.threads import cpu_count

from .conftest import base_path
from .utils import check_bias, check_types

input_path = base_path / "hdl"
ld_path = input_path / "UKB_imputed_hapmap2_SVD_eigen99_extraction"

sumstats_paths = sorted(input_path.rglob(f"*{suffix}"), key=attrgetter("stem"))
assert len(sumstats_paths) == 3

llfun1 = """llfun <-  function(param, N, M,Nref=1000, lam, bstar, lim=exp(-10)){
    h2 <- param[1]
    int <- param[2]
    lamh2 <- h2/M*lam^2 - h2*lam/Nref + int*lam/N
    lamh2 <- ifelse(lamh2<lim, lim, lamh2)
    ll <- sum(log(lamh2)) + sum(bstar^2/(lamh2))
    return(ll)
}"""
llfun2 = """llfun.gcov.part.2 <- function(param, h11, h22, rho12, M, N1, N2, N0, Nref, lam0, lam1, lam2, bstar1, bstar2, lim=exp(-10)){
  h12 <- param[1]
  int <- param[2]
  ## sample fractions
  p1 <- N0/N1; p2 <- N0/N2
  ## must follow the formula for lamh2 used in llfun4
  lam11 <- h11[1]/M*lam1^2 - h11[1]*lam1/Nref + h11[2]*lam1/N1
  lam11 <- ifelse(lam11<lim, lim, lam11)
  lam22 <- h22[1]/M*lam2^2 - h22[1]*lam2/Nref + h22[2]*lam2/N2
  lam22 <- ifelse(lam22<lim, lim, lam22)
  #lam12 = h12/M*lam1*lam2 - p1*p2*h12*lam1/Nref + p1*p2*int*lam1/N0
  if (N0>0) lam12 <- h12/M*lam1*lam2 + p1*p2*int*lam1/N0  ## key change here
  if (N0==0) lam12 <- h12/M*lam1*lam2
  ##  resid of bstar2 ~bstar1
  ustar <- bstar2 - lam12/lam11*bstar1  ## see note
  lam22.1 <- lam22 - lam12^2/lam11
  lam22.1 <- ifelse(lam22.1<lim, lim, lam22.1)
  ll <- sum(log(lam22.1)) + sum(ustar^2/(lam22.1))
  return(list(lam11=lam11, lam22=lam22, lam12=lam12, ustar=ustar, lam22.1=lam22.1, ll=ll))
}"""  # noqa: E501


@pytest.fixture(scope="module")
def data(sw: SharedWorkspace, request: pytest.FixtureRequest) -> Data:
    data = load(sw, ld_path, sumstats_paths, cpu_count())
    request.addfinalizer(data.snp_count_array.free)
    request.addfinalizer(data.eig_count_array.free)
    request.addfinalizer(data.marginal_effect_array.free)
    request.addfinalizer(data.ld_score_array.free)
    request.addfinalizer(data.rotated_effect_array.free)
    request.addfinalizer(data.eigenvalue_array.free)
    request.addfinalizer(data.correlation_array.free)
    request.addfinalizer(data.min_sample_count_array.free)
    request.addfinalizer(data.median_sample_count_array.free)
    return data


@pytest.mark.parametrize("type", ["piecewise", "jackknife"])
def test_hdl(
    tmp_path: UPath, data: Data, type: Literal["piecewise", "jackknife"]
) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    phenotypes = [sumstats_path.name.split(".")[0] for sumstats_path in sumstats_paths]
    hdl = HDL(
        phenotypes=phenotypes, data=data, path=dataset_path, num_threads=cpu_count()
    )

    hdl.calc_piecewise()
    if type == "jackknife":
        hdl.calc_jackknife()

    dataset = ds.dataset(dataset_path, schema=schema, format="parquet")
    filters = dict(
        piecewise=[pl.col("piecewise_index").is_not_null()],
        jackknife=[
            pl.col("piecewise_index").is_null(),
            pl.col("jackknife_index").is_null(),
        ],
    )[type]
    group_by = (
        pl.scan_pyarrow_dataset(dataset)
        .filter(*filters)
        .select(
            pl.col("phenotype1").map_elements(
                lambda phenotype: phenotypes.index(phenotype), return_dtype=pl.UInt32
            ),
            pl.col("phenotype2").map_elements(
                lambda phenotype: phenotypes.index(phenotype), return_dtype=pl.UInt32
            ),
            pl.col("slope"),
            pl.col("intercept"),
        )
        .group_by(pl.col("phenotype1"), pl.col("phenotype2"))
    )

    data_frame = group_by.agg(pl.col("slope").sum()).collect()
    _slope = np.zeros((len(sumstats_paths), len(sumstats_paths)))
    _slope[data_frame["phenotype1"], data_frame["phenotype2"]] = data_frame["slope"]
    _slope[data_frame["phenotype2"], data_frame["phenotype1"]] = data_frame["slope"]

    data_frame = group_by.agg(pl.col("intercept").mean()).collect()
    _intercept = np.zeros((len(sumstats_paths), len(sumstats_paths)))
    _intercept[data_frame["phenotype1"], data_frame["phenotype2"]] = data_frame[
        "intercept"
    ]
    _intercept[data_frame["phenotype2"], data_frame["phenotype1"]] = data_frame[
        "intercept"
    ]

    assert np.isfinite(_slope).all()
    assert np.isfinite(_intercept).all()

    rdata_name = "hdl.rdata"
    run(
        args=[*r_genomicsem],
        cwd=tmp_path,
        check=True,
        text=True,
        input=f"""traits = c({", ".join(f'"{s}"' for s in sumstats_paths)})
hdl = GenomicSEM::hdl(
    traits = traits,
    sample.prev = rep(NA, length(traits)),
    population.prev = rep(NA, length(traits)),
    LD.path = "{ld_path}",
    trait.names = c({", ".join(f'"{s.name.removesuffix(suffix)}"' for s in sumstats_paths)}),
    method = "{type}",
)
save(hdl, file = "{rdata_name}")
""",  # noqa: E501
    )
    elements = r["load"](str(tmp_path / rdata_name))
    assert set(elements) == {"hdl"}
    output = r["hdl"]
    sampling_covariance, slope, intercept, _ = map(np.copy, map(np.asarray, output))
    assert np.isfinite(sampling_covariance).all()

    assert check_bias(intercept, _intercept)
    assert check_bias(slope, _slope)

    if type == "piecewise":
        indices: list[tuple[int, int]] = list()
        for phenotype_index1 in range(len(sumstats_paths)):
            for phenotype_index2 in range(phenotype_index1, len(sumstats_paths)):
                if phenotype_index1 == phenotype_index2:
                    indices.append((phenotype_index1, phenotype_index2))
                elif phenotype_index1 != phenotype_index2:
                    indices.append((phenotype_index1, phenotype_index2))
        sampling_variances = (
            group_by.agg((pl.col("slope").sum() - pl.col("slope")).var())
            .collect()
            .to_pandas()
            .set_index(["phenotype1", "phenotype2"])
            .loc[indices]
            * data.piece_count
        )
        assert check_bias(sampling_variances.to_numpy(), np.diag(sampling_covariance))


def from_triangular_columns(
    a1: npt.NDArray[np.float64], a2: npt.NDArray[np.float64], n: int
) -> npt.NDArray[np.float64]:
    square = np.zeros((n, n))
    square[np.triu_indices_from(square, k=1)] = a2
    square += square.T
    square[np.diag_indices_from(square)] = a1
    return square


def test_minimize() -> None:
    key = jax.random.key(0)

    def fun(w):
        return jnp.sum(100.0 * (w[1:] - w[:-1] ** 2) ** 2 + (1.0 - w[:-1]) ** 2)

    init_params = jnp.zeros((8,))
    print(
        f"Initial value: {fun(init_params):.2e} "
        f"Initial gradient norm: {otu.tree_l2_norm(jax.grad(fun)(init_params))}"
    )
    value, final_params = check_types(try_hard)(fun, init_params, key=key)
    print(
        f"Final value: {value:.2e}, "
        f"Final gradient norm: {otu.tree_l2_norm(jax.grad(fun)(final_params))}"
    )
    assert value < 1e-10


def test_minus_two_log_likelihood1() -> None:
    r(llfun1)
    llfun = r["llfun"]

    variant_count = 1000
    snp_count = 1000
    eig_count = 1000
    sample_count = 1000
    n_ref = 1000

    params = np.array([0.5, 0.5])
    eigenvalues = np.random.uniform(size=(variant_count,))
    rotated_effects = np.random.normal(size=(eig_count,))
    limit = np.exp(-18)
    with (default_converter + numpy2ri.converter).context():
        target = llfun(
            params, sample_count, snp_count, n_ref, eigenvalues, rotated_effects, limit
        )

    ref = Reference(
        jnp.array(snp_count),
        jnp.array(eig_count),
        jnp.zeros(snp_count),
        jnp.array(eigenvalues),
    )
    with jax.debug_nans(True):
        value = check_types(minus_two_log_likelihood1)(
            jnp.array(params),
            jnp.array(rotated_effects),
            jnp.array(sample_count),
            ref,
            n_ref=jnp.array(n_ref),
            limit=jnp.asarray(limit),
        )

    assert np.isclose(target, value)


def test_minus_two_log_likelihood2() -> None:
    r(llfun2)
    llfun = r["llfun.gcov.part.2"]

    variant_count = 1000
    snp_count = 1000
    eig_count = 1000
    sample_count1 = 1000
    sample_count2 = 1500
    sample_count0 = 900
    n_ref = 1000

    params1 = np.array([0.04, 1.05])
    params2 = np.array([0.03, 0.97])

    correlation = 0.34

    params = np.array([0.05, 0.01])

    eigenvalues = np.random.uniform(size=(variant_count,))
    rotated_effects1 = np.random.normal(size=(eig_count,))
    rotated_effects2 = np.random.normal(size=(eig_count,))

    limit = np.exp(-18)
    with (default_converter + numpy2ri.converter).context():
        output = llfun(
            params,
            params1,
            params2,
            correlation,
            snp_count,
            sample_count1,
            sample_count2,
            sample_count0,
            n_ref,
            eigenvalues,
            eigenvalues,
            eigenvalues,
            rotated_effects1,
            rotated_effects2,
            limit,
        )
        target = output["ll"]

    scaled_eigenvalues1 = scale_eigenvalues1(
        jnp.array(params1),
        jnp.array(eigenvalues),
        jnp.array(snp_count),
        jnp.array(sample_count1),
        n_ref=jnp.array(n_ref),
        limit=jnp.asarray(limit),
    )
    assert np.allclose(scaled_eigenvalues1, output["lam11"])
    scaled_eigenvalues2 = scale_eigenvalues1(
        jnp.array(params2),
        jnp.array(eigenvalues),
        jnp.array(snp_count),
        jnp.array(sample_count2),
        n_ref=jnp.array(n_ref),
        limit=jnp.asarray(limit),
    )
    assert np.allclose(scaled_eigenvalues2, output["lam22"])

    sample_proportion1 = sample_count0 / sample_count1
    sample_proportion2 = sample_count0 / sample_count2
    sample_count = sample_count0 / sample_proportion1 / sample_proportion2

    ref = Reference(
        jnp.array(snp_count),
        jnp.array(eig_count),
        jnp.zeros(snp_count),
        jnp.array(eigenvalues),
    )
    with jax.debug_nans(True):
        value = check_types(minus_two_log_likelihood2)(
            jnp.array(params),
            jnp.array(scaled_eigenvalues1),
            jnp.array(scaled_eigenvalues2),
            jnp.array(rotated_effects1),
            jnp.array(rotated_effects2),
            ref,
            jnp.array(sample_count),
            limit=jnp.asarray(limit),
        )

    assert np.isclose(target, value)


def test_estimate1(data: Data) -> None:
    key = jax.random.key(0)

    marginal_effects = jnp.array(data.marginal_effect_array.to_numpy())
    rotated_effects = jnp.array(data.rotated_effect_array.to_numpy())
    median_sample_counts = jnp.array(data.median_sample_count_array.to_numpy())
    min_sample_counts = jnp.array(data.min_sample_count_array.to_numpy())
    snp_counts = jnp.array(data.snp_count_array.to_numpy())
    eig_counts = jnp.array(data.eig_count_array.to_numpy())
    ld_scores = jnp.array(data.ld_score_array.to_numpy())
    eigenvalues = jnp.array(data.eigenvalue_array.to_numpy())

    r(llfun1)

    r("""estimate1 <- function(bhat1, bstar1, LDsc, N1, M, Nref, lam){
a11 <- bhat1^2
reg <- lm(a11 ~ LDsc)
h11.ols <- c(summary(reg)$coef[1:2, 1:2] * c(N1,  M))
h11v <- (h11.ols[2] * LDsc/M + 1/N1)^2
reg <- lm(a11 ~ LDsc, weight = 1/h11v)
h11.wls <- c(summary(reg)$coef[1:2, 1:2] * c(N1,  M))
opt <- optim(c(h11.wls[2], 1), llfun, N = N1, Nref = Nref,
                lam = lam, bstar = bstar1, M = M, lim = exp(-18),
                method = "L-BFGS-B", lower = c(0, 0), upper = c(1,10))
h11.hdl <- opt$par
logL <- opt$value
return(list(h11 = h11.hdl, logL = logL))
}""")

    piece_count = data.piece_count
    phenotype_count = data.phenotype_count

    jobs = [
        (piece_index, phenotype_index)
        for piece_index, phenotype_index in product(
            range(piece_count), range(phenotype_count)
        )
    ]

    for piece_index, phenotype_index in tqdm(jobs, unit="tests"):
        snp_slice = slice(
            snp_counts[:piece_index].sum(), snp_counts[: piece_index + 1].sum()
        )
        eig_slice = slice(
            eig_counts[:piece_index].sum(), eig_counts[: piece_index + 1].sum()
        )

        n1 = median_sample_counts[phenotype_index].item()
        m = snp_counts[piece_index].item()
        nref = 335265

        bhat = np.array(marginal_effects[phenotype_index, snp_slice])
        bstar1 = np.array(rotated_effects[phenotype_index, eig_slice])
        ldsc = np.array(ld_scores[snp_slice])
        lam = np.array(eigenvalues[eig_slice])

        with (default_converter + numpy2ri.converter).context():
            result = r["estimate1"](bhat, bstar1, ldsc, n1, m, nref, lam)
            target_params = result["h11"]
            target_minus_two_log_likelihood = result["logL"].item()

        s = Sample1(
            marginal_effects[phenotype_index, snp_slice],
            rotated_effects[phenotype_index, eig_slice],
            median_sample_counts[phenotype_index],
            min_sample_counts[phenotype_index],
        )
        ref = Reference(
            snp_counts[piece_index],
            eig_counts[piece_index],
            ld_scores[snp_slice],
            eigenvalues[eig_slice],
        )
        value, params = estimate1(s, ref, key)

        assert value <= target_minus_two_log_likelihood
        np.testing.assert_allclose(params, target_params, rtol=1)


def test_estimate2(data: Data) -> None:
    key = jax.random.key(0)

    marginal_effects = jnp.array(data.marginal_effect_array.to_numpy())
    rotated_effects = jnp.array(data.rotated_effect_array.to_numpy())
    median_sample_counts = jnp.array(data.median_sample_count_array.to_numpy())
    min_sample_counts = jnp.array(data.min_sample_count_array.to_numpy())
    snp_counts = jnp.array(data.snp_count_array.to_numpy())
    eig_counts = jnp.array(data.eig_count_array.to_numpy())
    ld_scores = jnp.array(data.ld_score_array.to_numpy())
    eigenvalues = jnp.array(data.eigenvalue_array.to_numpy())
    correlation_matrix = jnp.array(data.correlation_array.to_numpy())

    r(llfun1)
    r(llfun2)
    r("""llfun.gcov.part.2_ <- function(...) {
    return(llfun.gcov.part.2(...)$ll)
}""")

    r("""estimate2 <- function(bhat1, bhat2, bstar1, bstar2, LDsc, N0, N1, N2, M, Nref, lam, rho12){
N <- sqrt(N1) * sqrt(N2)
p1 <- N0/N1
p2 <- N0/N2
a11 <- bhat1^2
a22 <- bhat2^2
a12 <- bhat1 * bhat2
reg <- lm(a11 ~ LDsc)
h11.ols <- c(summary(reg)$coef[1:2, 1:2] * c(N1,M))
reg <- lm(a22 ~ LDsc)
h22.ols <- c(summary(reg)$coef[1:2, 1:2] * c(N2,   M))
reg <- lm(a12 ~ LDsc)
if (N0 > 0)
h12.ols <- c(summary(reg)$coef[1:2, 1:2] * c((N0/p1/p2), M))
if (N0 == 0)
h12.ols <- c(summary(reg)$coef[1:2, 1:2] * c(N, M))
h11v <- (h11.ols[2] * LDsc/M + 1/N1)^2
h22v <- (h22.ols[2] * LDsc/M + 1/N2)^2
reg <- lm(a11 ~ LDsc, weight = 1/h11v)
h11.wls <- c(summary(reg)$coef[1:2, 1:2] * c(N1, M))
reg <- lm(a22 ~ LDsc, weight = 1/h22v)
h22.wls <- c(summary(reg)$coef[1:2, 1:2] * c(N2,M))
if (N0 > 0)
h12v <- sqrt(h11v * h22v) + (h12.ols[2] * LDsc/M + p1 * p2 * rho12/N0)^2
if (N0 == 0)
h12v <- sqrt(h11v * h22v) + (h12.ols[2] * LDsc/M)^2
reg <- lm(a12 ~ LDsc, weight = 1/h12v)
if (N0 > 0)
h12.wls <- c(summary(reg)$coef[1:2, 1:2] * c((N0/p1/p2), M))
if (N0 == 0)
h12.wls <- c(summary(reg)$coef[1:2, 1:2] * c(N, M))
opt <- optim(c(h11.wls[2], 1), llfun, N = N1, Nref = Nref,
            lam = lam, bstar = bstar1, M = M, lim = exp(-18),
            method = "L-BFGS-B", lower = c(0, 0), upper = c(1, 10))
h11.hdl <- opt$par
opt <- optim(c(h22.wls[2], 1), llfun, N = N2, Nref = Nref,
            lam = lam, bstar = bstar2, M = M, lim = exp(-18),
            method = "L-BFGS-B", lower = c(0, 0), upper = c(1, 10))
h22.hdl <- opt$par
opt <- optim(c(h12.wls[2], rho12), llfun.gcov.part.2_,
            h11 = h11.hdl, h22 = h22.hdl, rho12 = rho12,
            M = M, N1 = N1, N2 = N2, N0 = N0, Nref = Nref,
            lam0 = lam, lam1 = lam, lam2 = lam, bstar1 = bstar1,
            bstar2 = bstar2, lim = exp(-18), method = "L-BFGS-B",
            lower = c(-1, -10), upper = c(1, 10))
h12.hdl <- opt$par
logL <- opt$value
return(list(h12 = h12.hdl, logL = logL, initial.params = c(h12.wls[2], rho12)))
}""")  # noqa: E501

    piece_count = data.piece_count
    phenotype_count = data.phenotype_count

    jobs = [
        (piece_index, phenotype_index1, phenotype_index2)
        for piece_index, phenotype_index1 in product(
            range(piece_count), range(phenotype_count)
        )
        for phenotype_index2 in range(phenotype_index1)
    ]

    for piece_index, phenotype_index1, phenotype_index2 in tqdm(jobs, unit="tests"):
        snp_slice = slice(
            snp_counts[:piece_index].sum(), snp_counts[: piece_index + 1].sum()
        )
        eig_slice = slice(
            eig_counts[:piece_index].sum(), eig_counts[: piece_index + 1].sum()
        )
        median_sample_count1 = median_sample_counts[phenotype_index1]
        median_sample_count2 = median_sample_counts[phenotype_index2]
        min_sample_count = jnp.minimum(
            min_sample_counts[phenotype_index1],
            min_sample_counts[phenotype_index2],
        )

        # bhat1, bhat2, LDsc, N1, N2, M, Nref
        n0 = min_sample_count.item()
        n1 = median_sample_count1.item()
        n2 = median_sample_count2.item()
        m = snp_counts[piece_index].item()
        nref = 335265

        bhat1 = np.array(marginal_effects[phenotype_index1, snp_slice])
        bhat2 = np.array(marginal_effects[phenotype_index2, snp_slice])
        bstar1 = np.array(rotated_effects[phenotype_index1, eig_slice])
        bstar2 = np.array(rotated_effects[phenotype_index2, eig_slice])
        ldsc = np.array(ld_scores[snp_slice])
        lam = np.array(eigenvalues[eig_slice])
        rho12 = correlation_matrix[phenotype_index1, phenotype_index2].item()

        with (default_converter + numpy2ri.converter).context():
            result = r["estimate2"](
                bhat1, bhat2, bstar1, bstar2, ldsc, n0, n1, n2, m, nref, lam, rho12
            )
            target_params = result["h12"]
            target_minus_two_log_likelihood = result["logL"].item()

        ref = Reference(
            snp_counts[piece_index],
            eig_counts[piece_index],
            ld_scores[snp_slice],
            eigenvalues[eig_slice],
        )

        s1 = Sample1(
            marginal_effects[phenotype_index1, snp_slice],
            rotated_effects[phenotype_index1, eig_slice],
            median_sample_counts[phenotype_index1],
            min_sample_counts[phenotype_index1],
        )
        with jax.debug_nans(True):
            _, params1 = check_types(estimate1)(s1, ref, key)

        s2 = Sample1(
            marginal_effects[phenotype_index2, snp_slice],
            rotated_effects[phenotype_index2, eig_slice],
            median_sample_counts[phenotype_index2],
            min_sample_counts[phenotype_index2],
        )
        with jax.debug_nans(True):
            _, params2 = check_types(estimate1)(s2, ref, key)

        s = Sample2(
            s1,
            s2,
            params1,
            params2,
            correlation_matrix[phenotype_index1, phenotype_index2],
        )
        with jax.debug_nans(True):
            value, params = check_types(estimate2)(s, ref, key)

        assert (value <= target_minus_two_log_likelihood) or np.isclose(
            value, target_minus_two_log_likelihood
        )
        np.testing.assert_allclose(params, target_params, rtol=1.0, atol=1.0)
