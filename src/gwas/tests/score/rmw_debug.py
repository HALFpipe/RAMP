# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass, fields
from subprocess import DEVNULL, PIPE, Popen
from typing import Any, Callable

import numpy as np
import pytest
from gwas.log import logger
from gwas.raremetalworker.score import RaremetalworkerScoreCommand
from gwas.utils import unwrap_which
from more_itertools import peekable
from numpy import typing as npt

field_mapping: dict[str, str] = {
    "X:": "x",
    "allPairs:": "all_pairs",
    "transU_del:": "trans_u_del",
    "D:": "d",
    "betaHat is:": "beta_hat",
    "transUX are:": "trans_u_x",
    "transUY is:": "trans_u_y",
    "sigma2 is:": "sigma2",
    "residuals are:": "residuals",
    "genotype is:": "genotype",
    "transGeno is:": "trans_geno",
    "beta is:": "beta",
    "factor is:": "factor",
}

sigma_hat_pattern = re.compile(
    r"sigma_g2Hat,sigma_e2Hat,sigma_gXHat,sigma_s2Hat are: "
    r"(?P<sigma_g2_hat>.+), (?P<sigma_e2_hat>.+), "
    r"(?P<sigma_gx_hat>.+), (?P<sigma_s2_hat>.+)"
)
fraction_pattern = re.compile(
    r"numerator and denominator are: (?P<numerator>.+),(?P<denominator>.+)."
)
evaluate_pattern = re.compile(
    r"sigma is (?P<sigma>.+), delta is (?P<delta>.+), "
    r"constant is (?P<constant>.+), logLikelihood is (?P<log_likelihood>.+)"
)
hat_pattern = re.compile(
    r"logLikelihood is:(?P<log_likelihood_hat>.+),deltaHat is:(?P<delta_hat>.+)"
)


@dataclass
class RmwDebug:
    x: npt.NDArray[np.float64]
    all_pairs: npt.NDArray[np.float64]
    trans_u_del: npt.NDArray[np.float64]
    d: npt.NDArray[np.float64]  # Eigenvalues
    beta: list[npt.NDArray[np.float64]]
    factor: list[npt.NDArray[np.float64]]
    sigma: npt.NDArray[np.float64]
    delta: npt.NDArray[np.float64]
    constant: npt.NDArray[np.float64]
    log_likelihood: npt.NDArray[np.float64]
    beta_hat: npt.NDArray[np.float64]
    log_likelihood_hat: float
    delta_hat: float
    sigma_g2_hat: float
    sigma_e2_hat: float
    sigma_gx_hat: float
    sigma_s2_hat: float
    trans_u_x: npt.NDArray[np.float64]
    trans_u_y: npt.NDArray[np.float64]
    sigma2: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    numerator: float
    denominator: float
    genotype: npt.NDArray[np.float64]
    trans_geno: npt.NDArray[np.float64]


def read_matrix(
    file_handle: "peekable[str]",
) -> npt.NDArray[np.float64]:
    pattern = re.compile(r"^[0-9-]")

    lines: list[str] = list()
    for line in file_handle:
        if pattern.match(line) is not None:
            lines.append(line)
        if not file_handle:
            break
        if pattern.match(file_handle.peek()) is None:
            break

    return np.loadtxt(lines, dtype=np.float64)


@pytest.fixture(scope="module")
def rmw_debug(
    phenotype_index: int,
    raremetalworker_score_commands: list[RaremetalworkerScoreCommand],
) -> RmwDebug:
    command, _ = raremetalworker_score_commands[phenotype_index]

    command = [unwrap_which("raremetalworker-debug"), *command[1:]]

    prefix_index = command.index("--prefix")
    command[prefix_index + 1] = f"{command[prefix_index + 1]}-debug"

    process_handle = Popen(
        command,
        stderr=DEVNULL,
        stdin=DEVNULL,
        stdout=PIPE,
        text=True,
    )
    stdout = process_handle.stdout
    if stdout is None:
        raise ValueError("stdout is None")

    peekable_stdout = peekable(stdout)

    data: dict[str, Any] = dict(
        sigma=list(),
        delta=list(),
        constant=list(),
        log_likelihood=list(),
        beta=list(),
        factor=list(),
    )

    def put_match(groupdict: dict[str, str]) -> None:
        for key, value in groupdict.items():
            if key in data:
                raise ValueError(f'Field "{key}" has already been set')
            logger.debug(f'Found "{key}"')
            data[key] = float(value)

    field_count = len(fields(RmwDebug))
    process_data(peekable_stdout, data, put_match, field_count)

    process_handle.terminate()

    data["sigma"] = np.array(data["sigma"], dtype=np.float64)
    data["delta"] = np.array(data["delta"], dtype=np.float64)
    data["constant"] = np.array(data["constant"], dtype=np.float64)
    data["log_likelihood"] = np.array(data["log_likelihood"], dtype=np.float64)

    rmw_debug = RmwDebug(**data)
    sample_count, covariate_count = rmw_debug.x.shape
    assert rmw_debug.all_pairs.shape == (sample_count, sample_count)
    assert rmw_debug.trans_u_del.shape == (sample_count, sample_count)
    assert rmw_debug.d.shape == (sample_count,)

    assert rmw_debug.log_likelihood.size > 0
    assert rmw_debug.delta.size > 0

    assert rmw_debug.beta_hat.shape == (covariate_count,)
    assert rmw_debug.trans_u_x.shape == (sample_count, covariate_count)
    assert rmw_debug.trans_u_y.shape == (sample_count,)
    assert rmw_debug.sigma2.shape == (sample_count,)
    assert rmw_debug.residuals.shape == (sample_count,)
    assert rmw_debug.genotype.shape == (sample_count,)

    assert np.allclose(
        rmw_debug.residuals,
        rmw_debug.trans_u_y - rmw_debug.trans_u_x @ rmw_debug.beta_hat,
        atol=1e-3,
        rtol=1e-3,
    )

    assert np.allclose(
        rmw_debug.sigma2,
        rmw_debug.sigma_g2_hat * rmw_debug.d + rmw_debug.sigma_e2_hat,
        atol=1e-3,
        rtol=1e-3,
    )

    assert np.isclose(
        rmw_debug.numerator,
        (rmw_debug.trans_geno / rmw_debug.sigma2 * rmw_debug.residuals).sum(),
        atol=1e-3,
        rtol=1e-3,
    )

    assert np.isclose(
        rmw_debug.denominator,
        (np.square(rmw_debug.trans_geno) / rmw_debug.sigma2).sum(),
        atol=1e-3,
        rtol=1e-3,
    )

    return rmw_debug


def process_data(
    peekable_stdout: "peekable[str]",
    data: dict[str, Any],
    put_match: Callable[[dict[str, str]], None],
    field_count: int,
) -> None:
    for line in peekable_stdout:
        line = line.strip()

        evaluate_match = evaluate_pattern.fullmatch(line)
        sigma_hat_match = sigma_hat_pattern.fullmatch(line)
        fraction_match = fraction_pattern.fullmatch(line)
        hat_match = hat_pattern.fullmatch(line)

        if evaluate_match is not None:
            for key, value in evaluate_match.groupdict().items():
                data[key].append(float(value))
        elif sigma_hat_match is not None:
            put_match(sigma_hat_match.groupdict())
        elif fraction_match is not None:
            put_match(fraction_match.groupdict())
        elif hat_match is not None:
            process_hat(data, hat_match)
        elif line in field_mapping:
            process_matrix(peekable_stdout, data, line)
        if len(data) == field_count:
            break


def process_hat(data: dict[str, Any], hat_match: re.Match[str]) -> None:
    for key, value in hat_match.groupdict().items():
        if key not in data:
            logger.debug(f'Found "{key}"')
        data[key] = float(value)


def process_matrix(
    peekable_stdout: "peekable[str]", data: dict[str, Any], line: str
) -> None:
    field = field_mapping[line]
    if field in data:
        if isinstance(data[field], list):
            data[field].append(read_matrix(peekable_stdout))
        else:
            raise ValueError(f'Field "{field}" has already been set')
    else:
        logger.debug(f'Found "{field}"')
        data[field] = read_matrix(peekable_stdout)
