# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import scipy
import torch
from numpy import typing as npt

from ..log import logger
from .ml import OptimizeInput, OptimizeResult, ProfileMaximumLikelihood


@dataclass
class FaSTLMM(ProfileMaximumLikelihood):
    step: float = 0.2

    requires_grad: ClassVar[bool] = False

    def minus_two_log_likelihood(
        self, terms: torch.Tensor, o: OptimizeInput
    ) -> torch.Tensor:
        t = self.get_minus_two_log_likelihood_terms(terms, o)

        variance_ratio = terms[0]
        genetic_variance = t.deviation / t.sample_count
        error_variance = variance_ratio * genetic_variance

        minus_two_log_likelihood = (
            (t.sample_count - 4)
            + t.logarithmic_determinant
            + t.sample_count * torch.log(genetic_variance)
        )

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(
                torch.stack((error_variance, genetic_variance)), o
            )

        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )

    def wrapper(self, log_variance_ratio: npt.NDArray, o: OptimizeInput):
        variance_ratio = np.power(10, log_variance_ratio)
        terms = torch.tensor([variance_ratio, 1], dtype=torch.float64)
        return self.func(terms, o).item()

    def optimize(
        self,
        o: OptimizeInput,
        method: str = "",
        enable_hessian: bool = False,
        disp: bool = False,
        **kwargs,
    ) -> OptimizeResult:
        func = self.wrapper

        upper_bound = float(np.log10(o.rotated_phenotype.var()))
        lower_bound = float(np.log10(self.minimum_variance) - upper_bound)
        xa = np.arange(lower_bound, upper_bound, step=self.step)
        logger.debug(
            f"FaSTLMM will optimize between {lower_bound} to {upper_bound} "
            f"in {xa.size} steps of size {self.step}"
        )

        fmin = np.inf
        best_optimize_result: OptimizeResult | None = None
        with np.errstate(all="ignore"):
            for bounds in zip(xa, xa + self.step):
                try:
                    optimize_result = scipy.optimize.minimize_scalar(  # type: ignore
                        func,
                        args=(o,),
                        bounds=bounds,
                        # options=dict(disp=disp),
                    )
                    if optimize_result.fun < fmin:
                        fmin = optimize_result.fun
                        best_optimize_result = optimize_result
                except FloatingPointError:
                    pass

        if best_optimize_result is None:
            raise RuntimeError

        log_variance_ratio = best_optimize_result.x
        variance_ratio = np.power(10, log_variance_ratio)

        # Scale by genetic variance
        terms = torch.Tensor([variance_ratio, 1])
        _, _, residuals, _ = self.get_standard_errors(terms, o)
        genetic_variance = float(np.square(residuals).mean())
        terms *= genetic_variance

        return OptimizeResult(
            x=terms.numpy(),
            fun=fmin,
        )


@dataclass
class PenalizedFaSTLMM(FaSTLMM):
    def minus_two_log_likelihood(
        self, terms: torch.Tensor, o: OptimizeInput
    ) -> torch.Tensor:
        t = self.get_minus_two_log_likelihood_terms(terms, o)

        variance_ratio = terms[0]
        genetic_variance = t.deviation / (t.sample_count - 4)
        error_variance = variance_ratio * genetic_variance

        penalty = -2 * torch.log(variance_ratio)

        minus_two_log_likelihood = (
            (t.sample_count - 4)
            + t.logarithmic_determinant
            + t.sample_count * torch.log(genetic_variance)
            + penalty
        )

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(
                torch.stack((error_variance, genetic_variance)), o
            )

        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )
