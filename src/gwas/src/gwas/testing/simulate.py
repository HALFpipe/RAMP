import random
from dataclasses import dataclass
from subprocess import check_call
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from numpy import typing as npt
from upath import UPath

from ..mem.wkspace import SharedWorkspace
from ..pheno import VariableCollection
from ..tools import gcta64


@dataclass
class SimulationResult:
    phenotype_names: list[str]

    phen_frame: pd.DataFrame
    par_frame: pd.DataFrame

    patterns: npt.NDArray[np.bool_]
    pattern_indices: npt.NDArray[np.int_]

    @property
    def simulation_count(self) -> int:
        return len(self.phenotype_names)

    @property
    def phenotypes(self) -> npt.NDArray[np.float64]:
        return self.phen_frame[self.phenotype_names].to_numpy()

    @property
    def samples(self) -> list[str]:
        return list(self.phen_frame.individual_id)

    def to_variable_collection(
        self, sw: SharedWorkspace, covariate_count: int = 4, seed: int = 47
    ) -> VariableCollection:
        samples = list(self.phen_frame.individual_id)
        covariate_names = [f"covariate-{i + 1:02d}" for i in range(covariate_count)]

        np.random.seed(seed)
        covariates = np.random.normal(size=(len(samples), covariate_count))
        covariates -= covariates.mean(axis=0)

        return VariableCollection.from_arrays(
            samples,
            self.phenotype_names,
            self.phenotypes,
            covariate_names,
            covariates,
            sw,
            missing_value_strategy="listwise_deletion",
        )


def simulate(
    bfile_path: UPath,
    variant_ids: list[str],
    causal_variant_count: int,
    heritability: float,
    simulation_count: int,
    seed: int,
    missing_value_rate: float,
    missing_value_pattern_count: int,
    simulation_path: UPath,
) -> SimulationResult:
    phen_path = simulation_path.parent / f"{simulation_path.name}.phen"
    par_path = simulation_path.parent / f"{simulation_path.name}.par"

    if not phen_path.is_file() or not par_path.is_file():
        random.seed(seed)
        causal_variants = random.sample(variant_ids, k=causal_variant_count)
        with NamedTemporaryFile(mode="wt", delete_on_close=False) as file_handle:
            file_handle.write("\n".join(causal_variants))
            file_handle.close()

            variant_list_path = UPath(file_handle.name)

            gcta_command = [
                *gcta64,
                "--bfile",
                str(bfile_path),
                "--simu-qt",
                "--simu-hsq",
                str(heritability),
                "--simu-causal-loci",
                str(variant_list_path),
                "--simu-rep",
                str(simulation_count),
                "--simu-seed",
                f"{seed}",
                "--out",
                str(simulation_path),
            ]
            check_call(gcta_command)

    digits = len(str(simulation_count))
    phenotype_names = [f"phenotype-{i:0{digits}d}" for i in range(simulation_count)]

    # Load simulation result
    phen_frame = pd.read_table(
        phen_path,
        sep=" ",
        header=None,
        index_col=False,
        names=["family_id", "individual_id", *phenotype_names],
        dtype={
            "family_id": "string",
            "individual_id": "string",
        },
    )
    par_frame = pd.read_table(
        par_path,
        sep="\t",
        dtype={
            "QTL": "string",
            "RefAllele": "category",
            "Frequency": "float64",
            "Effect": "float64",
        },
    )
    sample_count, _ = phen_frame.shape

    # Generate missing value patterns
    np.random.seed(seed)
    patterns = generate_missing_value_patterns(
        missing_value_rate, missing_value_pattern_count, sample_count
    )
    pattern_indices: npt.NDArray[np.int_] = np.random.permutation(
        np.resize(np.arange(missing_value_pattern_count), simulation_count)
    )

    # Apply missing value patterns
    phen_frame[phenotype_names] = phen_frame[phenotype_names].mask(
        patterns[:, pattern_indices], other=np.nan
    )

    return SimulationResult(
        phenotype_names, phen_frame, par_frame, patterns, pattern_indices
    )


def generate_missing_value_patterns(
    missing_value_rate: float, missing_value_pattern_count: int, sample_count: int
) -> npt.NDArray[np.bool_]:
    patterns: npt.NDArray[np.bool_] = np.vstack(
        [
            np.random.choice(
                a=[False, True],
                size=sample_count,
                p=[1 - missing_value_rate, missing_value_rate],
            )
            for _ in range(missing_value_pattern_count)
        ]
    ).transpose()

    return patterns
