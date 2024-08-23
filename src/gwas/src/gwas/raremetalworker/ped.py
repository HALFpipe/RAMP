import numpy as np
from numpy import typing as npt
from upath import UPath

from ..pheno import VariableCollection
from ..utils import to_str


def format_row(a: npt.NDArray[np.float64]) -> str:
    if a.size == 1:
        return to_str(a)
    else:
        return " ".join(map(to_str, a))


def write_ped_and_dat_files(
    vc: VariableCollection, phenotype_index: int, path: UPath
) -> tuple[UPath, UPath]:
    base = vc.phenotype_names[phenotype_index]
    covariates = vc.covariates
    phenotypes = vc.phenotypes

    ped_path = path / f"{base}.ped"
    with ped_path.open("w") as file_handle:
        for i, s in enumerate(vc.samples):
            c = format_row(covariates[i, 1:])  # Skip intercept
            p = format_row(phenotypes[i, phenotype_index])
            file_handle.write(f"{s} {s} 0 0 1 {p} {c}\n")

    dat_path = path / f"{base}.dat"
    with dat_path.open("w") as file_handle:
        file_handle.write(f"T {base}\n")
        for name in vc.covariate_names[1:]:  # Skip intercept
            file_handle.write(f"C {name}\n")

    return ped_path, dat_path


def write_dummy_ped_and_dat_files(
    samples: list[str], path: UPath
) -> tuple[UPath, UPath]:
    ped_path = path / "dummy.ped"
    with ped_path.open("w") as file_handle:
        for s in samples:
            file_handle.write(f"{s} {s} 0 0 1 0\n")

    dat_path = path / "dummy.dat"
    with dat_path.open("w") as file_handle:
        file_handle.write("T variable")

    return ped_path, dat_path
