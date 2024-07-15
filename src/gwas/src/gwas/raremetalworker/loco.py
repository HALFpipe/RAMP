from pathlib import Path
from subprocess import check_call
from typing import NamedTuple

from ..defaults import (
    default_kinship_minor_allele_frequency_cutoff,
)
from ..plink import FamFile
from ..tools import plink2, raremetalworker, tabix
from .ped import write_dummy_ped_and_dat_files


class RaremetalworkerKinshipCommand(NamedTuple):
    command: list[str]
    kinship_path: Path


def make_loco_kinship_command(
    chromosome: int | str,
    bfile_path: Path,
    prefix: Path,
    minor_allele_frequency_cutoff: float = default_kinship_minor_allele_frequency_cutoff,
) -> RaremetalworkerKinshipCommand:
    output_directory = prefix / f"chr{chromosome}"
    output_directory.mkdir(parents=True, exist_ok=True)

    samples = FamFile(bfile_path).read_samples()
    ped_path, dat_path = write_dummy_ped_and_dat_files(samples, prefix)

    vcf_prefix = output_directory / "loco"
    vcf_path = vcf_prefix.with_suffix(".vcf.gz")
    if not vcf_path.is_file():
        check_call(
            [
                *plink2,
                "--bfile",
                str(bfile_path),
                "--not-chr",
                str(chromosome),
                "--export",
                "vcf",
                "bgz",
                "id-paste=iid",
                "--out",
                str(vcf_prefix),
            ]
        )
        check_call([*tabix, "--preset", "vcf", str(vcf_path)])

    kinship_path = output_directory / "Empirical.Kinship.gz"
    command: list[str] = [
        *raremetalworker,
        "--ped",
        str(ped_path),
        "--dat",
        str(dat_path),
        "--vcf",
        str(vcf_path),
        "--kinGeno",
        "--kinSave",
        "--kinOnly",
        "--kinMaf",
        f"{minor_allele_frequency_cutoff}",
        "--noPhoneHome",
        "--prefix",
        f"{output_directory}/",
    ]
    return RaremetalworkerKinshipCommand(command, kinship_path)
