from shlex import join
from typing import NamedTuple

from upath import UPath

from ..pheno import VariableCollection
from ..tools import raremetalworker
from .ped import write_ped_and_dat_files


class RaremetalworkerScoreCommand(NamedTuple):
    command: list[str]
    scorefile_path: UPath


def make_raremetalworker_score_commands(
    chromosome: int | str,
    vc: VariableCollection,
    vcf_gz_path: UPath,
    prefix: UPath,
    dosage: bool = True,
    kinship_path: UPath | None = None,
) -> list[RaremetalworkerScoreCommand]:
    output_directory = prefix / f"chr{chromosome}"
    output_directory.mkdir(parents=True, exist_ok=True)

    commands: list[RaremetalworkerScoreCommand] = list()

    commands_path = output_directory / "commands.txt"
    with commands_path.open("a") as file_handle:
        for j, phenotype_name in enumerate(vc.phenotype_names):
            ped_path, dat_path = write_ped_and_dat_files(vc, j, prefix)

            command: list[str] = [
                *raremetalworker,
                "--ped",
                str(ped_path),
                "--dat",
                str(dat_path),
                "--vcf",
                str(vcf_gz_path),
                "--prefix",
                f"{output_directory}/",
                "--LDwindow",
                "100",
                "--zip",
                "--thin",
                "--noPhoneHome",
                "--useCovariates",
            ]
            if dosage:
                command.append("--dosage")
            if kinship_path is not None:
                command.extend(["--kinFile", str(kinship_path)])
            file_handle.write(join(command))
            file_handle.write("\n")

            scorefile_path = (
                output_directory / f"{phenotype_name}.singlevar.score.txt.gz"
            )
            commands.append(RaremetalworkerScoreCommand(command, scorefile_path))

    return commands
