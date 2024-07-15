from multiprocessing import cpu_count
from pathlib import Path
from shlex import join
from subprocess import check_call
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import NamedTuple, Sequence

from tqdm.auto import tqdm

from ..compression.pipe import CompressedTextReader
from ..defaults import default_score_r_squared_cutoff
from ..plink import PVarFile
from ..tools import bcftools, plink2
from ..utils import IterationOrder, is_bfile, is_pfile, make_pool_or_null_context

plink2_extra_arguments: Sequence[str] = [
    "--threads",
    str(1),
    "--memory",
    str(2**10),
]


class ConvertVCFToPfileJob(NamedTuple):
    vcf_path: Path
    pfile_path: Path


def convert_vcf_to_pfile(job: ConvertVCFToPfileJob) -> None:
    with (
        CompressedTextReader(job.vcf_path) as file_handle,
        TemporaryDirectory() as temporary_directory_str,
    ):
        temporary_directory_path = Path(temporary_directory_str)
        gen_path = temporary_directory_path / "plink.gen.gz"
        sample_path = temporary_directory_path / "plink.sample"
        check_call(
            [
                *bcftools,
                "convert",
                "--exclude",
                f"INFO/R2<{default_score_r_squared_cutoff}",
                "--gensample",
                f"{gen_path},{sample_path}",
                "--3N6",
                "-",
            ],
            stdin=file_handle,
        )
        check_call(
            [
                *plink2,
                "--silent",
                "--gen",
                str(gen_path),
                "ref-first",
                "--sample",
                str(sample_path),
                *plink2_extra_arguments,
                "--make-pgen",
                "--out",
                str(job.pfile_path),
            ]
        )


def convert_vcf_to_pfiles(
    vcf_paths: list[Path], output_directory: Path, num_threads: int = cpu_count()
) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)

    pfiles: list[Path] = list()
    jobs: list[ConvertVCFToPfileJob] = list()
    for vcf_path in vcf_paths:
        pfile_path = output_directory / vcf_path.name.split(".")[0]
        pfiles.append(pfile_path)

        if is_pfile(pfile_path):
            continue

        jobs.append(ConvertVCFToPfileJob(vcf_path, pfile_path))

    pool, iterator = make_pool_or_null_context(
        jobs,
        convert_vcf_to_pfile,
        num_threads=num_threads,
        iteration_order=IterationOrder.ORDERED,
    )

    with pool:
        for _ in tqdm(
            iterator, unit="files", desc="converting vcf to pfile", total=len(jobs)
        ):
            pass

    return pfiles


def convert_vcf_to_bgen(vcf_path: Path, bgen_prefix: Path) -> Path:
    converted_bgen_path = bgen_prefix.with_suffix(".bgen")
    if not converted_bgen_path.is_file():
        with TemporaryDirectory() as temporary_directory_str:
            temporary_directory = Path(temporary_directory_str)
            check_call(
                [
                    *plink2,
                    "--vcf",
                    str(vcf_path),
                    "dosage=DS",
                    "--double-id",
                    "--make-pgen",
                    "erase-phase",
                    "--out",
                    str(temporary_directory / "plink"),
                ]
            )
            check_call(
                [
                    *plink2,
                    "--pfile",
                    str(temporary_directory / "plink"),
                    "--export",
                    "bgen-1.2",
                    "bits=8",
                    "--out",
                    str(bgen_prefix),
                ]
            )
    return converted_bgen_path


def merge_pfiles_to_bfile(pfile_paths: list[Path], bfile_path: Path) -> None:
    if is_bfile(bfile_path):
        return

    with NamedTemporaryFile(mode="wt", delete_on_close=False) as file_handle:
        file_handle.write("\n".join(map(str, pfile_paths)))
        file_handle.close()

        pfile_list_path = Path(file_handle.name)
        check_call(
            [
                *plink2,
                "--silent",
                "--pmerge-list",
                str(pfile_list_path),
                "--delete-pmerge-result",
                "--maf",
                "0.01",
                "--mind",
                "0.1",
                "--geno",
                "0.1",
                "--hwe",
                "1e-50",
                "--make-bed",
                "--out",
                str(bfile_path),
            ]
        )


def get_pfile_variant_ids(pfile_paths: list[Path]) -> list[str]:
    variant_ids: list[str] = list()
    for pfile_path in pfile_paths:
        pvar_file = PVarFile(pfile_path.with_suffix(".pvar"))
        variant_ids.extend(pvar_file.read_variant_ids())

    return variant_ids


def merge_vcf_gz_files(
    vcf_gz_paths: list[Path],
    bgzip_prefix: Path,
    num_threads: int = cpu_count(),
) -> Path:
    vcf_gz_path = bgzip_prefix / "dose.vcf.gz"
    if not vcf_gz_path.is_file():
        concat_command: list[str] = [
            *bcftools,
            "concat",
            f"--threads={num_threads}",
            *map(str, vcf_gz_paths),
        ]
        filter_command: list[str] = [
            *bcftools,
            "view",
            f"--threads={num_threads}",
            "--min-ac",
            "1",
            "--output-type=z",
            f"--output={vcf_gz_path}",
            "-",
        ]
        check_call(["bash", "-c", f"{join(concat_command)} | {join(filter_command)}"])
    return vcf_gz_path
