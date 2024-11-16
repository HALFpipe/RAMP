import zipfile
from subprocess import run
from typing import Iterator

import polars as pl
from tqdm.auto import tqdm
from upath import UPath

from ...log import logger
from ...tools import sha256sum
from ...utils.download import download
from ...utils.genetics import chromosomes_list

hostname = "download.gwas.science"
directory = "imputationserver"
filename = "1000genomes-phase3-3.0.0.zip"
checksum_sha256 = "1910bebd658a406aa0f263bc886151b3060569c808aaf58dc149ad73b1283594"


def get_panel_path(cache_path: UPath) -> UPath:
    path = cache_path / filename
    if not path.is_file():
        url = f"https://{hostname}/{directory}/{filename}"
        with path.open("wb") as destination:
            download(url, destination)
        run(
            [*sha256sum, "--check"],
            input=f"{checksum_sha256}  {path}\n",
            check=True,
            text=True,
        )
    return path


def make_data_frame_iterator(path: UPath) -> Iterator[pl.DataFrame]:
    for chromosome in tqdm(chromosomes_list(), unit=" " + "chromosomes"):
        with (
            zipfile.Path(path)
            / "legends"
            / f"ALL_1000G_phase3_integrated_v5_chr{chromosome}.legend.gz"
        ).open("rb") as file_handle:
            data_frame = pl.read_csv(file_handle, has_header=True, separator=" ")
            chromosome_series = pl.Series(
                "chromosome", [str(chromosome)] * data_frame.height, dtype=pl.String
            )
            data_frame = data_frame.insert_column(0, chromosome_series)

            chromosome_str = "chrX" if chromosome == "X" else str(chromosome)
            rsid = (
                data_frame["id"]
                .str.split(":")
                .list.first()
                .replace(chromosome_str, None)
                .alias("rsid")
            )
            id = pl.concat_str(
                data_frame.select(["chromosome", "position", "a0", "a1"]),
                separator=":",
            ).alias("id")
            data_frame = (
                data_frame.drop("id").insert_column(1, id).insert_column(2, rsid)
            )

            yield data_frame


def make_legends_data_frame(cache_path: UPath) -> pl.DataFrame:
    cache_path.mkdir(parents=True, exist_ok=True)

    legends_path = cache_path / "legends.parquet"
    if legends_path.is_file():
        return pl.read_parquet(legends_path)

    panel_path = get_panel_path(cache_path)
    data_frame = pl.concat(make_data_frame_iterator(panel_path))

    data_frame.write_parquet(legends_path)

    return data_frame


def join_legends_data_frame(
    cache_path: UPath, data_frame: pl.DataFrame, population: str
) -> tuple[pl.DataFrame, str]:
    legends = make_legends_data_frame(cache_path)

    column = f"{population.lower()}.aaf"
    if column not in legends.columns:
        logger.warning(
            f'Population "{population}" not found in reference. '
            'Using "super_pop" instead'
        )
        column = "super_pop.aaf"

    legends = legends.select(["id", "rsid", column])
    legends = legends.rename(
        {"id": "marker_name", column: "reference_alternate_allele_frequency"}
    )
    data_frame = data_frame.join(legends, on="marker_name", how="left")
    return data_frame, column.split(".")[0]
