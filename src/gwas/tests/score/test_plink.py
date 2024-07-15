from pathlib import Path

from gwas.plink import BimFile, FamFile, PsamFile, PVarFile


def test_psam(pfile_paths: list[Path]) -> None:
    pfile_path = pfile_paths[0]
    psam_file = PsamFile(pfile_path)
    samples = psam_file.read_samples()

    assert len(samples) > 0


def test_pvar(pfile_paths: list[Path]) -> None:
    pfile_path = pfile_paths[0]
    pvar_file = PVarFile(pfile_path)
    variant_ids = pvar_file.read_variant_ids()

    assert len(variant_ids) > 0


def test_fam(bfile_path: Path) -> None:
    fam_file = FamFile(bfile_path)
    samples = fam_file.read_samples()

    assert len(samples) > 0


def test_bim(bfile_path: Path) -> None:
    bim_file = BimFile(bfile_path)
    variant_ids = bim_file.read_variant_ids()

    assert len(variant_ids) > 0
