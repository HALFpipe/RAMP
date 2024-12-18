from unittest import mock

from upath import UPath

from gwas.upload_score.cli import run


def test_upload_score(tmp_path: UPath) -> None:
    input_paths = [tmp_path / "chr1.metadata.yaml.gz", tmp_path / "chr1.score.b2array"]
    for path in input_paths:
        path.touch()
    with mock.patch("gwas.upload_score.cli.call_upload_client") as call_upload_client:
        run(
            [
                "--input-directory",
                str(tmp_path),
                "--token",
                "token",
                "--endpoint",
                "endpoint",
            ],
            error_action="raise",
        )
        call_upload_client.assert_called_once()
        _, paths = call_upload_client.call_args.args
        assert set(paths) == {str(path).removeprefix("/") for path in input_paths}


def test_upload_score_multiple(tmp_path: UPath) -> None:
    input_paths = [
        tmp_path / "a" / "a" / "a" / "chr1.metadata.yaml.gz",
        tmp_path / "a" / "a" / "a" / "chr1.score.b2array",
        tmp_path / "a" / "b" / "a" / "chr1.metadata.yaml.gz",
        tmp_path / "a" / "b" / "a" / "chr1.score.b2array",
    ]
    for path in input_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with mock.patch("gwas.upload_score.cli.call_upload_client") as call_upload_client:
        run(
            [
                "--input-directory",
                str(tmp_path),
                "--token",
                "token",
                "--endpoint",
                "endpoint",
            ],
            error_action="raise",
        )
        call_upload_client.assert_called_once()
        _, paths = call_upload_client.call_args.args
        assert set(paths) == {str(path).removeprefix("/") for path in input_paths}
