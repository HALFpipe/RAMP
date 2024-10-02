import pandas as pd

from gwas.mem.data_frame import SharedDataFrame, concat
from gwas.mem.wkspace import SharedWorkspace

data_frame = pd.DataFrame(
    data=dict(
        a=pd.Series(pd.Categorical.from_codes([0, 1, 2], pd.Index([1, 2, 0]))),
        b=pd.Series(pd.Categorical.from_codes([0, 1, 2], pd.Index(["a", "b", "c"]))),
        c=pd.Series([1, 2, 3]),
        d=pd.Series(["a", "b", "c"]),
        e=pd.Series([1.0, 2.0, 3.0]),
    )
)


def test_shared_data_frame(sw: SharedWorkspace) -> None:
    shared_data_frame = SharedDataFrame.from_pandas(data_frame, sw)
    pd.testing.assert_frame_equal(data_frame, shared_data_frame.to_pandas())

    for name, column in data_frame.items():
        assert isinstance(name, str)
        assert isinstance(column, pd.Series)
        shared_series = shared_data_frame[name]
        pd.testing.assert_series_equal(column, shared_series.to_pandas())


def test_concat(sw: SharedWorkspace) -> None:
    other_data_frame = pd.DataFrame(
        data=dict(
            a=pd.Series(pd.Categorical.from_codes([0, 1, 2], pd.Index([1, 2, 0]))),
            b=pd.Series(pd.Categorical.from_codes([0, 1, 2], pd.Index(["c", "d", "e"]))),
            c=pd.Series([1, 2, 3]),
            d=pd.Series(["a", "b", "c"]),
            e=pd.Series([1.0, 2.0, 3.0]),
        )
    )
    data_frames = [data_frame, other_data_frame]
    shared_data_frames = [SharedDataFrame.from_pandas(d, sw) for d in data_frames]
    shared_data_frame = concat(shared_data_frames)
    pd.testing.assert_frame_equal(
        pd.concat(data_frames, ignore_index=True),
        shared_data_frame.to_pandas(),
        check_dtype=False,
        check_categorical=False,
    )

    b = shared_data_frame["b"].to_pandas()
    assert b.dtype == pd.CategoricalDtype(["a", "b", "c", "d", "e"])
    assert b.tolist() == ["a", "b", "c", "c", "d", "e"]
