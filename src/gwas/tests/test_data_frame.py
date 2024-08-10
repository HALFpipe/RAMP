import pandas as pd
from gwas.mem.data_frame import SharedDataFrame
from gwas.mem.wkspace import SharedWorkspace


def test_shared_data_frame(sw: SharedWorkspace):
    data_frame = pd.DataFrame(
        data=dict(
            a=pd.Series(pd.Categorical.from_codes([0, 1, 2], pd.Index([1, 2, 0]))),
            b=pd.Series(pd.Categorical.from_codes([0, 1, 2], pd.Index(["a", "b", "c"]))),
            c=pd.Series([1, 2, 3]),
            d=pd.Series(["a", "b", "c"]),
            e=pd.Series([1.0, 2.0, 3.0]),
        )
    )
    shared_data_frame = SharedDataFrame.from_pandas(data_frame, sw)
    pd.testing.assert_frame_equal(data_frame, shared_data_frame.to_pandas())
