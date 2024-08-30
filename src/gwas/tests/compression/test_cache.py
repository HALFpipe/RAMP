import numpy as np
import pandas as pd
from upath import UPath

from gwas.compression.cache import load_from_cache, save_to_cache
from gwas.mem.arr import SharedArray
from gwas.mem.data_frame import SharedDataFrame
from gwas.mem.wkspace import SharedWorkspace


def test_cache_array(tmp_path: UPath, sw: SharedWorkspace) -> None:
    numpy_array = np.full((7, 7), 7, dtype=np.int64)

    shared_array = SharedArray.from_numpy(numpy_array, sw)

    save_to_cache(tmp_path, "shared_array", shared_array, num_threads=1)

    loaded_shared_array = load_from_cache(tmp_path, "shared_array", sw)
    np.testing.assert_array_equal(loaded_shared_array.to_numpy(), numpy_array)


def test_cache_data_frame(tmp_path: UPath, sw: SharedWorkspace) -> None:
    data_frame = pd.DataFrame(
        dict(
            a=pd.Series(pd.Categorical.from_codes([0, 1, 2], pd.Index([1, 2, 0]))),
            b=pd.Series(pd.Categorical.from_codes([0, 1, 2], pd.Index(["a", "b", "c"]))),
            c=pd.Series([1, 2, 3]),
            d=pd.Series(["a", "b", "c"]),
            e=pd.Series([1.0, 2.0, 3.0]),
        )
    )

    shared_data_frame = SharedDataFrame.from_pandas(data_frame, sw)

    save_to_cache(tmp_path, "shared_data_frame", shared_data_frame, num_threads=1)

    loaded_shared_data_frame = load_from_cache(tmp_path, "shared_data_frame", sw)
    pd.testing.assert_frame_equal(loaded_shared_data_frame.to_pandas(), data_frame)
