from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Self, Sequence, override

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from .arr import SharedArray
from .wkspace import SharedWorkspace


@dataclass
class SharedSeries:
    name: Hashable | None
    values: SharedArray

    @property
    def sw(self) -> SharedWorkspace:
        return self.values.sw

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def allocation_names(self) -> set[str]:
        return {self.values.name}

    def free(self) -> None:
        self.values.free()

    def to_pandas(self) -> pd.Series:
        return pd.Series(data=self.values.to_numpy(), name=self.name, copy=False)

    @classmethod
    def _from_pandas(cls, series: pd.Series, sw: SharedWorkspace) -> Self:
        shared_array = SharedArray.from_numpy(series.to_numpy(), sw, prefix="series")
        return cls(series.name, shared_array)

    @staticmethod
    def from_pandas(series: pd.Series, sw: SharedWorkspace) -> "SharedSeries":
        if isinstance(series.dtype, pd.CategoricalDtype):
            return SharedCategorical._from_pandas(series, sw)
        elif is_object_dtype(series) or is_string_dtype(series):
            return SharedString._from_pandas(series, sw)
        else:
            return SharedSeries._from_pandas(series, sw)

    @classmethod
    def _concat(cls, series: Iterable[pd.Series]) -> pd.Series:
        return pd.concat(series, copy=False)

    @classmethod
    def concat(cls, shared_series: Sequence[Self]) -> "SharedSeries":
        sw = shared_series[0].sw
        names = {s.name for s in shared_series}
        (name,) = names

        series = cls._concat((s.to_pandas() for s in shared_series))
        series.name = name
        return cls.from_pandas(series, sw)


@dataclass
class SharedString(SharedSeries):
    @override
    @classmethod
    def _from_pandas(cls, series: pd.Series, sw: SharedWorkspace) -> Self:
        values_numpy = series.to_numpy().astype(np.str_)
        values = SharedArray.from_numpy(values_numpy, sw, prefix="string")
        return cls(series.name, values)


class Categories(SharedArray):
    def to_pandas(self) -> Iterable[Any]:
        return self.to_numpy()

    @classmethod
    def _from_pandas(cls, series: pd.Series, sw: SharedWorkspace) -> Self:
        return cls.from_numpy(series.cat.categories.to_numpy(), sw, prefix="categories")

    @staticmethod
    def from_pandas(series: pd.Series, sw: SharedWorkspace) -> "Categories":
        if is_string_dtype(series.cat.categories):
            return StringCategories._from_pandas(series, sw)
        else:
            return Categories._from_pandas(series, sw)


class StringCategories(Categories):
    @override
    def to_pandas(self) -> Iterable[Any]:
        return self.to_numpy().tobytes().decode().split("\0")

    @override
    @classmethod
    def _from_pandas(cls, series: pd.Series, sw: SharedWorkspace) -> Self:
        categories_bytes = "\0".join(series.cat.categories).encode()
        categories_numpy = np.frombuffer(categories_bytes, dtype=np.uint8)
        return cls.from_numpy(categories_numpy, sw, prefix="string-categories")


@dataclass
class SharedCategorical(SharedSeries):
    values: SharedArray[np.integer]
    categories: Categories | list[Any]

    @override
    @property
    def allocation_names(self) -> set[str]:
        names = {self.values.name}
        if isinstance(self.categories, Categories):
            names.add(self.categories.name)
        return names

    @override
    def free(self) -> None:
        if isinstance(self.categories, Categories):
            self.categories.free()
        super().free()

    @override
    def to_pandas(self) -> pd.Series:
        if isinstance(self.categories, Categories):
            categories = self.categories.to_pandas()
        else:
            categories = self.categories
        categorical = pd.Categorical.from_codes(
            self.values.to_numpy(),  # type: ignore
            categories,  # type: ignore
        )
        return pd.Series(data=categorical, name=self.name, copy=False)

    @override
    @classmethod
    def _from_pandas(cls, series: pd.Series, sw: SharedWorkspace) -> Self:
        values_numpy = series.cat.codes.to_numpy()
        values = SharedArray.from_numpy(values_numpy, sw, prefix="categorical")

        if len(series.cat.categories) < 10:
            categories: Categories | list[Any] = series.cat.categories.to_list()
        else:
            categories = Categories.from_pandas(series, sw)
        return cls(series.name, values, categories)

    @override
    @classmethod
    def _concat(cls, series: Iterable[pd.Series]) -> pd.Series:
        categoricals = [pd.Categorical(s) for s in series]
        return pd.Series(pd.api.types.union_categoricals(categoricals))


@dataclass
class SharedDataFrame:
    columns: list[SharedSeries]

    @property
    def shape(self) -> tuple[int, int]:
        column_count = len(self.columns)
        if column_count == 0:
            return 0, 0
        first_column = self.columns[0]
        (row_count,) = first_column.shape
        return row_count, column_count

    @property
    def allocation_names(self) -> set[str]:
        if len(self.columns) == 0:
            return set()
        return set.union(*(column.allocation_names for column in self.columns))

    def __getitem__(self, key: str) -> SharedSeries:
        for column in self.columns:
            if column.name == key:
                return column
        raise KeyError(key)

    def __delitem__(self, key: str) -> None:
        for i, column in enumerate(self.columns):
            if column.name == key:
                column.free()
                self.columns.pop(i)
                return
        raise KeyError(key)

    def free(self) -> None:
        for column in self.columns:
            column.free()

    def to_pandas(self) -> pd.DataFrame:
        series = [column.to_pandas() for column in self.columns]
        return pd.DataFrame(data={s.name: s for s in series}, copy=False)

    @classmethod
    def from_pandas(cls, data_frame: pd.DataFrame, sw: SharedWorkspace) -> Self:
        columns: list[SharedSeries] = list()
        for _, series in data_frame.items():
            columns.append(SharedSeries.from_pandas(series, sw))
        return cls(columns)


def concat(shared_data_frames: Iterable[SharedDataFrame]) -> SharedDataFrame:
    c = (shared_data_frame.columns for shared_data_frame in shared_data_frames)
    column_groups: list[tuple[SharedSeries, ...]] = list(zip(*c, strict=True))

    columns: list[SharedSeries] = list()
    for column_group in column_groups:
        if len(column_group) == 0:
            continue

        s = column_group[0]
        columns.append(s.concat(column_group))

    return SharedDataFrame(columns)
