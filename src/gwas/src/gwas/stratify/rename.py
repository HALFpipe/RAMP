import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
from numpy import typing as npt
from upath import UPath

from .base import SampleID

DType = TypeVar("DType", bound=np.number, covariant=True)


@dataclass
class SampleRenamer:
    arguments: Namespace

    removed_samples: set[str] = field(default_factory=set)
    sample_mapping: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        header: list[str] | None = None
        for from_column, to_column, path in self.arguments.rename_samples_from_file:
            path = UPath(path)
            with path.open("rt") as file_handle:
                for line in file_handle:
                    header = line.strip().split(",")
                    break

            if header is None:
                raise ValueError(f'No header found for "{path}"')
            from_index = header.index(from_column)
            to_index = header.index(to_column)

            with path.open("rt") as file_handle:
                for line in file_handle:
                    tokens = line.strip().split(",")
                    from_token = self.rename_sample(tokens[from_index])
                    self.sample_mapping[from_token] = tokens[to_index]

    def rename_sample(self, sample: str) -> str:
        sample = sample.strip()
        for old, new in self.arguments.rename_samples_replace:
            sample = re.sub(old, new, sample)
        if self.arguments.match_case_insensitive:
            sample = sample.lower()
        if self.arguments.match_alphanumeric_only:
            sample = "".join(filter(str.isalnum, sample))
        sample = self.sample_mapping.get(sample, sample)
        return sample

    def rename_samples(
        self,
        samples: list[SampleID],
        array_samples: list[str],
        array: npt.NDArray[DType],
    ) -> tuple[list[SampleID], npt.NDArray[DType]]:
        iid = [self.rename_sample(sample.iid) for sample in samples]
        underscored = [self.rename_sample(f"{fid}_{iid}") for fid, iid in samples]
        sample_indices = np.full(len(array_samples), -1, dtype=int)
        for i, sample in enumerate(array_samples):
            sample = self.rename_sample(sample)
            sample = self.rename_sample(sample)  # Run again to apply normalization.
            if sample in iid:
                sample_indices[i] = iid.index(sample)
            elif sample in underscored:
                sample_indices[i] = underscored.index(sample)

        self.removed_samples |= {
            c for c, i in zip(array_samples, sample_indices, strict=True) if i == -1
        }
        array_samples_new = [samples[i] for i in sample_indices if i >= 0]
        array = array[sample_indices >= 0, :]

        return array_samples_new, array
