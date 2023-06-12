# -*- coding: utf-8 -*-

from typing import NamedTuple


class SampleID(NamedTuple):
    fid: str
    iid: str

    def to_str(self, method: str = "merge_both_with_underscore") -> str:
        if method == "merge_both_with_underscore":
            return f"{self.fid}_{self.iid}"
        elif method == "iid":
            return self.iid
        elif method == "fid":
            return self.fid
        else:
            raise ValueError(f"Unknown method {method}")
