# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from ..log import logger
from ..utils import Process
from .base import TaskSyncCollection
from .tsqr import TallSkinnyQR


class TriWorker(Process):
    def __init__(
        self,
        tsqr: TallSkinnyQR,
        tri_path: Path,
        t: TaskSyncCollection,
        *args,
        **kwargs,
    ) -> None:
        self.tsqr = tsqr
        self.tri_path = tri_path
        self.t = t

        if "name" not in kwargs:
            kwargs["name"] = f"tri-worker-chr{tsqr.vcf_file.chromosome}"

        super().__init__(t.exception_queue, *args, **kwargs)

    def func(self) -> None:
        logger.debug(f"Triangularizing chromosome {self.tsqr.vcf_file.chromosome}")
        tri = self.tsqr.map_reduce()

        if tri is None:
            vcf_file = self.tsqr.vcf_file
            raise ValueError(f"Could not triangularize {vcf_file.file_path}")

        tri.to_file(self.tri_path)
        tri.free()
        # Indicate that we can start another task as this one has finished.
        self.t.can_run.set()
