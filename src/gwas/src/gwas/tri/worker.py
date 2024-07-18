# -*- coding: utf-8 -*-
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
        name: str | None = None,
    ) -> None:
        self.tsqr = tsqr
        self.tri_path = tri_path
        self.t = t

        if name is None:
            name = f"TriWorkerChr{tsqr.vcf_file.chromosome}"

        super().__init__(t.exception_queue, name=name)

    def func(self) -> None:
        logger.debug(f"Triangularizing chromosome {self.tsqr.vcf_file.chromosome}")
        tri = self.tsqr.map_reduce()

        tri.to_file(self.tri_path, num_threads=self.tsqr.num_threads)
        tri.free()
        # Indicate that we can start another task as this one has finished.
        self.t.can_run.set()
