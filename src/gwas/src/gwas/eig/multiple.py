from dataclasses import dataclass, field
from multiprocessing import TimeoutError
from multiprocessing.pool import ApplyResult
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from gwas.mem.arr import SharedFloat64Array

from ..log import logger
from ..mem.wkspace import SharedWorkspace
from ..tri.base import Triangular
from ..utils import Pool
from .base import Eigendecomposition, load_tri_arrays


def func(
    eig: Eigendecomposition,
    tri_arrays: list[Triangular],
    tri_array: SharedFloat64Array,
) -> None:
    _, column_count = tri_array.shape

    i = 0
    for tri in tri_arrays:
        a = tri.to_numpy()
        b = tri_array.to_numpy()[:, i : i + tri.sample_count]
        i += tri.sample_count

        sample_indices = tri.get_sample_indices(eig.samples)
        a.take(sample_indices, axis=0, out=b)

    assert i == column_count

    eig.set_from_tri_array(tri_array)


@dataclass
class EigendecompositionsCalc:
    chromosome: int | str

    tri_arrays: list[Triangular]
    samples_lists: list[list[str]]

    num_threads: int = 1

    results: dict[int, ApplyResult[None]] = field(init=False)

    def __post_init__(self) -> None:
        self.results = dict()

    def wait(self) -> set[int]:
        if len(self.results) == 0:
            return set()
        while True:
            ready = {i for i, result in self.results.items() if result.ready()}
            for i in ready:
                logger.debug(f"Finished task for eigendecomposition {i}")
                self.results.pop(i)
            if ready:
                return ready

            for result in self.results.values():
                try:
                    result.wait(timeout=1)
                    break
                except TimeoutError:
                    pass

    def run(self) -> list[Eigendecomposition]:
        tri_arrays = self.tri_arrays
        sw = tri_arrays[0].sw
        column_count = sum(tri.sample_count for tri in tri_arrays)
        count = len(self.samples_lists)

        # Start with largest eigendecomposition first
        order = sorted(range(count), key=lambda i: -len(self.samples_lists[i]))

        # Calculate how many processes we can run in parallel
        average_sample_count = round(
            sum(len(samples) for samples in self.samples_lists) / count
        )
        average_size = average_sample_count * column_count * np.float64().itemsize
        capacity = sw.unallocated_size // average_size
        processes = min((self.num_threads, count, capacity))
        pool = Pool(processes=processes, num_threads=self.num_threads // processes)

        # Allocate output arrays
        variant_count = sum(tri.variant_count for tri in tri_arrays)
        eigendecompositions = [
            Eigendecomposition.empty(self.chromosome, samples, variant_count, sw)
            for samples in self.samples_lists
        ]

        # Prepare for squashing
        can_squash: set[str] = set(sw.allocations.keys())
        for tri in tri_arrays:
            can_squash.remove(tri.name)

        progress_bar = tqdm(
            desc="decomposing kinship matrices",
            unit="eigendecompositions",
            leave=False,
        )

        with pool, progress_bar:
            while True:
                if len(order) > 0:  # There are more tasks to run
                    i = order[0]
                    samples = self.samples_lists[i]
                    eig = eigendecompositions[i]

                    try:
                        tri_array = sw.alloc(
                            Triangular.get_name(sw), len(samples), column_count
                        )
                        order.pop(0)  # Consume
                        logger.debug(f"Submitting task for eigendecomposition {i}")
                        self.results[i] = pool.apply_async(
                            func, args=(eig, tri_arrays, tri_array)
                        )
                        continue  # We can run another task
                    except MemoryError:
                        pass

                # Check completion
                finished = self.wait()
                progress_bar.update(len(finished))
                can_squash.update(eigendecompositions[i].name for i in finished)
                sw.squash(can_squash)

                if len(order) == 0:
                    if len(self.results) == 0:
                        break

        return eigendecompositions


def calc_eigendecompositions(
    *tri_paths: Path,
    sw: SharedWorkspace,
    samples_lists: list[list[str]],
    chromosome: int | str,
    num_threads: int = 1,
) -> list[Eigendecomposition]:
    tri_arrays = load_tri_arrays(tri_paths, sw, num_threads=num_threads)

    worker = EigendecompositionsCalc(
        chromosome, tri_arrays, samples_lists, num_threads=num_threads
    )
    eigendecompositions = worker.run()

    for tri in tri_arrays:
        tri.free()

    return eigendecompositions
