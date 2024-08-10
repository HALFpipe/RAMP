from dataclasses import dataclass, field
from multiprocessing import TimeoutError
from multiprocessing.pool import ApplyResult

import numpy as np
from tqdm.auto import tqdm
from upath import UPath

from ..log import logger
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..tri.base import Triangular
from ..tri.tsqr import TallSkinnyQR
from ..utils import Pool, get_processes_and_num_threads, global_lock
from .base import Eigendecomposition, load_tri_arrays


def func(
    eig: Eigendecomposition,
    base_tri_array: Triangular,
    tri_array: SharedArray,
) -> None:
    a = base_tri_array.to_numpy()
    b = tri_array.to_numpy()

    sample_indices = base_tri_array.get_sample_indices(eig.samples)
    a.take(sample_indices, axis=0, out=b)

    eig.set_from_tri_array(tri_array)


def error_callback(exception: BaseException) -> None:
    raise exception


@dataclass
class EigendecompositionsCalc:
    chromosome: int | str

    base_tri_array: Triangular
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

            # Wait for one second
            for result in self.results.values():
                try:
                    result.wait(timeout=1)
                except TimeoutError:
                    pass
                break

    def run(self) -> list[Eigendecomposition]:
        base_tri_array = self.base_tri_array
        sw = base_tri_array.sw
        column_count = base_tri_array.sample_count
        count = len(self.samples_lists)

        # Prepare a list of arrays that can be moved while running
        sw.squash()
        can_squash: set[str] = set(sw.allocations.keys())
        can_squash.remove(base_tri_array.name)

        # Allocate output arrays
        variant_count = base_tri_array.variant_count
        eigendecompositions = [
            Eigendecomposition.empty(self.chromosome, samples, variant_count, sw)
            for samples in self.samples_lists
        ]

        # Calculate how many processes we can run in parallel
        average_sample_count = round(
            sum(len(samples) for samples in self.samples_lists) / count
        )
        average_size = average_sample_count * column_count * np.float64().itemsize
        capacity = sw.unallocated_size // average_size
        processes, num_threads_per_process = get_processes_and_num_threads(
            self.num_threads, count, capacity
        )

        logger.debug(f"Running {count} eigendecompositions in {processes} processes")
        pool = Pool(processes=processes, num_threads=num_threads_per_process)

        progress_bar = tqdm(
            desc="decomposing kinship matrices",
            unit="eigendecompositions",
            total=count,
            leave=False,
        )

        # Start with largest eigendecomposition first
        order = sorted(range(count), key=lambda i: -len(self.samples_lists[i]))
        with pool, progress_bar:
            while True:
                if len(order) > 0:  # There are more tasks to run
                    i = order[0]
                    samples = self.samples_lists[i]
                    eig = eigendecompositions[i]

                    try:
                        with global_lock:
                            name = Triangular.get_name(sw)
                            tri_array = sw.alloc(name, len(samples), column_count)
                        order.pop(0)  # Consume
                        logger.debug(f"Submitting task for eigendecomposition {i}")
                        self.results[i] = pool.apply_async(
                            func,
                            args=(eig, base_tri_array, tri_array),
                            error_callback=error_callback,
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
    *tri_paths: UPath,
    sw: SharedWorkspace,
    samples_lists: list[list[str]],
    chromosome: int | str,
    num_threads: int = 1,
) -> list[Eigendecomposition]:
    sw.squash()
    tri_arrays = load_tri_arrays(tri_paths, sw, num_threads=num_threads)
    base_tri_array = TallSkinnyQR.reduce(*tri_arrays)
    sw.squash()

    worker = EigendecompositionsCalc(
        chromosome, base_tri_array, samples_lists, num_threads=num_threads
    )
    eigendecompositions = worker.run()

    base_tri_array.free()
    sw.squash()

    return eigendecompositions
