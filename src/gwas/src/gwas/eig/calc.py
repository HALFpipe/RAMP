from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm
from upath import UPath

from ..log import logger
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..tri.base import Triangular
from ..tri.tsqr import TallSkinnyQR
from ..utils import (
    Process,
    SharedState,
    get_global_lock,
    get_processes_and_num_threads,
    wait,
)
from .base import Eigendecomposition, load_tri_arrays


@dataclass
class TaskSyncCollection(SharedState):
    pass


class EigendecompositionWorker(Process):
    def __init__(
        self,
        eig: Eigendecomposition,
        base_tri_array: Triangular,
        tri_array: SharedArray,
        t: TaskSyncCollection,
        num_threads: int,
        name: str | None = None,
    ) -> None:
        if name is None:
            k = int(eig.name.split("-")[-1])
            name = f"EigWorker-{k}"

        super().__init__(t.exception_queue, num_threads=num_threads, name=name)

        self.eig = eig
        self.base_tri_array = base_tri_array
        self.tri_array = tri_array

    def func(self) -> None:
        eig = self.eig
        base_tri_array = self.base_tri_array
        tri_array = self.tri_array

        a = base_tri_array.to_numpy()
        b = tri_array.to_numpy()

        samples = eig.samples
        sample_indices = base_tri_array.get_sample_indices(samples)
        a.take(sample_indices, axis=0, out=b)

        eig.set_from_tri_array(tri_array)


@dataclass
class EigendecompositionsCalc:
    chromosome: int | str

    base_tri_array: Triangular
    samples_lists: list[list[str]]

    num_threads: int = 1

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

        # Start with largest eigendecomposition first
        order = sorted(range(count), key=lambda i: -len(self.samples_lists[i]))

        t = TaskSyncCollection()
        running: list[EigendecompositionWorker] = list()
        with tqdm(
            desc="decomposing kinship matrices",
            unit="eigendecompositions",
            total=count,
            leave=False,
        ) as progress_bar:
            while True:
                # Check if an error has occurred
                if not t.exception_queue.empty():
                    raise t.exception_queue.get()

                # Sleep for one second if a process is running
                wait(running)

                # Update progress bar with the number of processes that have finished
                progress_bar.update(sum(1 for proc in running if not proc.is_alive()))

                # Update list of running processes
                if any(not proc.is_alive() for proc in running):
                    running = [proc for proc in running if proc.is_alive()]

                    can_squash.update(
                        proc.eig.name for proc in running if not proc.is_alive()
                    )
                    sw.squash(can_squash)

                # Check if we can exit
                if len(running) == 0 and len(order) == 0:
                    # All tasks have been completed
                    break

                if len(order) == 0:
                    # No more tasks to run
                    continue

                if len(running) >= processes:
                    # We are at capacity
                    continue

                i = order[0]
                samples = self.samples_lists[i]
                eig = eigendecompositions[i]

                try:
                    with get_global_lock():
                        name = Triangular.get_name(sw)
                        tri_array = sw.alloc(name, len(samples), column_count)
                except MemoryError:
                    continue

                order.pop(0)  # Consume
                logger.debug(
                    f"Starting process for eigendecomposition {i} "
                    f'with tri array "{name}"'
                )

                proc = EigendecompositionWorker(
                    eig,
                    base_tri_array,
                    tri_array,
                    t,
                    num_threads=num_threads_per_process,
                )
                proc.start()
                running.append(proc)

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
