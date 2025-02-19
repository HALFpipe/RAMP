import faulthandler
import multiprocessing as mp
import os
import signal
from functools import cache
from pprint import pformat
from subprocess import check_output
from typing import Sequence

num_threads_variables: Sequence[str] = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NPROC",
    "POLARS_MAX_THREADS",
]


def apply_num_threads(
    num_threads: int | None, enable_dump_traceback_later: bool = False
) -> None:
    from threadpoolctl import threadpool_info, threadpool_limits

    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if not faulthandler.is_enabled():
        raise RuntimeError("Could not enable faulthandler")
    if enable_dump_traceback_later:
        # Write a traceback to standard out every six hours
        faulthandler.dump_traceback_later(60 * 60 * 6, repeat=True)

    xla_flags = f"{os.getenv('XLA_FLAGS', '')} --xla_cpu_enable_fast_math=false"
    if num_threads is not None:
        threadpool_limits(limits=num_threads)
        for variable in num_threads_variables:
            os.environ[variable] = str(num_threads)
        xla_flags = (
            f"--xla_cpu_multi_thread_eigen={str(num_threads > 1).lower()} "
            f"intra_op_parallelism_threads={num_threads} "
            f"inter_op_parallelism_threads={num_threads} "
            f"{xla_flags}"
        )
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["XLA_FLAGS"] = xla_flags
    # os.environ["JAX_PLATFORMS"] = "cpu"

    from ..log import logger

    logger.debug(
        f'Configured process "{mp.current_process().name}" '
        f"with {pformat(threadpool_info())}"
    )


@cache
def cpu_count() -> int:
    return int(check_output(["nproc"]).decode().strip())
