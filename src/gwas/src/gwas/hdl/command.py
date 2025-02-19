from argparse import Namespace
from operator import attrgetter

from upath import UPath

from ..log import logger
from ..mem.wkspace import SharedWorkspace
from ..utils.jax import setup_jax
from .calc import HDL
from .load import load
from .ml import eig_count as eig_count

suffix = ".sumstats.gz"


def hdl(sw: SharedWorkspace, arguments: Namespace) -> None:
    ld_path: UPath = arguments.ld_path
    input_path: UPath = arguments.input_path

    setup_jax()

    sumstats_paths = sorted(
        input_path.rglob(f"*genomic-sem{suffix}"), key=attrgetter("stem")
    )
    phenotypes = [sumstats_path.name.split(".")[0] for sumstats_path in sumstats_paths]

    data = load(sw, ld_path, sumstats_paths, arguments.num_threads)
    logger.debug(f"Loaded {len(sumstats_paths)} sumstats files")
    hdl = HDL(phenotypes, data, arguments.output_path, arguments.num_threads)

    hdl.calc_piecewise()
    hdl.calc_jackknife()
