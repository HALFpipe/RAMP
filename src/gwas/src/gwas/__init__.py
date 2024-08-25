from importlib.metadata import PackageNotFoundError, version

from .compression.pipe import CompressedBytesReader as CompressedBytesReader
from .compression.pipe import CompressedBytesWriter as CompressedBytesWriter
from .compression.pipe import CompressedTextReader as CompressedTextReader
from .compression.pipe import CompressedTextWriter as CompressedTextWriter
from .eig.calc import calc_eigendecompositions as calc_eigendecompositions
from .null_model.calc import calc_null_model_collections as calc_null_model_collections
from .plink import BimFile as BimFile
from .plink import FamFile as FamFile
from .plink import PsamFile as PsamFile
from .plink import PVarFile as PVarFile
from .tri.calc import calc_tri as calc_tri
from .utils.genetics import chromosome_from_int as chromosome_from_int
from .utils.genetics import chromosome_to_int as chromosome_to_int
from .utils.genetics import chromosomes_list as chromosomes_list
from .utils.genetics import chromosomes_set as chromosomes_set
from .vcf.base import VCFFile as VCFFile
from .vcf.cpp import CppVCFFile as CppVCFFile
from .vcf.python import PyVCFFile as PyVCFFile

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
