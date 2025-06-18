import sys
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python <3.8, use the backport
    from importlib_metadata import version, PackageNotFoundError

if sys.version_info < (3,8):
    raise Exception("megpy does not support Python < 3.8")

from .equilibrium import Equilibrium
from .localequilibrium import LocalEquilibrium
from .tracer import *


__all__= [
    "Equilibrium",
    "LocalEquilibrium"
]

try:
    __version__ = version('megpy')
except PackageNotFoundError:
    __version__ = 'unknown'