import sys
from pkg_resources import (
    DistributionNotFound as _DistributionNotFound,
    get_distribution as _get_distribution,
)

if sys.version_info < (3,7):
    raise Exception("MEGPy does not support Python < 3.7")

from .equilibrium import Equilibrium
from .localequilibrium import LocalEquilibrium
from .tracer import *


__all__= [
    "Equilibrium",
    "LocalEquilibrium"
]

try:
    _distribution = _get_distribution("megpy")
    __version__ = _distribution.version
except _DistributionNotFound:
    __version__ = "unknown"
