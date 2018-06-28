import logging

from .version import version as __version__

# manually import modules in correct order for Cython
from . import loihi_api
from . import loihi_cx
from . import builder
from . import allocators
from . import loihi_interface

from .simulator import Simulator

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass

__copyright__ = "2018, Applied Brain Research"
