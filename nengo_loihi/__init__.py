import logging

from .version import check_nengo_version
from .version import version as __version__

check_nengo_version()
del check_nengo_version

# Import builders so they are registered
from . import builder

# Import into top-level namespace
from .config import BlockShape, add_params, set_defaults
from .neurons import LoihiLIF, LoihiSpikingRectifiedLinear
from .simulator import Simulator

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:  # pragma: no cover
    pass

__copyright__ = "2018-2021, Applied Brain Research"
