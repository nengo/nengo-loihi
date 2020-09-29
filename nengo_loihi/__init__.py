import logging

from .version import check_nengo_version, version as __version__

check_nengo_version()
del check_nengo_version

from .simulator import Simulator
from .config import add_params, BlockShape, set_defaults
from .neurons import LoihiLIF, LoihiSpikingRectifiedLinear

# Import builders so they are registered
from . import builder

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:  # pragma: no cover
    pass

__copyright__ = "2018, Applied Brain Research"
