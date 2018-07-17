import logging

from .version import version as __version__

from . import cx
from . import builder
from .simulator import Simulator
from .config import add_params, set_defaults

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass

__copyright__ = "2018, Applied Brain Research"
