import logging

from .version import version as __version__

from .simulator import Simulator
from .config import add_params, set_defaults

# Import builders so they are registered
from . import builder

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:  # pragma: no cover
    pass

__copyright__ = "2018, Applied Brain Research"
