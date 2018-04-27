import logging

from .version import version as __version__

from .simulator import Simulator

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass

__copyright__ = "2018, Applied Brain Research"
