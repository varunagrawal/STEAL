"""Module for help with learning of trajectory data."""

from steal.learning.base import *
from steal.learning.context_nets import *
from steal.learning.controllers import *
from steal.learning.loss import *
from steal.learning.taskmap_nets import *
from steal.learning.trainer import *


class Params(object):
    """Helper class to record various parameters."""
    def __init__(self, **kwargs):
        super(Params, self).__init__()
        self.__dict__.update(kwargs)
