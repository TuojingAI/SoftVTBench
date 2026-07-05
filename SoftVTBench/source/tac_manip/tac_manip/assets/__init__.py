"""Package containing asset and sensor configurations."""

import os

# Conveniences to other module directories via relative paths
ASSETS_EXT_DIR = os.path.abspath(os.path.dirname(__file__))
"""Path to the extension source directory."""

ASSETS_DATA_DIR = os.path.join(ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""


from .robots import *
from .sensors import *
