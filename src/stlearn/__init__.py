"""

stellar-learn is a library that provides pre-built models as well as an API to
build your own models to classify stellar curves.

"""
# relative imports
from . import io
from . import utils
from . import models
from . import conventions

# versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
