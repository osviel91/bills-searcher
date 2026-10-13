"""Compatibility module for the Elasticsearch adapter."""

import sys

from infrastructure import storage as _implementation

sys.modules[__name__] = _implementation
