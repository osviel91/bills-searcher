"""Compatibility entrypoint for the REST application interface."""

import sys

from interfaces import rest as _implementation

sys.modules[__name__] = _implementation
