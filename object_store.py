"""Compatibility module for the blob-storage adapter."""

import sys

from infrastructure import object_store as _implementation

sys.modules[__name__] = _implementation
