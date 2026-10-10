"""Compatibility module for the embedding infrastructure adapter."""

import sys

from infrastructure import embeddings as _implementation

sys.modules[__name__] = _implementation
