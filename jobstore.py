"""Compatibility module for the PostgreSQL job-store adapter."""

import sys

from infrastructure import jobstore as _implementation

sys.modules[__name__] = _implementation
