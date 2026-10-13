"""Compatibility executable/import for the worker interface adapter."""

import sys

from interfaces import worker as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
