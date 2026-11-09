# conftest.py
# Anna Preto
# AstroDeep Project
# friendly is not installed with pip (its pyproject pins numpy < 2): copaings.deps adds it to the import path
# from FRIENDLY_PATH, before any test.

import os

os.environ.setdefault("FRIENDLY_PATH", os.path.expanduser("~/src/friendly"))
try:
    import copaings.deps  # noqa: F401
except ImportError:
    pass
