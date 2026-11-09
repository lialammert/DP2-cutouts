# conftest.py
# Anna Preto
# AstroDeep Project
# friendly n'est pas installé avec pip (son pyproject impose numpy < 2) : copaings.deps l'ajoute au chemin
# d'import à partir de FRIENDLY_PATH, avant tout test.

import os

os.environ.setdefault("FRIENDLY_PATH", os.path.expanduser("~/src/friendly"))
try:
    import copaings.deps  # noqa: F401
except ImportError:
    pass
