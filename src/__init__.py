"""Legacy checkout imports. New code should import the installed catalogiq package.

The exported brand list is retained for historical analyses, not label assignment.
"""
from .paths.paths import (
    ROOT_PATH, TRAIN_PATH, TARGET_PATH,
    PROVIDED_DATA_PATH, DATA_PATH, DOCS_PATH,
    NOTEBOOKS_PATH
)

from .whitelist import (
    jj_whitelist
)