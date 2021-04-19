"""Top-level package for Atomic Simulation Recipes."""
from asr.core import (  # noqa
    command,
    option,
    argument,
    migration,
    Selector,
    comparators,
    initialize_root,
    find_root,
    atomsopt,
    calcopt,
    instruction,
)

cmp = comparators
__author__ = """Morten Niklas Gjerding"""
__email__ = 'mortengjerding@gmail.com'
__version__ = '0.4.1'
name = "asr"
