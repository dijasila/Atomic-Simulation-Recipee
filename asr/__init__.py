"""Top-level package for Atomic Simulation Recipes."""
import sys
from htwutil import *  # XXX

from asr.core import (  # noqa
    ASRResult,
    Dependencies,
    Metadata,
    Mutation,
    NonMigratableRecord,
    Parameters,
    Record,
    Resources,
    Revision,
    RevisionHistory,
    RunSpecification,
    Selector,
    argument,
    atomsopt,
    calcopt,
    command,
    comparators,
    get_cache,
    instruction,
    mutation,
    option,
    prepare_result,
)

matchers = comparators
__author__ = """Morten Niklas Gjerding"""
__email__ = 'mortengjerding@gmail.com'
__version__ = '0.4.1'
name = "asr"

__all__ = [
    'command',
    'option',
    'argument',
    'mutation',
    'Selector',
    'matchers',
    'atomsopt',
    'calcopt',
    'instruction',
]
