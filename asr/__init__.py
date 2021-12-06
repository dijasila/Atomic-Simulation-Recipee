"""Top-level package for Atomic Simulation Recipes."""
from asr.core import (  # noqa
    Dependencies,
    Metadata,
    NonMigratableRecord,
    Record,
    Resources,
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
    migration,
    option,
    ASRResult,
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
    'migration',
    'Selector',
    'matchers',
    'atomsopt',
    'calcopt',
    'instruction',
]
