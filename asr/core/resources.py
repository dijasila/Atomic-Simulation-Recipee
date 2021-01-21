from .specification import RunSpecification
from .utils import make_property
import time
import contextlib


class Resources:  # noqa

    def __init__(  # noqa
            self,
            execution_start: float = None,
            execution_end: float = None,
            execution_duration: float = None,
            ncores: int = None
    ):
        self.data = dict(
            execution_start=execution_start,
            execution_end=execution_end,
            execution_duration=execution_duration,
            ncores=ncores)

    execution_start = make_property('execution_start')
    execution_end = make_property('execution_end')
    execution_duration = make_property('execution_duration')
    ncores = make_property('ncores')

    def __str__(self):
        return (f'Resources(time={self.execution_duration:.1f}s, '
                f'ncores={self.ncores})')


@contextlib.contextmanager
def _register_resources(run_specification: RunSpecification):
    from ase.parallel import world
    execution_start = time.time()
    resources = Resources()
    yield resources
    execution_end = time.time()
    resources.execution_start = execution_start
    resources.execution_end = execution_end
    resources.execution_duration = execution_end - execution_start
    resources.ncores = world.size


def register_resources():  # noqa
    def wrapper(func):
        def wrapped(run_specification):
            with _register_resources(run_specification) as resources:
                run_record = func(run_specification)
            run_record.resources = resources
            return run_record
        return wrapped
    return wrapper
