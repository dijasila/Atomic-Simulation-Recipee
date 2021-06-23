"""Implement RunSpec and Record."""
from __future__ import annotations
import numpy as np

import typing
import copy
from dataclasses import dataclass
from .specification import RunSpecification
from .resources import Resources
from .metadata import Metadata
from .history import History

# XXX: Make Tags object.


@dataclass
class Record:

    result: typing.Optional[typing.Any] = None
    run_specification: typing.Optional[RunSpecification] = None
    resources: typing.Optional[Resources] = None
    dependencies: typing.Optional[typing.List[str]] = None
    history: typing.Optional[History] = None
    tags: typing.Optional[typing.List[str]] = None
    metadata: typing.Optional[Metadata] = None

    @property
    def parameters(self):
        return self.run_specification.parameters

    @parameters.setter
    def parameters(self, value):
        self.run_specification.parameters = value

    @property
    def uid(self):
        return self.run_specification.uid

    @property
    def version(self):
        return self.run_specification.version

    @version.setter
    def version(self, value):
        self.run_specification.version = value

    @property
    def revision(self):
        if not self.history:
            return None
        return self.history.latest_revision.uid

    @property
    def name(self):
        return self.run_specification.name

    def copy(self):
        data = copy.deepcopy(self.__dict__)
        return Record(**data)

    def __repr__(self):
        strings = []
        for name, value in self.__dict__.items():
            if name == 'result':
                txt = str(value)
                if len(txt) > 30:
                    strings.append('result=' + str(value)[:30] + '...')
                    continue
            if value is not None:
                strings.append('='.join([str(name), str(value)]))
        return 'Record(' + ', '.join(strings) + ')'

    def __eq__(self, other):
        if not isinstance(other, Record):
            return False

        return compare_dct_with_numpy_arrays(self.__dict__, other.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]


def compare_dct_with_numpy_arrays(dct1, dct2):
    """Compare dictionaries that might contain a numpy array.

    Numpy arrays are special since their equality test can return an
    array and meaning that we cannot just evaluate dct1 == dct2 since
    that would raise an error.

    """
    if set(dct1) != set(dct2):
        return False

    for key in dct1:
        value1 = dct1[key]
        value2 = dct2[key]

        if isinstance(value1, np.ndarray) or isinstance(value2, np.ndarray):
            if not np.array_equal(value1, value2):
                return False
        else:
            if not value1 == value2:
                return False
    return True
