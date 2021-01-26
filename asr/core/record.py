"""Implement RunSpec and RunRecord."""
import numpy as np

import typing
import copy
from .specification import RunSpecification
from .resources import Resources
from .results import get_object_matching_obj_id


# XXX: Change RunRecord name to Record
# XXX: Make MigrationLog object to store migration related info.
# XXX: Make Tags object.

class RunRecord:

    def __init__(  # noqa
            self,
            result: typing.Optional[typing.Any] = None,
            run_specification: typing.Optional[RunSpecification] = None,
            resources: typing.Optional[Resources] = None,
            dependencies: typing.Optional[typing.List[str]] = None,
            migration_id: typing.Optional[str] = None,
            migrated_from: typing.Optional[str] = None,
            migrated_to: typing.Optional[str] = None,
            tags: typing.Optional[typing.List[str]] = None,
            version: int = 0,
    ):
        assert type(run_specification) in [RunSpecification, type(None)]
        assert type(resources) in [Resources, type(None)]
        # XXX strictly enforce rest of types.

        data = dict(
            run_specification=run_specification,
            result=result,
            version=version,
            resources=resources,
            dependencies=dependencies,
            migration_id=migration_id,
            migrated_from=migrated_from,
            migrated_to=migrated_to,
            tags=tags,
        )
        self.__dict__.update(data)

    @property
    def parameters(self):  # noqa
        return self.run_specification.parameters

    @property
    def uid(self):  # noqa
        return self.run_specification.uid

    @property
    def name(self):  # noqa
        return self.run_specification.name

    def get_migrations(self):
        """Delegate migration to function objects."""
        from .migrate import Migration
        obj = get_object_matching_obj_id(self.run_specification.name)
        version = self.version
        if version != obj.version:
            migration = None
            visited_versions = set()
            while True:
                if version == obj.version:
                    break
                if version not in obj.migrations:
                    raise AssertionError(
                        'Record cannot be migrated to newest version. '
                        'Cannot migrate past version={version}'
                    )
                to_version, migration_func = obj.migrations[version]
                assert to_version not in visited_versions, \
                    'Circular migration detected.'
                visited_versions.add(to_version)
                if not migration:
                    migration = Migration(
                        migration_func,
                        from_version=version,
                        to_version=to_version,
                        record=self,
                        name=self.name + self.uid[:10],
                    )
                else:
                    migration = Migration(
                        migration_func,
                        from_version=self.version,
                        to_version=to_version,
                        dep=migration,
                        name=self.name + self.uid[:10],
                    )
                version = to_version

            return [
                Migration(
                    self.migrate,
                    from_version=self.version,
                    to_version=to_version,
                )
            ]

    def copy(self):
        data = copy.deepcopy(self.__dict__)
        return RunRecord(**data)

    def __str__(self):  # noqa
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

    def __repr__(self):  # noqa
        return self.__str__()

    def __eq__(self, other):  # noqa
        if not isinstance(other, RunRecord):
            return False

        return compare_dct_with_numpy_arrays(self.__dict__, other.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]


def compare_dct_with_numpy_arrays(dct1, dct2):
    """Compare dictionaries that might containt a numpy array.

    Numpy array are special since their equality test can return an
    array and meaning that we cannot just evaluate dct1 == dct2 since
    that would raise an error.

    """
    dct1keys = dct1.keys()
    dct2keys = dct2.keys()
    for key in dct1keys:
        if key not in dct2keys:
            return False
        value1 = dct1[key]
        value2 = dct2[key]

        if isinstance(value1, np.ndarray):
            if not np.array_equal(value1, value2):
                return False
        else:
            if not value1 == value2:
                return False
    return True
