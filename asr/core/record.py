"""Implement RunSpec and RunRecord."""
import typing
import copy
from .specification import RunSpecification
from .resources import Resources
from .utils import make_property
from .results import get_object_matching_obj_id


class RunRecord:  # noqa

    record_version: int = 0
    result = make_property('result')
    side_effects = make_property('side_effects')
    dependencies = make_property('dependencies')
    run_specification = make_property('run_specification')
    resources = make_property('resources')
    migrated_from = make_property('migrated_from')
    migrated_to = make_property('migrated_to')
    migration_id = make_property('migration_id')
    tags = make_property('tags')

    def __init__(  # noqa
            self,
            result: typing.Any,
            run_specification: RunSpecification = None,
            resources: Resources = None,
            side_effects: dict = None,
            dependencies: typing.List[str] = None,
            migration_id: str = None,
            migrated_from: str = None,
            migrated_to: str = None,
            tags: typing.List[str] = None,
    ):
        if resources is None:
            resources = Resources()

        assert type(run_specification) == RunSpecification
        assert type(resources) == Resources
        # XXX strictly enforce rest of types.
        self.data = dict(
            run_specification=run_specification,
            result=result,
            resources=resources,
            side_effects=side_effects,
            dependencies=dependencies,
            migration_id=migration_id,
            migrated_from=migrated_from,
            migrated_to=migrated_to,
            tags=tags,
        )

    @property
    def parameters(self):  # noqa
        return self.data['run_specification'].parameters

    @property
    def uid(self):  # noqa
        return self.data['run_specification'].uid

    @property
    def name(self):  # noqa
        return self.data['run_specification'].name

    def get_migrations(self, cache):
        """Delegate migration to function objects."""
        obj = get_object_matching_obj_id(self.run_specification.name)
        if obj.migrations:
            return obj.migrations(cache)

    def copy(self):
        data = copy.deepcopy(self.data)
        return RunRecord(**data)

    def __str__(self):  # noqa
        strings = []
        for name, value in self.data.items():
            if name == 'result':
                txt = str(value)
                if len(txt) > 30:
                    strings.append('result=' + str(value)[:30] + '...')
                    continue
            strings.append(f'{name}={value}')
        return 'RunRecord(' + ', '.join(strings) + ')'

    def __repr__(self):  # noqa
        return self.__str__()

    def __getstate__(self):  # noqa
        return self.__dict__

    def __setstate__(self, state):  # noqa
        self.__dict__.update(state)

    def __hash__(self):  # noqa
        return hash(str(self.run_specification))

    def __getattr__(self, attr):  # noqa
        if attr in self.data:
            return self.data[attr]
        raise AttributeError

    def __eq__(self, other):  # noqa
        if not isinstance(other, RunRecord):
            return False
        return hash(self) == hash(other)
