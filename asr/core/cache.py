"""Implement cache functionality."""
import abc
import os
import pathlib
import functools
import numpy as np
import fnmatch
import uuid
from .params import Parameters
from .record import RunRecord
from .specification import RunSpecification
from .utils import write_file
from .serialize import Serializer, JSONSerializer
from ase import Atoms


class RunSpecificationAlreadyExists(Exception):  # noqa
    pass


class AbstractCache(abc.ABC):
    """Abstract cache interface."""

    @abc.abstractmethod
    def add(self, run_record: RunRecord):  # noqa
        pass

    @abc.abstractmethod
    def get(self, run_record: RunRecord):  # noqa
        pass

    @abc.abstractmethod
    def has(self, run_specification: RunSpecification):  # noqa
        pass


class NoCache(AbstractCache):  # noqa

    def add(self, run_record: RunRecord):
        """Add record of run to cache."""
        ...

    def has(self, run_specification: RunSpecification) -> bool:
        """Has run record matching run specification."""
        return False

    def get(self, run_specification: RunSpecification):
        """Get run record matching run specification."""
        ...


class SingleRunFileCache(AbstractCache):  # noqa

    def __init__(self, serializer: Serializer = JSONSerializer()):  # noqa
        self.serializer = serializer
        self._cache_dir = None
        self.depth = 0

    @property
    def cache_dir(self) -> pathlib.Path:  # noqa
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value):
        self._cache_dir = value

    @staticmethod
    def _name_to_results_filename(name: str):
        name = name.replace('::', '@').replace('@main', '')
        return f'results-{name}.json'

    def add(self, run_record: RunRecord):  # noqa
        if self.has(run_record.run_specification):
            raise RunSpecificationAlreadyExists(
                'You are using the SingleRunFileCache which does not'
                'support multiple runs of the same function. '
                'Please specify another cache.'
            )
        name = run_record.run_specification.name
        filename = self._name_to_results_filename(name)
        serialized_object = self.serializer.serialize(run_record)
        self._write_file(filename, serialized_object)
        return filename

    def has(self, run_specification: RunSpecification):  # noqa
        name = run_specification.name
        filename = self._name_to_results_filename(name)
        return (self.cache_dir / filename).is_file()

    def get(self, run_specification: RunSpecification):  # noqa
        name = run_specification.name
        filename = self._name_to_results_filename(name)
        serialized_object = self._read_file(filename)
        obj = self.serializer.deserialize(serialized_object)
        return obj

    def select(self):  # noqa
        pattern = self._name_to_results_filename('*')
        paths = list(pathlib.Path(self.cache_dir).glob(pattern))
        serialized_objects = [self._read_file(path) for path in paths]
        deserialized_objects = [self.serializer.deserialize(ser_obj)
                                for ser_obj in serialized_objects]
        return deserialized_objects

    def _write_file(self, filename: str, text: str):
        write_file(self.cache_dir / filename, text)

    def _read_file(self, filename: str) -> str:
        serialized_object = pathlib.Path(self.cache_dir / filename).read_text()
        return serialized_object

    def __enter__(self):
        """Enter context manager."""
        if self.depth == 0:
            self.cache_dir = pathlib.Path('.').absolute()
        self.depth += 1
        return self

    def __exit__(self, type, value, traceback):
        """Exit context manager."""
        self.depth -= 1
        if self.depth == 0:
            self.cache_dir = None

    def __call__(self):  # noqa

        def wrapper(func):
            def wrapped(run_specification):
                with self:
                    if self.has(run_specification):
                        run_record = self.get(run_specification)
                    else:
                        run_record = func(run_specification)
                        self.add(run_record)
                return run_record
            return wrapped
        return wrapper


def find_results_files():
    relevant_patterns = [
        'results-asr.*.json',
        'displacements*/*/results-*.json',
        'strains*/results-*.json',
    ]

    skip_patterns = [
        'results-asr.database.fromtree.json',
        'results-asr.database.app.json',
        'results-asr.database.key_descriptions.json',
        'displacements*/*/results-asr.database.material_fingerprint.json',
        'strains*/results-asr.database.material_fingerprint.json',
        '*asr.setup.params.json',
        '*asr.setup.params.json',
    ]

    for pattern in relevant_patterns:
        for path in pathlib.Path().glob(pattern):
            filepath = str(path)
            if any(fnmatch.fnmatch(filepath, skip_pattern)
                   for skip_pattern in skip_patterns):
                continue
            yield path


def construct_record_from_resultsfile(path):
    from asr.core.results import MetaDataNotSetError
    from asr.core import read_json, get_recipe_from_name, ASRResult
    from ase.io import read
    result = read_json(path)
    recipename = path.with_suffix('').name.split('-')[1]

    if isinstance(result, ASRResult):
        data = result.data
        metadata = result.metadata.todict()
    else:
        assert isinstance(result, dict)
        if recipename == 'asr.gs@calculate':
            from asr.gs import GroundStateCalculationResult
            from asr.calculators import Calculation
            calculation = Calculation(
                id='gs',
                cls_name='gpaw',
                paths=['gs.gpw'],
            )
            result = GroundStateCalculationResult.fromdata(
                calculation=calculation)
            data = result.data
            metadata = {'asr_name': recipename}
        else:
            raise AssertionError(f'Unparsable old results file: path={path}')

    recipe = get_recipe_from_name(recipename)
    result = recipe.returns(
        data=data,
        metadata=metadata,
        strict=False)

    atoms = read(path.parent / 'structure.json')
    try:
        parameters = result.metadata.params
    except MetaDataNotSetError:
        parameters = {}

    parameters = Parameters(parameters)
    if 'atoms' not in parameters:
        parameters.atoms = atoms.copy()

    try:
        code_versions = result.metadata.code_versions
    except MetaDataNotSetError:
        code_versions = {}

    from asr.core.resources import Resources
    try:
        resources = result.metadata.resources
        resources = Resources(
            execution_start=resources.get('tstart'),
            execution_end=resources.get('tend'),
            execution_duration=resources['time'],
            ncores=resources['ncores'],
        )
    except MetaDataNotSetError:
        resources = Resources()

    name = result.metadata.asr_name
    if '@' not in name:
        name += '::main'
    else:
        name = name.replace('@', '::')

    return RunRecord(
        run_specification=RunSpecification(
            name=name,
            parameters=parameters,
            version=-1,
            codes=code_versions,
            uid=uuid.uuid4().hex,
        ),
        resources=resources,
        result=result,
    )


class LegacyFileSystemBackend:

    def migrate(self):
        pass

    def add(self, run_record: RunRecord):  # noqa
        raise NotImplementedError

    @property
    def uid_table(self):  # noqa
        table = {}
        for pattern in self.relevant_patterns:
            for path in pathlib.Path().glob(pattern):
                filepath = str(path)
                if any(fnmatch.fnmatch(filepath, skip_pattern)
                       for skip_pattern in self.skip_patterns):
                    continue
                table[str(path)] = str(path)
        return table

    def has(self, selection: 'Selection'):  # noqa
        records = self.select()
        for record in records:
            if selection.matches(record):
                return True
        return False

    def get_result_from_uid(self, uid):  # noqa: D102
        from asr.core import read_json
        filename = self.uid_table[uid]
        result = read_json(filename)
        return result

    def get_record_from_uid(self, uid):  # noqa
        from asr.core.results import MetaDataNotSetError
        result = self.get_result_from_uid(uid)
        try:
            parameters = result.metadata.params
        except MetaDataNotSetError:
            parameters = {}
        try:
            code_versions = result.metadata.code_versions
        except MetaDataNotSetError:
            code_versions = {}

        name = result.metadata.asr_name

        if '@' not in name:
            name += '::main'
        else:
            name = name.replace('@', '::')

        return RunRecord(
            run_specification=RunSpecification(
                name=name,
                parameters=parameters,
                version=-1,
                codes=code_versions,
                uid=uuid.uuid4().hex,
            ),
            result=result,
        )

    def select(self, selection: 'Selection' = None):
        all_records = [self.get_record_from_uid(uid)
                       for uid in self.uid_table]
        if selection is None:
            return all_records
        selected = []
        for record in all_records:
            if selection.matches(record):
                selected.append(record)
        return selected


class FileCacheBackend():  # noqa

    def migrate(self):
        for resultsfile in find_results_files():
            record = construct_record_from_resultsfile(resultsfile)
            migrated_record = record.migrate()
            if migrated_record:
                print('-#' * 20, 'found migrated_record', migrated_record)
                record = migrated_record
            selection = {
                'run_specification.name':
                record.run_specification.name,
                'run_specification.parameters':
                record.run_specification.parameters,
            }
            sel = Selection(**selection)
            if not self.has(sel):
                self.add(record)

    def __init__(
            self,
            cache_dir: str = '.asr/records',
            serializer: Serializer = JSONSerializer(),
    ):
        self.serializer = serializer
        self.cache_dir = pathlib.Path(cache_dir)
        self.filename = 'run-data.json'

    @staticmethod
    def _name_to_results_filename(name: str):
        return f'results-{name}.json'

    def add(self, run_record: RunRecord):  # noqa
        run_specification = run_record.run_specification
        run_uid = run_specification.uid
        name = run_record.run_specification.name + '-' + run_uid[:10]
        filename = self._name_to_results_filename(name)
        serialized_object = self.serializer.serialize(run_record)
        self._write_file(filename, serialized_object)
        self.add_uid_to_table(run_uid, filename)
        return run_uid

    @property
    def initialized(self):  # noqa
        return (self.cache_dir / pathlib.Path(self.filename)).is_file()

    def initialize(self):  # noqa
        assert not self.initialized
        serialized_object = self.serializer.serialize({})
        self._write_file(self.filename, serialized_object)

    def add_uid_to_table(self, run_uid, filename):  # noqa
        uid_table = self.uid_table
        uid_table[run_uid] = filename
        self._write_file(
            self.filename,
            self.serializer.serialize(uid_table)
        )

    @property
    def uid_table(self):  # noqa
        if not self.initialized:
            self.initialize()
        text = self._read_file(self.filename)
        uid_table = self.serializer.deserialize(text)
        return uid_table

    def has(self, selection: 'Selection'):  # noqa
        records = self.select()
        for record in records:
            if selection.matches(record):
                return True
        return False

    def get_record_from_uid(self, run_uid):  # noqa
        filename = self.uid_table[run_uid]
        serialized_object = self._read_file(filename)
        obj = self.serializer.deserialize(serialized_object)
        return obj

    def select(self, selection: 'Selection' = None):
        all_records = [self.get_record_from_uid(run_uid)
                       for run_uid in self.uid_table]
        if selection is None:
            return all_records
        selected = []
        for record in all_records:
            if selection.matches(record):
                selected.append(record)
        return selected

    def _write_file(self, filename: str, text: str):
        if not self.cache_dir.is_dir():
            os.makedirs(self.cache_dir)
        write_file(self.cache_dir / filename, text)

    def _read_file(self, filename: str) -> str:
        serialized_object = pathlib.Path(self.cache_dir / filename).read_text()
        return serialized_object


class NoSuchAttribute(Exception):

    pass


def flatten_list(lst):
    flatlist = []
    for value in lst:
        if isinstance(value, list):
            flatlist.extend(flatten_list(value))
        else:
            flatlist.append(value)
    return flatlist


def compare_lists(lst1, lst2):
    flatlst1 = flatten_list(lst1)
    flatlst2 = flatten_list(lst2)
    if not len(flatlst1) == len(flatlst2):
        return False
    for value1, value2 in zip(flatlst1, flatlst2):
        if not value1 == value2:
            return False
    return True


def compare_ndarrays(array1, array2):
    lst1 = array1.tolist()
    lst2 = array2.tolist()
    return compare_lists(lst1, lst2)


def approx(value1, rtol=1e-3):

    def wrapped_approx(value2):
        return np.isclose(value1, value2, rtol=rtol)

    return wrapped_approx


def compare_atoms(atoms1, atoms2):
    dct1 = atoms1.todict()
    dct2 = atoms2.todict()
    keys = [
        'numbers',
        'positions',
        'cell',
        'pbc',
    ]
    for key in keys:
        if not np.allclose(dct1[key], dct2[key]):
            return False
    return True


def allways_match(value):
    return True


class Selection:

    def __init__(self, **selection):
        self.selection = self.normalize_selection(selection)

    def do_not_compare(self, x):
        return True

    def normalize_selection(self, selection: dict):
        normalized = {}

        for key, value in selection.items():
            comparator = None
            if isinstance(value, dict):
                norm = self.normalize_selection(value)
                for keynorm, valuenorm in norm.items():
                    normalized['.'.join([key, keynorm])] = valuenorm
            elif value is None:
                pass
            elif isinstance(value, Atoms):
                comparator = functools.partial(compare_atoms, value)
            elif isinstance(value, np.ndarray):
                comparator = functools.partial(
                    compare_ndarrays,
                    value,
                )
            elif isinstance(value, (list, tuple)):
                comparator = functools.partial(
                    compare_lists,
                    value,
                )
            elif type(value) in {str, bool, int}:
                comparator = value.__eq__
            elif type(value) is float:
                comparator = approx(value)
            elif hasattr(value, '__dict__'):
                norm = self.normalize_selection(value.__dict__)
                for keynorm, valuenorm in norm.items():
                    normalized['.'.join([key, keynorm])] = valuenorm
            else:
                raise AssertionError(f'Unknown type {type(value)}')

            if comparator is not None:
                normalized[key] = comparator
        return normalized

    def get_attribute(self, obj, attrs):

        if not attrs:
            return obj

        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            elif attr in obj:
                obj = obj[attr]
            else:
                raise NoSuchAttribute

        return obj

    def matches(self, obj):

        for attr, comparator in self.selection.items():
            try:
                objvalue = self.get_attribute(obj, attr.split('.'))
            except NoSuchAttribute:
                return False
            if not comparator(objvalue):
                return False
        return True

    def __str__(self):  # noqa
        return str(self.selection)


class Cache:  # noqa

    def migrate(self):
        """Migrate cache data."""
        self.backend.migrate()

    def __init__(self, backend):
        self.backend = backend

    def add(self, run_record: RunRecord):  # noqa
        if self.has(run_specification=run_record.run_specification):
            raise RunSpecificationAlreadyExists(
                'This Run specification already exists in cache.'
            )

        self.backend.add(run_record)

    def has(self, **selection):  # noqa
        selection = Selection(
            **selection,
        )
        return self.backend.has(selection)

    def get(self, **selection):  # noqa
        assert self.has(**selection), \
            'No matching run_specification.'
        records = self.select(**selection)
        assert len(records) == 1, f'More than one record matched! records={records}'
        return records[0]

    def select(self, **selection):  # noqa
        """Select records.

        Selection can be in the style of

        cache.select(uid=uid)
        cache.select(name='asr.gs::main')
        """
        selection = Selection(
            **selection
        )
        return self.backend.select(selection)

    def wrapper(self, func):  # noqa
        def wrapped(asrcontrol, run_specification):
            selection = {
                'run_specification.name': run_specification.name,
                'run_specification.parameters': run_specification.parameters,
            }
            if self.has(**selection):
                run_record = self.get(**selection)
                print(f'{run_specification.name}: '
                      f'Found cached record.uid={run_record.uid}')
            else:
                run_record = func(asrcontrol, run_specification)
                self.add(run_record)
            return run_record
        return wrapped

    def __call__(self):  # noqa
        return self.wrapper


class MemoryCache:

    def __init__(self):
        self.records = {}

    def add(self, record):
        self.records[record.uid] = record

    def has(self, selection: Selection):
        for value in self.records.values():
            if selection.matches(value):
                return True
        return False

    def select(self, selection: Selection):
        selected = []
        for record in self.records.values():
            if selection.matches(record):
                selected.append(record)
        return selected


def get_cache():
    from .config import config

    if config.backend == 'fscache':
        return file_system_cache
    elif config.backend == 'legacyfscache':
        return legacy_file_system_cache


file_system_cache = Cache(backend=FileCacheBackend())
legacy_file_system_cache = Cache(backend=LegacyFileSystemBackend())
