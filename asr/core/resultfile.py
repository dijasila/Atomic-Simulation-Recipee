"""Functionality for converting old resultfiles to records."""

import fnmatch
import pathlib
import typing
from dataclasses import dataclass

from ase import Atoms
from ase.db.core import AtomsRow
from ase.io import read

from asr.core import ASRResult, read_json

from .dependencies import Dependencies, Dependency
from .metadata import construct_metadata
from .parameters import Parameters
from .record import Record
from .results import decode_object
from .specification import RunSpecification, get_new_uuid
from .utils import (
    get_recipe_from_name,
    parse_mod_func,
    fix_recipe_name_if_recipe_has_been_moved,
    add_main_to_name_if_missing,
)
from .old_defaults import get_old_defaults


@dataclass
class RecordContext:
    """Class that contain the contextual data to create a record.

    The record context is an intermediate data format that can be used to
    construct the final records.

    Attributes
    ----------
    result : Any
        The result of the record to be constructed.
    instruction_name : str
        The name of the instruction that was used to construct the record.
    atomic_structures : Dict[str, Atoms]
        Mapping that maps a filename to an atoms object, ie.
        dict("structure.json"=Atoms(...)).
    uid : str
        The UID to be assigned to the record to be constructed.
    dependency_matcher: Callable[[RecordContext], bool]
        A function that can determine if another context is a dependency
        of this context.
    dependencies : Optional[Dependencies]
        The dependencies of the record to be constructed. The dependency
        matcher is typically used to set this attribute after all contexts
        has been created.
    directory : str
        The directory that the record is constructed in (relative to the cache).
    path : Path
        The path to the actual resultfile.
    """

    result: typing.Any
    instruction_name: str
    atomic_structures: typing.Dict[str, Atoms]
    uid: str
    dependency_matcher: typing.Callable[["RecordContext"], bool]
    dependencies: typing.Optional[Dependencies]
    directory: str
    path: typing.Optional[pathlib.Path] = None


def generate_uids(
    filenames: typing.Union[typing.List[str], typing.List[pathlib.Path]],
) -> typing.Dict[typing.Union[str, pathlib.Path], str]:
    """Generate new uids for a set of files."""
    return {filename: get_new_uuid() for filename in filenames}


def find_results_files(directory: pathlib.Path) -> typing.List[pathlib.Path]:
    """Find all resultfiles in directory (recursively)."""
    filenames = [
        str(filename)
        for filename in pathlib.Path(directory).rglob("results-asr.*.json")
    ]
    filenames = filter_filenames_for_unused_recipe_results(filenames)
    return [pathlib.Path(filename) for filename in filenames]


def filter_filenames_for_unused_recipe_results(
    filenames: typing.List[str],
) -> typing.List[str]:
    """Filter filenames for files from instructions that no longer exist."""
    skip_patterns = [
        "*results-asr.database.fromtree.json",
        "*results-asr.database.app.json",
        "*results-asr.database.key_descriptions.json",
        "*results-asr.setup.strains*.json",
        "*displacements*/*/results-asr.database.material_fingerprint.json",
        "*strains*/results-asr.database.material_fingerprint.json",
        "*asr.setinfo*",
        "*asr.setup.params.json",
        "*asr.c2db.exchange@calculate*",
        "*asr.c2db.exchange:calculate*",
    ]

    paths = []
    for path in filenames:
        if any(fnmatch.fnmatch(path, skip_pattern) for skip_pattern in skip_patterns):
            continue
        paths.append(path)

    return paths


def filter_contexts_for_unused_recipe_results(
    contexts: typing.List[RecordContext],
) -> typing.List[RecordContext]:
    """Filter contexts for instructions that no longer exist."""
    skip_patterns = [
        "*asr.database.fromtree*",
        "*asr.database.app*",
        "*asr.database.key_descriptions*",
        "*asr.setup.strains*",
        "*asr.database.material_fingerprint*",
        "*asr.setinfo*",
        "*asr.setup.params*",
        "*asr.c2db.exchange:calculate*",
        "*asr.c2db.plasmafrequency:calculate*",
    ]
    filtered = []
    for context in contexts:
        if any(
            fnmatch.fnmatch(context.instruction_name, skip_pattern)
            for skip_pattern in skip_patterns
        ):
            continue
        filtered.append(context)
    return filtered


ATOMSFILES = ["structure.json", "original.json", "start.json", "unrelaxed.json"]


def construct_record_from_context(
    record_context: RecordContext,
) -> Record:
    """Construct a Record from a context."""
    from asr.core.codes import Code, Codes
    from asr.core.results import MetaDataNotSetError

    result = record_context.result
    instruction_name = record_context.instruction_name
    atomic_structures = record_context.atomic_structures
    uid = record_context.uid
    dependencies = record_context.dependencies
    directory = record_context.directory

    mod, func = parse_mod_func(instruction_name)
    instruction_name = ":".join([mod, func])
    if isinstance(result, ASRResult):
        data = result.data
        metadata = result.metadata.todict()
    else:
        assert isinstance(result, dict)
        data = result
        params: typing.Dict[str, typing.Any] = {}
        metadata = {
            "asr_name": instruction_name,
            "params": params,
        }

    recipe = get_recipe_from_name(instruction_name)
    if not issubclass(recipe.returns, ASRResult):
        returns = ASRResult
    else:
        returns = recipe.returns
    result = returns(
        data=data,
        metadata=metadata,
        strict=False,
    )

    try:
        parameters = result.metadata.params
        if instruction_name == "asr.c2db.gs:calculate" and "name" in parameters:
            del parameters["name"]
    except MetaDataNotSetError:
        parameters = {}

    parameters = Parameters(parameters)
    params = recipe.get_argument_descriptors()
    atomsparam = [param for name, param in params.items() if name == "atoms"][0]
    atomsfilename = atomsparam["default"]
    if atomsfilename not in atomic_structures:
        atomic_structures[atomsfilename] = "Unknown atoms file"

    parameters.atomic_structures = atomic_structures

    try:
        code_versions = result.metadata.code_versions
        lst_codes = []
        for package, codestr in code_versions.items():
            version, githash = codestr.split("-")
            code = Code(package, version, githash)
            lst_codes.append(code)
        codes = Codes(lst_codes)
    except MetaDataNotSetError:
        codes = Codes([])

    from asr.core.resources import Resources

    try:
        resources = result.metadata.resources
        resources = Resources(
            execution_start=resources.get("tstart"),
            execution_end=resources.get("tend"),
            execution_duration=resources["time"],
            ncores=resources["ncores"],
        )
    except MetaDataNotSetError:
        resources = Resources()

    record = Record(
        run_specification=RunSpecification(
            name=instruction_name,
            parameters=parameters,
            version=-1,
            codes=codes,
            uid=uid,
        ),
        metadata=construct_metadata(
            directory=directory,
        ),
        resources=resources,
        result=result,
        tags=["resultfile"],
        dependencies=dependencies,
    )

    return record


def fix_asr_gs_result_missing_calculator(
    folder: pathlib.Path,
    result: typing.Any,
    instruction_name: str,
) -> typing.Any:
    """Fix ground state "calculate" results that didn't store result as ASRResult.

    Parameters
    ----------
    folder : pathlib.Path
        The folder where the corresponding gs.gpw exists.
    result : typing.Any
        The result that might have to be fixed.
    instruction_name : str
        The instruction name of the result.

    Returns
    -------
    typing.Any
        A potentially fixed result object.
    """
    if isinstance(result, dict) and instruction_name == "asr.c2db.gs:calculate":
        from asr.c2db.gs import GroundStateCalculationResult
        from asr.calculators import Calculation

        calculation = Calculation(
            id="gs",
            cls_name="gpaw",
            paths=[folder / "gs.gpw"],
        )
        result = GroundStateCalculationResult.fromdata(calculation=calculation)
        calc = calculation.load()
        calculator = calc.parameters
        calculator["name"] = "gpaw"
        params = {"calculator": calculator}
        metadata = {
            "asr_name": instruction_name,
            "params": params,
        }
        result.metadata = metadata
    return result


def fix_asr_gs_record_missing_calculator(record: Record) -> None:
    """Fix asr gs calculate records that are missing calculator parameters.

    Assumes that the missing calculator is given by the old default value.
    """
    if record.name == "asr.c2db.gs:calculate":
        if "calculator" not in record.parameters:
            record.parameters.calculator = get_old_defaults()["asr.c2db.gs:calculate"][
                "calculator"
            ]


def set_context_dependencies(
    contexts: typing.List[RecordContext],
) -> typing.List[RecordContext]:
    """Use dependency matchers on contexts to set context.dependencies attributes.

    Parameters
    ----------
    contexts : typing.List[RecordContext]
        Input context where dependencies haven't been set yet.

    Returns
    -------
    typing.List[RecordContext]
        Output contexts where the concrete dependencies attribute has been set.
    """
    for context in contexts:
        deps = []
        for context2 in contexts:
            if context.dependency_matcher(context2):
                dep = Dependency(
                    uid=context2.uid,
                    revision=None,
                )
                deps.append(dep)
        if context.instruction_name == "asr.c2db.stiffness:main":
            assert deps
        context.dependencies = Dependencies(deps)
    return contexts


def get_recipe_name_from_filename(filename: str) -> str:
    """Get recipe/instruction name from resultsfile filename.

    Takes core of old files with @ and add "main" to name if missing. Also
    takes care of adding c2db to names that represent instructions that were
    moved to the c2db sub-package.
    """
    from os.path import splitext

    name = splitext(filename.split("-")[1])[0]
    name = name.replace("@", ":")
    name = add_main_to_name_if_missing(name)
    name = fix_recipe_name_if_recipe_has_been_moved(name)
    return name


def make_dependency_matcher(
    patterns: typing.List[str],
    attribute: str = "instruction_name",
) -> typing.Callable[[RecordContext], bool]:
    """Make dependency matcher.

    Helper function for making callables that can be used as dependency
    matchers for contexts. The callable will use get `attribute` from another
    context and see if it matches with any of the `patterns`.

    Parameters
    ----------
    patterns : typing.List[str]
        fnmatch patterns to match to "attribute" on other context.
        E.g. ["*asr.gs*", "asr.magnetic_anis*"].
    attribute : str, optional
        Try matching patterns with this attribute, by default "instruction_name".

    Returns
    -------
    typing.Callable[[RecordContext], bool]
        A callable that can be used as dependency matcher.
    """

    def dependency_matcher(context: RecordContext) -> bool:
        return any(
            fnmatch.fnmatch(str(getattr(context, attribute)), pattern)
            for pattern in patterns
        )

    return dependency_matcher


def get_dependency_matcher_from_name(
    name: str,
) -> typing.Callable[[RecordContext], bool]:
    """Map instruction name to a dependency matcher.

    Parameters
    ----------
    name : str
        The instruction name.

    Returns
    -------
    typing.Callable[[RecordContext], bool]
        The associated dependency matcher.
    """
    # Some manually implemented dependencies
    if name == "asr.c2db.piezoelectrictensor:main":
        patterns = [
            "*strains*/results-asr.relax.json",
            "*strains*/results-asr.formalpolarization.json",
        ]
        return make_dependency_matcher(patterns, "path")
    elif name == "asr.c2db.stiffness:main":
        patterns = [
            "*strains*/results-asr.relax.json",
        ]
        return make_dependency_matcher(patterns, "path")

    deps: typing.Dict[str, typing.List[str]] = {
        "asr.c2db.infraredpolarizability": [
            "asr.c2db.phonons",
            "asr.c2db.borncharges",
            "asr.c2db.polarizability",
        ],
        "asr.c2db.emasses:refine": [
            "asr.structureinfo",
            "asr.c2db.magnetic_anisotropy",
            "asr.c2db.gs",
        ],
        "asr.c2db.emasses": [
            "asr.c2db.emasses:refine",
            "asr.c2db.gs:calculate",
            "asr.c2db.gs",
            "asr.structureinfo",
            "asr.c2db.magnetic_anisotropy",
        ],
        "asr.c2db.emasses:validate": ["asr.c2db.emasses"],
        "asr.c2db.berry:calculate": ["asr.c2db.gs"],
        "asr.c2db.berry": ["asr.c2db.berry:calculate", "asr.c2db.gs"],
        "asr.c2db.gw:gs": ["asr.c2db.gs:calculate"],
        "asr.c2db.gw:gw": ["asr.c2db.gw:gs"],
        "asr.c2db.gw:empirical_mean_z": ["asr.c2db.gw:gw"],
        "asr.c2db.gw": ["asr.c2db.bandstructure", "asr.c2db.gw:empirical_mean_z"],
        "asr.c2db.pdos:calculate": ["asr.c2db.gs"],
        "asr.c2db.pdos": ["asr.c2db.gs", "asr.c2db.pdos:calculate"],
        "asr.c2db.phonons:calculate": [],
        "asr.c2db.phonons": ["asr.c2db.phonons:calculate"],
        "asr.c2db.push": ["asr.structureinfo", "asr.c2db.phonons"],
        "asr.c2db.phonopy:calculate": ["asr.c2db.gs:calculate"],
        "asr.c2db.phonopy": ["asr.c2db.phonopy:calculate"],
        "asr.c2db.hse:calculate": [
            "asr.structureinfo",
            "asr.c2db.gs:calculate",
            "asr.c2db.gs",
        ],
        "asr.c2db.hse": ["asr.c2db.hse:calculate", "asr.c2db.bandstructure"],
        "asr.c2db.exchange:calculate": ["asr.c2db.gs:calculate"],
        "asr.c2db.exchange": ["asr.c2db.gs:main"],
        "asr.c2db.plasmafrequency:calculate": ["asr.c2db.gs:calculate"],
        "asr.c2db.plasmafrequency": [
            "asr.c2db.plasmafrequency:calculate",
            "asr.c2db.gs",
        ],
        "asr.c2db.shg": ["asr.c2db.gs:calculate", "asr.c2db.gs"],
        "asr.c2db.magstate": ["asr.c2db.gs:calculate"],
        "asr.c2db.fermisurface": ["asr.c2db.gs", "asr.structureinfo"],
        "asr.c2db.magnetic_anisotropy": ["asr.c2db.gs:calculate", "asr.c2db.magstate"],
        "asr.c2db.convex_hull": [
            "asr.structureinfo",
            "asr.database.material_fingerprint",
        ],
        "asr.c2db.borncharges": ["asr.c2db.gs:calculate"],
        "asr.c2db.gs": [
            "asr.c2db.gs:calculate",
            "asr.c2db.magnetic_anisotropy",
            "asr.structureinfo",
        ],
        "asr.c2db.bandstructure:calculate": ["asr.c2db.gs:calculate"],
        "asr.c2db.bandstructure": [
            "asr.c2db.bandstructure:calculate",
            "asr.c2db.gs",
            "asr.structureinfo",
            "asr.c2db.magnetic_anisotropy",
        ],
        "asr.defectformation": ["asr.setup.defects", "asr.c2db.gs"],
        "asr.c2db.deformationpotentials": ["asr.c2db.gs"],
        "asr.c2db.bader": ["asr.c2db.gs"],
        "asr.bse:calculate": ["asr.c2db.gs:calculate"],
        "asr.bse": ["asr.bse:calculate", "asr.c2db.gs"],
        "asr.c2db.projected_bandstructure": ["asr.c2db.gs", "asr.c2db.bandstructure"],
        "asr.c2db.shift": ["asr.c2db.gs:calculate"],
        "asr.c2db.polarizability": ["asr.structureinfo", "asr.c2db.gs:calculate"],
    }
    deps = {add_main_to_name_if_missing(key): value for key, value in deps.items()}
    dependencies = []
    for dep in deps.get(name, []):
        dep = fix_recipe_name_if_recipe_has_been_moved(add_main_to_name_if_missing(dep))
        dependencies.append(dep)

    return make_dependency_matcher(dependencies, "instruction_name")


def get_resultfile_records_in_directory(directory: pathlib.Path) -> typing.List[Record]:
    contexts = get_contexts_in_directory(directory)
    contexts = filter_contexts_for_unused_recipe_results(contexts)
    contexts = set_context_dependencies(contexts)
    records = make_records_from_contexts(contexts)
    return records


def get_resultfile_records_from_database_row(row: AtomsRow):
    contexts = convert_row_data_to_contexts(row.data, row.get("folder", "."))
    contexts = filter_contexts_for_unused_recipe_results(contexts)
    contexts = set_context_dependencies(contexts)
    records = make_records_from_contexts(contexts)
    return records


def get_contexts_in_directory(
    directory=pathlib.Path("."),
) -> typing.List[RecordContext]:
    resultsfiles = find_results_files(directory=directory)
    uids = generate_uids(resultsfiles)
    contexts = []
    for path in resultsfiles:
        uid = uids[path]
        try:
            context = get_context_from_path_to_resultfile(path, uid, directory)
        except ModuleNotFoundError as error:
            print(error)
            continue
        contexts.append(context)
    return contexts


def get_context_from_path_to_resultfile(
    path: pathlib.Path, uid: str, directory: pathlib.Path
) -> RecordContext:
    result = read_json(path)
    instruction_name = get_recipe_name_from_filename(path.name)
    folder = path.parent
    atomic_structures = {
        atomsfilename: read(folder / atomsfilename).copy()
        for atomsfilename in ATOMSFILES
        if pathlib.Path(folder / atomsfilename).is_file()
    }
    matcher = get_dependency_matcher_from_name(instruction_name)
    rel_directory = str(folder.absolute().relative_to(directory))
    result = fix_asr_gs_result_missing_calculator(folder, result, instruction_name)
    context = RecordContext(
        result=result,
        instruction_name=instruction_name,
        atomic_structures=atomic_structures,
        uid=uid,
        dependency_matcher=matcher,
        dependencies=None,
        directory=rel_directory,
        path=path,
    )
    return context


def convert_row_data_to_contexts(
    data: typing.Dict[str, typing.Any], directory: str
) -> typing.List[RecordContext]:
    """Convert row data to contexts.

    If it exists, utilizes __children_data__ set by the collapse tool to also construct
    contexts from child materials.

    Parameters
    ----------
    data : typing.Dict[str, typing.Any]
        A row data object.
    directory : str
        The directory that the row originally was collected from.

    Returns
    -------
    typing.List[RecordContext]
        The collected set of contexts.
    """
    filenames = []
    for filename in data:
        is_results_file = filename.startswith("results-") and filename.endswith(".json")
        if not is_results_file:
            continue
        filenames.append(filename)

    contexts = []

    children_data = data.get("__children_data__", {})
    for child_values in children_data.values():
        child_directory = child_values["directory"]
        child_data = child_values["data"]
        child_contexts = convert_row_data_to_contexts(child_data, child_directory)
        contexts.extend(child_contexts)

    uids = generate_uids(filenames)
    for filename in filenames:
        instruction_name = get_recipe_name_from_filename(filename)
        result = data[filename]

        try:
            result = decode_object(result)
        except ModuleNotFoundError as err:
            print(err)
            continue
        result = fix_result_object_missing_name(instruction_name, result)

        path = pathlib.Path(directory) / filename
        atomic_structures = {}
        for name, value in data.items():
            if name in ATOMSFILES:
                dct = value["1"]
                atoms = Atoms(
                    numbers=dct.get("numbers"),
                    positions=dct.get("positions"),
                    cell=dct.get("cell"),
                    pbc=dct.get("pbc"),
                    magmoms=dct.get("initial_magmoms"),
                    charges=dct.get("initial_charges"),
                    tags=dct.get("tags"),
                    masses=dct.get("masses"),
                    momenta=dct.get("momenta"),
                    constraint=dct.get("constraints"),
                )
                atomic_structures[name] = atoms

        uid = uids[filename]
        matcher = get_dependency_matcher_from_name(instruction_name)
        context = RecordContext(
            result=result,
            instruction_name=instruction_name,
            atomic_structures=atomic_structures,
            uid=uid,
            dependency_matcher=matcher,
            dependencies=None,
            directory=directory,
            path=path,
        )
        contexts.append(context)

    return contexts


def fix_result_object_missing_name(
    instruction_name: str, result: typing.Any
) -> typing.Any:
    """Check if result is dict and missing name, if yes, fix.

    Parameters
    ----------
    instruction_name : str
        The name of the instruction
    result : typing.Any
        The result.

    Returns
    -------
    typing.Any
        A potentially fixed name.
    """
    if isinstance(result, dict) and "__asr_name__" not in result:
        result["__asr_name__"] = instruction_name
        result = decode_object(result)
    return result


def make_records_from_contexts(
    contexts: typing.List[RecordContext],
) -> typing.List[Record]:
    """Convert record contexts to records.

    Parameters
    ----------
    contexts : typing.List[RecordContext]
        Input contexts.

    Returns
    -------
    typing.List[Record]
        Output records.
    """
    records = []
    for context in contexts:
        record = construct_record_from_context(context)
        fix_asr_gs_record_missing_calculator(record)
        records.append(record)
    records = inherit_dependency_parameters(records)
    return records


def inherit_dependency_parameters(records: typing.List[Record]) -> typing.List[Record]:
    """Add a parameter 'dependency_parameters' with parameters of dependencies.

    Uses context.dependencies and stores all dependency parameters (recursively).

    Parameters
    ----------
    records : typing.List[Record]
        Input records where dependencies have been set.

    Returns
    -------
    typing.List[Record]
        Output records where a the new parameter 'dependency_parameters' has been set.
    """
    for record in records:
        deps = record.dependencies
        dep_params = get_dependency_parameters(deps, records)
        dep_params = Parameters({"dependency_parameters": dep_params})
        record.parameters.update(dep_params)

    return records


def get_dependency_parameters(
    dependencies: typing.Optional[Dependencies], records: typing.List[Record]
) -> Parameters:
    """Get parameters of dependencies (recursively).

    Returns a new parameters object with key equalling the dependency
    instruction names like 'asr.c2db.gs:calculate' and values being the
    dependency parameters.

    Parameters
    ----------
    dependencies : typing.Optional[Dependencies]
        Find parameters for these dependencies.
    records : typing.List[Record]
        Look for the parameters in these records.

    Returns
    -------
    "Parameters"
        Dependency parameters where key=instruction_name and values
        are the dependency parameters.
    """
    params = Parameters({})
    if dependencies is None:
        return params

    for dependency in dependencies:
        deps = [other for other in records if other.uid == dependency.uid]
        if not deps:
            continue
        dep = deps[0]
        depparams = Parameters(
            {
                dep.name: {
                    key: value
                    for key, value in dep.parameters.items()
                    if key != "dependency_parameters"
                }
            }
        )
        params.update(depparams)
        params.update(get_dependency_parameters(dep.dependencies, records))

    return params
