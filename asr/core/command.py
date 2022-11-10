"""Implement ASRCommand class and related decorators."""
import functools
import copy
import inspect
import typing
from pathlib import Path

import click
from ase.parallel import parprint

from . import clickify_docstring
from .cache import get_cache
from .parameters import get_default_parameters, Parameters
from .record import Record
from .specification import construct_run_spec
from .runner import runner
from .dependencies import register_dependencies
from .resources import register_resources
from .selector import Selector
from .metadata import register_metadata


def format_param_string(params: dict):
    """Represent params as comma separated string."""
    return ', '.join([f'{key}={repr(value)}' for key, value in
                      params.items()])


class ASRCommand:
    """Class that represents an instruction."""

    def __init__(
            self,
            wrapped_function,
            module=None,
            version=0,
            argument_hooks=None,
            package_dependencies=('asr', 'ase', 'gpaw'),
    ):
        """Construct an instance of an ASRCommand.

        Parameters
        ----------
        func : callable
            Wrapped function that

        """
        assert callable(wrapped_function), \
            'The wrapped object should be callable'

        self.package_dependencies = package_dependencies
        self.module = module
        self.version = version
        self.argument_hooks = argument_hooks or []
        self._wrapped_function = wrapped_function
        self.package_dependencies = self.package_dependencies

        funcname = wrapped_function.__name__
        self.name = f'{module}:{funcname}'

        hints = typing.get_type_hints(wrapped_function)

        if 'return' in hints:
            self.returns = hints['return']
        else:
            self.returns = None

        # Figure out the parameters for this function
        if not hasattr(self._wrapped_function, '__asr_params__'):
            self._wrapped_function.__asr_params__ = {}

        self.myparams = copy.deepcopy(self._wrapped_function.__asr_params__)

        sig = inspect.signature(self._wrapped_function)
        self.__signature__ = sig

        # Setup the CLI
        functools.update_wrapper(self, self._wrapped_function)

    @property
    def cache(self):
        # Commands are defined at import time, but (meaningfully)
        # getting the cache requires that the root is initalized.
        # Therefore cache is lazy.
        return get_cache(backend='filesystem')

    def new(self, **newkwargs):
        """Make new instance of instruction with new settings."""
        cls = type(self)
        kwargs = dict(
            wrapped_function=self._wrapped_function,
            module=self.module,
            version=self.version,
            cache=self.cache,
            argument_hooks=self.argument_hooks,
            package_dependencies=self.package_dependencies,
        )
        kwargs.update(newkwargs)
        return cls(**kwargs)

    def get_signature(self):
        """Return signature with updated defaults based on params.json."""
        myparams = []
        for key, value in self.__signature__.parameters.items():
            assert key in self.myparams, \
                f'Missing description for param={key},value={value}.'
            myparams.append(key)

        # Check that all annotated parameters can be found in the
        # actual function signature.
        myparams = [k for k, v in self.__signature__.parameters.items()]
        for key in self.myparams:
            assert key in myparams, f'param={key} is unknown.'

        default_parameters = get_default_parameters(self.name)
        if default_parameters:
            signature_parameters = dict(self.__signature__.parameters)
            for key, new_default in default_parameters.items():
                assert key in signature_parameters, \
                    f'Unknown param in params.json: param={key}.'
                parameter = signature_parameters[key]
                signature_parameters[key] = parameter.replace(
                    default=new_default)

            new_signature = self.__signature__.replace(
                parameters=[val for val in signature_parameters.values()])
            return new_signature

        return self.__signature__

    @property
    def defaults(self):
        """Get default parameters based on signature and params.json."""
        signature = self.get_signature()
        defparams = {}
        for key, value in signature.parameters.items():
            if value.default is not inspect.Parameter.empty:
                defparams[key] = value.default
        return Parameters(parameters=defparams)

    def get_argument_descriptors(self):
        """Get the parameters of this function."""
        return self.myparams

    def cli(self, args=None):
        """Parse parameters from command line and call wrapped function.

        Parameters
        ----------
        args : List of strings or None
            List of command line arguments. If None: Read arguments from
            sys.argv.
        """
        command = make_cli_command(self)
        return command(standalone_mode=False,
                       prog_name=f'asr run {self.name}', args=args)

    def get_wrapped_function(self):
        """Return wrapped function."""
        return self._wrapped_function

    def __call__(self, *args, **kwargs):
        """Delegate to self.get."""
        return self.get(*args, **kwargs).result

    def prepare_parameters(self, *args, **kwargs):
        parameters = apply_defaults(
            self.get_signature(), *args, **kwargs)
        parameters = Parameters(parameters=parameters)
        for hook in self.argument_hooks:
            parameters = hook(parameters)

        return parameters

    def make_run_specification(self, parameters: Parameters):
        # func = self.get_wrapped_function()
        run_specification = construct_run_spec(
            name=self.name,
            parameters=parameters,
            version=self.version,
            codes=self.package_dependencies,
        )
        return run_specification

    def get(self, *args, **kwargs):
        """Return record.

        This is the main function of an ASRCommand. It takes care of
        reading parameters, creating metadata, checksums etc. If you
        want to understand what happens when you execute an ASRCommand
        this is a good place to start.

        Implementation goes as follows::

            1. Parse input parameters
            2. Check if a cached result already exists
               and return that if it does.
            --- Otherwise
            3. Run all dependencies.
            4. Get execution metadata, ie., code_versions, created files and
               required files.

        """
        # Inspired by: lab-notebook, provenance, invoke, fabric, joblib
        # TODO: Tag results with random run #ID.
        # TODO: Converting old result files to new format.
        # TODO: Save date and time.
        # TODO: We should call external files side effects.
        # TODO: When to register current run as a dependency.
        # TODO: Locking reading of results file.
        # TODO: Use time stamp for hashing files as well.
        # TODO: Easy to design a system for pure functions,
        # but we need side effects as well.
        # TODO: Should we have an ignore keyword?
        # TODO: Some parameters need to know about others in order
        # to properly initialize, eg., in GPAW the poisson solver need
        # to know about the dimensionality to set dipole layer and also to
        # get the setup fingerprints, also 1D materials need higher
        # kpoint density.
        # TODO: SHA256 vs MD5 speeeeeeed?
        # TODO: All arguments requires a JSON serialization method.
        # TODO: How do Django make data migrations?
        # TODO: Require completely flat ASRResults data structure?
        # TODO: Should we have a way to Signal ASR (think click.Context)?

        # TODO: The caching database could be of a non-relational
        # format (would be similar to current format).

        # TODO: Should Result objects have some sort of verification
        # mechanism? Like checking acoustic sum rules?

        # TODO: Make clean run environment class?
        # TODO: Make an old type results object.

        # REQ: Recipe must be able to run multiple times and cache
        # their results (think LRU-cache).

        # REQ: Must be possible to change implementation of recipe
        #      without invalidating previous results
        # REQ: Must support side-effects such as files written to disk.
        # REQ: Must store information about code versions
        # REQ: Must be able to read defaults from configuration file on
        #      a per-folder basis.
        # REQ: Must support chaining of recipes (dependencies).

        # REQ: Caching database should be simple and decentralized
        # (think sqlite).

        # REQ: Caching database should be plain text.

        # REQ: Returned object should be self-contained (think ase
        # BandStructure object).

        # REQ: Returned objects must be able to present themselves as
        # figures and HTML.

        # REQ: Must be delocalized from ASR (ie. must be able to have
        # a seperate set of recipes locally, non-related to asr).

        # REQ: Must also have a packaging mechanism for entire
        # projects (ie. ASE databases).

        # REQ: Must be possible to call as a simple python function
        # thus circumventing CLI.

        # REQ: Must be able to call without scripting, eg. through a CLI.
        # REQ: Must support all ASE calculators.

        parameters = self.prepare_parameters(*args, **kwargs)
        run_specification = self.make_run_specification(parameters)
        cache = self.cache

        @register_dependencies.register
        @cache(make_selector=self.make_selector)
        @register_metadata()
        @register_dependencies()
        @runner()
        @register_resources()
        def execute_run_spec(run_spec):
            name = run_spec.name
            parameters = run_spec.parameters
            paramstring = ', '.join([f'{key}={repr(value)}' for key, value in
                                     parameters.items()])
            parprint(f'Running {name}({paramstring})')
            result = run_spec()
            record = Record(
                result=result,
                run_specification=run_spec,
            )
            return record

        run_record = execute_run_spec(run_specification)
        return run_record

    def has(self, *args, **kwargs):

        try:
            parameters = self.prepare_parameters(*args, **kwargs)
        except FileNotFoundError:
            # Some recipes have files as input, and they may trigger
            # errors which definitely indicate a cache miss.
            #
            # Maybe this check could be done in a more appropriate way
            # or place.
            return False

        run_spec = self.make_run_specification(parameters)
        sel = self.make_selector(run_spec)
        return self.cache.has(selector=sel)

    def make_selector(self, run_specification):
        """Make selector for matching previous records."""
        selector = Selector()

        selector.run_specification.name = selector.EQ(run_specification.name)
        selector.run_specification.version = selector.EQ(run_specification.version)
        selector.run_specification.parameters = \
            lambda value: set(value.keys()) == set(
                run_specification.parameters.keys())

        for name, param in self.get_argument_descriptors().items():
            matcher = param['matcher']
            if matcher is None:
                matcher = selector.EQ
            setattr(selector, f'parameters.{name}',
                    matcher(run_specification.parameters[name]))
        return selector


def get_recipe_module_names():
    # Find all modules containing recipes
    asrfolder = Path(__file__).parent.parent
    folders_with_recipes = [asrfolder / '.',
                            asrfolder / 'c2db',
                            asrfolder / 'setup',
                            asrfolder / 'database',
                            asrfolder / 'c2db']
    files = [filename for folder in folders_with_recipes
             for filename in folder.glob("[a-zA-Z]*.py")]
    modulenames = []
    for file in files:
        name = str(file.with_suffix(''))[len(str(asrfolder)):]
        modulename = 'asr' + name.replace('/', '.')
        modulenames.append(modulename)
    return modulenames


def get_recipe_modules():
    # Get recipe modules
    import importlib
    modules = get_recipe_module_names()

    mods = []
    for module in modules:
        mod = importlib.import_module(module)
        mods.append(mod)
    return mods


def get_recipes():
    # Get all recipes in all modules
    modules = get_recipe_modules()

    functions = []
    for module in modules:
        for obj in module.__dict__.values():
            # Commands may exist in multiple module namespaces.
            # We only include commands as defined in their own module.
            if isinstance(obj, ASRCommand) and (obj.__module__
                                                == module.__name__):
                functions.append(obj)
    return functions


def make_cli_command(asr_command: ASRCommand):
    command = setup_cli(
        asr_command.get_wrapped_function(),
        asr_command.get,
        asr_command.defaults,
        asr_command.get_argument_descriptors(),
    )
    return command


def setup_cli(wrapped, wrapper, defparams, parameters):
    # Click CLI Interface
    CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

    cc = click.command
    co = click.option

    help = clickify_docstring(wrapped.__doc__) or ''

    command = cc(context_settings=CONTEXT_SETTINGS,
                 help=help)(wrapper)

    # Convert parameters into CLI Parameters!
    for name, param in parameters.items():
        param = param.copy()
        alias = param.pop('alias')
        argtype = param.pop('argtype')
        name2 = param.pop('name')
        param.pop('matcher')
        assert name == name2
        assert name in parameters
        if 'default' in param:
            default = param.pop('default')
        else:
            default = defparams.get(name, None)

        if 'type' in param:
            try:
                param['type'].default = default
            except (AttributeError, TypeError):
                pass

        if argtype == 'option':
            command = co(show_default=True, default=default,
                         *alias, **param)(command)
        else:
            assert argtype == 'argument'
            command = click.argument(*alias, **param)(command)

    return command


def apply_defaults(signature, *args, **kwargs):
    """Apply defaults to args and kwargs.

    Reads the signature of the wrapped function and applies the
    defaults where relevant.

    """
    bound_arguments = signature.bind_partial(*args, **kwargs)
    bound_arguments.apply_defaults()
    params = copy.deepcopy(dict(bound_arguments.arguments))
    return params
