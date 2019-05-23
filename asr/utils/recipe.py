import importlib
from pathlib import Path


class Recipe:

    all_recipes = []
    
    known_attributes = ['main', 'collect_data',
                        'webpanel', 'group', 'creates',
                        'dependencies', 'resources', 'diskspace',
                        'restart']

    def __init__(self, module):
        self.name = self.__name__ = module.__name__
        self.implemented_attributes = []
        for attr in Recipe.known_attributes:
            if hasattr(module, attr):
                self.implemented_attributes.append(attr)
                setattr(self, attr, getattr(module, attr))

    # Alternative contructors
    @classmethod
    def frompath(cls, name, reload=False):
        module = importlib.import_module(f'{name}')
        if reload:
            module = importlib.reload(module)
        return cls(module)

    def done(self):
        name = self.name[4:]
        creates = [f'results-{name}.json']
        if self.creates:
            creates += self.creates

        for file in creates:
            if not Path(file).exists():
                return False
        return True

    def collect(self, atoms):
        kvp = {}
        key_descriptions = {}
        data = {}
        if self.done():
            if self.collect_data:
                kvp, key_descriptions, data = self.collect_data(atoms)

            name = self.name[4:]
            resultfile = Path(f'results-{name}.json')
            from ase.io import jsonio
            results = jsonio.decode(resultfile.read_text())
            key = f'results-{name}'
            msg = f'{self.name}: You cannot put a {key} in data'
            assert key not in data, msg
            data[key] = results

        return kvp, key_descriptions, data
    
    def run(self):
        return self.main(args=[])


for attr in Recipe.known_attributes:
    setattr(Recipe, attr, None)
