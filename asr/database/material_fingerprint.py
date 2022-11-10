"""Unique hash of atomic structure."""
import numpy as np
from asr.core import command, ASRResult, atomsopt
from ase import Atoms


def todict(atoms):
    d = dict(atoms.arrays)
    d['cell'] = np.asarray(atoms.cell)
    d['pbc'] = atoms.pbc
    if atoms._celldisp.any():
        d['celldisp'] = atoms._celldisp
    # if atoms.constraints:
    #     d['constraints'] = atoms.constraints
    if atoms.info:
        d['info'] = atoms.info
    return d


def get_hash_of_atoms(atoms):
    from hashlib import md5
    import json

    dct = todict(atoms)

    for key, value in dct.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dct[key] = value

    # hash is only reproducible if we sort the keys:
    hash = md5(json.dumps(dct, sort_keys=True).encode()).hexdigest()
    return hash


def get_uid_of_atoms(atoms, hash):
    formula = atoms.symbols.formula
    return f'{formula:abc}-' + hash[:12]


@command(module='asr.database.material_fingerprint')
@atomsopt
def main(atoms: Atoms) -> ASRResult:
    return fingerprint(atoms)


def fingerprint(atoms):
    hash = get_hash_of_atoms(atoms)
    uid = get_uid_of_atoms(atoms, hash)
    results = {'asr_id': hash,
               'uid': uid}
    return results


if __name__ == '__main__':
    main.cli()
