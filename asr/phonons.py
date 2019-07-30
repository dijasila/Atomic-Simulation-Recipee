from pathlib import Path

import numpy as np

from ase.parallel import world
from ase.io import read
from ase.phonons import Phonons as ASEPhonons

from asr.utils import command, option


class Phonons(ASEPhonons):
    def __init__(self, forces=None, Z_avv=None, eps_vv=None,
                 refcell=None, m_inv_x=None, *args, **kwargs):
        from ase.utils import opencew
        import pickle
        ASEPhonons.__init__(self, refcell=refcell,
                            *args, **kwargs)
        if forces:
            for fbasename, f_av in forces.items():
                fname = f'{fbasename}.pckl'
                if not Path(fname).exists():
                    fd = opencew(fname)
                    if world.rank == 0:
                        pickle.dump(f_av, fd, protocol=2)
                        fd.close()
        self.Z_avv = Z_avv
        self.eps_vv = eps_vv
        self.refcell = refcell
        self.m_inv_x = m_inv_x
        self.N_c = tuple(self.N_c)

    def todict(self):
        from ase.utils import pickleload
        # It would be better to save the calculated forces
        ASEPhonons.read(self)

        dct = dict(atoms=np.arange(len(self.atoms)),  # Dummy atoms
                   supercell=tuple(self.N_c),
                   name=self.name,
                   delta=self.delta,
                   refcell=self.refcell,
                   Z_avv=self.Z_avv,
                   eps_vv=self.eps_vv,
                   m_inv_x=self.m_inv_x)
        forces = {}
        for i, a in enumerate(self.indices):
            for j, v in enumerate('xyz'):
                # Atomic forces for a displacement of atom a in direction v
                basename = '%s.%d%s' % (self.name, a, v)
                fmname = basename + '-'
                fpname = basename + '+'
                fminus_av = pickleload(open(f'{fmname}.pckl', 'rb'))
                fplus_av = pickleload(open(f'{fpname}.pckl', 'rb'))
                forces[fmname] = fminus_av
                forces[fpname] = fplus_av

        feqname = f'{self.name}.eq'
        feq_av = pickleload(open(f'{feqname}.pckl', 'rb'))
        forces[feqname] = feq_av
        dct['forces'] = forces
        return dct


@command('asr.phonons')
@option('-n', default=2, help='Supercell size')
@option('--ecut', default=800, help='Energy cutoff')
@option('--kptdensity', default=6.0, help='Kpoint density')
def main(n, ecut, kptdensity):
    """Calculate Phonons"""
    from asr.calculators import get_calculator
    # Remove empty files:
    if world.rank == 0:
        for f in Path().glob('phonon.*.pckl'):
            if f.stat().st_size == 0:
                f.unlink()
    world.barrier()

    params = {'mode': {'name': 'pw', 'ecut': ecut},
              'kpts': {'density': kptdensity, 'gamma': True}}

    # Set essential parameters for phonons
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True}
    # Make sure to converge forces! Can be important
    if 'convergence' in params:
        params['convergence']['forces'] = 1e-4
    else:
        params['convergence'] = {'forces': 1e-4}

    atoms = read('structure.json')
    fd = open('phonons.txt'.format(n), 'a')
    calc = get_calculator()(txt=fd, **params)

    # Set initial magnetic moments
    from asr.utils import is_magnetic
    if is_magnetic():
        gsold = get_calculator()('gs.gpw', txt=None)
        magmoms_m = gsold.get_magnetic_moments()
        atoms.set_initial_magnetic_moments(magmoms_m)

    from asr.utils import get_dimensionality
    nd = get_dimensionality()
    if nd == 3:
        supercell = (n, n, n)
    elif nd == 2:
        supercell = (n, n, 1)
    elif nd == 1:
        supercell = (n, 1, 1)

    # Read existing forces from old calc
    if Path('results_phonons.json').exists():
        from asr.utils import read_json
        forces = read_json('results_phonons.json')['phonons']['forces']
    else:
        forces = None
    p = Phonons(atoms=atoms, calc=calc, supercell=supercell, forces=forces)
    p.run()
    results = {'phonons': p.todict()}
    return results


def analyse(points=300, modes=False, q_qc=None):
    from asr.utils import read_json
    dct = read_json('results_phonons.json')
    atoms = read('structure.json')
    p = Phonons(**dct['phonons'])
    p.atoms = atoms
    p.read()
    if q_qc is None:
        q_qc = np.indices(p.N_c).reshape(3, -1).T / p.N_c

    out = p.band_structure(q_qc, modes=modes, born=False, verbose=False)
    if modes:
        omega_kl, u_kl = out
        return np.array(omega_kl), u_kl, q_qc
    else:
        omega_kl = out
        return np.array(omega_kl), np.array(omega_kl), q_qc


def plot_phonons(row, fname):
    import matplotlib.pyplot as plt

    freqs = row.data.get('phonon_frequencies_3d')
    if freqs is None:
        return

    gamma = freqs[0]
    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.gca()

    x0 = -0.0005  # eV
    for x, color in [(gamma[gamma < x0], 'r'),
                     (gamma[gamma >= x0], 'b')]:
        if len(x) > 0:
            markerline, _, _ = ax.stem(x * 1000, np.ones_like(x), bottom=-1,
                                       markerfmt=color + 'o',
                                       linefmt=color + '-')
            plt.setp(markerline, alpha=0.4)
    ax.set_xlabel(r'phonon frequency at $\Gamma$ [meV]')
    ax.axis(ymin=0.0, ymax=1.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def collect_data(atoms):
    kvp = {}
    data = {}
    key_descriptions = {}
    try:
        eigs2, freqs2, _ = analyse(atoms)
        eigs3, freqs3, _ = analyse(atoms)
    except (FileNotFoundError, EOFError):
        return {}, {}, {}
    kvp['minhessianeig'] = eigs3.min()
    data['phonon_frequencies_2d'] = freqs2
    data['phonon_frequencies_3d'] = freqs3
    data['phonon_energies_2d'] = eigs2
    data['phonon_energies_3d'] = eigs3

    return kvp, key_descriptions, data


def webpanel(row, key_descriptions):
    from asr.utils.custom import table, fig
    phonontable = table(row, 'Property',
                        ['c_11', 'c_22', 'c_12', 'bulk_modulus',
                         'minhessianeig'], key_descriptions)

    panel = ('Elastic constants and phonons',
             [[fig('phonons.png')], [phonontable]])
    things = [(plot_phonons, ['phonons.png'])]

    return panel, things


group = 'property'
dependencies = ['asr.structureinfo', 'asr.gs']

if __name__ == '__main__':
    main()
