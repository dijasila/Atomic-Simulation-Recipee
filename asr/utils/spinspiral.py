from asr.utils.magnetism import magnetic_atoms
from asr.utils.symmetry import c2db_symmetry_eps
from scipy.spatial.transform import Rotation as R
import numpy as np


def get_collinear_magmoms(atoms, request='priority', calculator=None, verbose=False):
    if request == 'priority' or request == 'p':
        for get_moments, arg in zip([get_calculator_moments,
                                     get_magnetic_moments,
                                     get_initial_magnetic_moments,
                                     get_guess_moments],
                                    [calculator, atoms, atoms, atoms]):
            try:
                moments = get_moments(arg)
                break
            except MagmomError as err:
                if verbose:
                    print(err)

    elif request == 'calculator' or request == 'calc':
        moments = get_calculator_moments(calculator)
    elif request == 'calculated' or request == 'c':
        moments = get_magnetic_moments(atoms)
    elif request == 'initial' or request == 'i':
        moments = get_initial_magnetic_moments(atoms)
    elif request == 'guess' or request == 'g':
        moments = get_guess_moments(atoms)

    return moments


def get_noncollinear_magmoms(atoms, calculator=None, request='priority', verbose=False):
    try:
        moments = get_calculator_moments(calculator)
        return moments
    except MagmomError as err:
        if verbose:
            print(err)
    return col_to_ncolx(get_collinear_magmoms(atoms=atoms, calculator=calculator,
                                              request=request, verbose=verbose))


class MagmomError(Exception):
    pass


def get_calculator_moments(calculator=None):
    if not isinstance(calculator, dict):
        raise MagmomError('Calculator dictionary not provided')

    ncol_moments = calculator.get("magmoms", None)
    if ncol_moments is None:
        raise MagmomError('Magnetic moments not found in Calculator')

    moments = ncol_to_col(ncol_moments)
    return moments


def get_magnetic_moments(atoms):
    from ase.calculators.singlepoint import SinglePointCalculator
    if type(atoms.calc) is SinglePointCalculator:
        moments = atoms.calc.get_property('magmoms', allow_calculation=False)
        return moments

    newGPAW = getattr(atoms.calc, 'dft', None)
    oldGPAW = getattr(atoms.calc, 'setups', None)
    if newGPAW is not None:
        _, moments = atoms.calc.dft.magmoms()
    elif oldGPAW is not None:
        _, moments = atoms.calc.density.estimate_magnetic_moments()
    else:
        raise MagmomError('No calculated magnetic moment found')
    return ncol_to_col(moments)


def get_initial_magnetic_moments(atoms):
    if not atoms.has('initial_magmoms'):
        raise MagmomError('No initial magnetic moments were found')
    return atoms.get_initial_magnetic_moments()


def get_guess_moments(atoms):
    return np.array(magnetic_atoms(atoms), dtype=float)


def col_to_ncolx(moments):
    magmoms = np.zeros((len(moments), 3))
    magmoms[:, 0] = moments
    return np.array(magmoms, dtype=float)


def ncol_to_col(nmoments, eps=1e-16):
    nmoments = np.array(nmoments)
    axisarg = np.logical_not(np.all(abs(nmoments).T < eps, axis=-1))
    assert sum(axisarg) == 1, f'Magnetic moments should be collinear, {axisarg}'
    return nmoments.T[axisarg][0]


def true_magnetic_atoms(atoms, moments):
    if len(moments.shape) > 1:
        moments = ncol_to_col(moments)
    arg = magnetic_atoms(atoms)

    # Ignore ligand atoms
    magmoms = abs(moments[arg])
    try:
        primary_magmom = magmoms[np.argmax(magmoms)]
    except ValueError:
        print(f'Skipping {atoms.symbols}, no d-orbital present')
        return arg

    # Percentage of primary magnetic moment
    pmom = magmoms / primary_magmom

    # Threshold on small moments
    is_true_mag = pmom > 0.1
    arg[arg == 1] = is_true_mag
    return arg


def get_afms(moments, arg, return_raw=False):
    """
    Takes an atoms object, or optionally magnetic moments as a list of floats
    and creates a list of non-equivalent antiferromagnetic structures.
    Note can only take even number of moment such that the antiferromagnetic
    condition is sum(moments) = 0
    Returns [[0, 1, -1, 0, 0, ...]]
    """

    import numpy as np

    # Construct list with #up and downs
    start = np.ones(sum(arg))
    start[:len(start) // 2] = -1
    # If number of magnetic atoms is odd, cannot generate afms
    if sum(start) != 0:
        return []

    # Create all collinear afm structures
    from itertools import permutations
    afms = np.array(list(set(permutations(start))))

    # Remove inversion symmetric structures
    for i in range(len(afms) // 2):
        mask = np.all(-afms[i] == afms, axis=1)
        afms = afms[np.invert(mask), :]

    if return_raw is False:
        # Apply the afm structure to the magmom list
        afm_comps = np.ones((len(afms), len(moments)))
        for n in range(len(afms)):
            afm_comps[n, :] = moments
            afm_comps[n, arg] = moments[arg] * afms[n]
        return afm_comps
    else:
        # Apply the afm structure to the magmom list
        afm_comps = np.zeros((len(afms), len(moments)))
        for n in range(len(afms)):
            afm_comps[n, arg] = afms[n] * moments[arg]
        return afm_comps


def get_magmom_bands(arg, fm, collinear=False):
    afms = get_afms(fm, arg)
    if not collinear:
        magmoms = np.array([col_to_ncolx(fm)])
        for afm in afms:
            afm = col_to_ncolx(afm)
            magmoms = np.append(magmoms, [afm], axis=0)
    else:
        magmoms = [fm]
        for afm in afms:
            magmoms = np.append(magmoms, [afm], axis=0)

    return magmoms


def calc_tanEq(q_v, R_iv):
    return -np.angle(np.sum(np.exp(-1j * np.dot(R_iv, q_v))))


def nn(atoms, ref: int = 0):
    '''
    Output:
    R_a: np.array with lattice vectors to nearest neighbours
    npos_av: np.array with coordinates of nearest neighbours
    '''
    # Select reference atom
    pos_av = atoms.get_positions()[magnetic_atoms(atoms)]
    cell_cv = atoms.get_cell()

    allpos_av = []
    R_a = []
    for pos_v in pos_av:
        for n in [-1, 0, 1]:
            for m in [-1, 0, 1]:
                allpos_av.append(pos_v + n * cell_cv[0] + m * cell_cv[1])
                R_a.append(n * cell_cv[0] + m * cell_cv[1])

    allpos_av = np.asarray(allpos_av)
    R_a = np.asarray(R_a)

    R_nn = np.linalg.norm(allpos_av - pos_av[ref], axis=1)
    r = np.partition(R_nn, 2)[2]
    R_a = R_a[np.isclose(R_nn, r)]
    npos_av = allpos_av[np.isclose(R_nn, r)]
    return R_a, npos_av


def rotate_magmoms(atoms, magmoms, q_c, model='q.a'):
    from ase.dft.kpoints import kpoint_convert
    q_v = kpoint_convert(atoms.get_cell(), skpts_kc=[q_c])[0]
    if model == 'q.a':
        pos_av = atoms.get_positions()
        angles = np.dot(pos_av, q_v)
        # to_rotate = slice(None)
    if model == 'tan':
        # Automatic rotation of magnetic atoms require magnetic atoms
        if sum(magnetic_atoms(atoms)) == 0 or len(atoms) == 1:
            return np.array(magmoms)
        R_iv, _ = nn(atoms, ref=0)
        xi = calc_tanEq(q_v, R_iv)  # Arctan Equation
        angles = [xi, 0]
    to_rotate = true_magnetic_atoms(atoms, magmoms)

    moments_av = magmoms[to_rotate]
    for i, (mom_v, angle) in enumerate(zip(moments_av, angles)):
        Rz = R.from_euler('z', angle).as_matrix()
        moments_av[i] = Rz @ mom_v
    magmoms[to_rotate] = moments_av
    return np.array(magmoms)


def get_spiral_bandpath(atoms, qdens=None, qpts=None,
                        q_path=None, eps=None):
    # default q-sampling
    if qpts is None and qdens is None:
        qdens = 6.0
    if eps is None:
        eps = c2db_symmetry_eps

    if qdens is not None and qpts is not None:
        raise ValueError("Both q-density and q-points are provided")
    elif qpts is None:
        path = atoms.cell.bandpath(q_path, density=qdens,
                                   pbc=atoms.pbc, eps=eps)
    else:
        path = atoms.cell.bandpath(q_path, npoints=qpts,
                                   pbc=atoms.pbc, eps=eps)
    return path
