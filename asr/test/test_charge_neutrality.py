import pytest
import numpy as np
from .materials import std_test_materials
from asr.charge_neutrality import convert_concentration_units


@pytest.mark.parametrize('reduced', [True, False])
@pytest.mark.ci
def test_get_stoichiometry(reduced):
    from asr.charge_neutrality import get_stoichiometry

    materials = std_test_materials.copy()
    results = {'Si2': {'Si': 2},
               'BN': {'B': 1, 'N': 1},
               'Ag': {'Ag': 1},
               'Fe': {'Fe': 1}}
    for atom in materials:
        stoi = get_stoichiometry(atom, reduced=reduced)
        for el in stoi:
            ref = results[f'{atom.get_chemical_formula()}']
            if reduced and atom.get_chemical_formula() == 'Si2':
                assert stoi[el] == 1
            else:
                assert ref[el] == stoi[el]


@pytest.mark.parametrize('atoms', std_test_materials.copy())
@pytest.mark.ci
def test_get_element_list(atoms):
    from asr.charge_neutrality import get_element_list

    element_list = get_element_list(atoms)
    for i in range(len(atoms)):
        if atoms.get_chemical_formula() == 'Si2':
            assert atoms.symbols[i] == element_list[0]
            assert len(element_list) == 1
        else:
            assert atoms.symbols[i] == element_list[i]
            assert len(element_list) == len(atoms)


@pytest.mark.parametrize('hof', [-1, -2])
@pytest.mark.ci
def test_get_adjusted_chemical_potentials(hof):
    from asr.charge_neutrality import (get_adjusted_chemical_potentials,
                                       get_element_list)
    from .materials import BN

    atoms = BN.copy()
    element_list = get_element_list(atoms)

    for element in element_list:
        chempot = get_adjusted_chemical_potentials(atoms, hof, element)
        for symbol in chempot:
            if symbol == element:
                assert chempot[f'{symbol}'] == pytest.approx(hof)
            else:
                assert chempot[f'{symbol}'] == pytest.approx(0)


@pytest.mark.parametrize('energy', [0, 0.5, 1])
@pytest.mark.parametrize('efermi', [0, 0.5, 1])
@pytest.mark.ci
def test_fermi_dirac(energy, efermi):
    from asr.charge_neutrality import (fermi_dirac_electrons,
                                       fermi_dirac_holes)

    T = 300
    c_e = fermi_dirac_electrons(energy, efermi, T)
    c_h = fermi_dirac_holes(energy, efermi, T)

    if energy == efermi:
        assert c_e == pytest.approx(0.5)
        assert c_h == pytest.approx(0.5)
    elif energy > efermi:
        assert c_e > 0.5
        assert c_h < 0.5
    elif energy < efermi:
        assert c_e < 0.5
        assert c_h > 0.5


@pytest.mark.parametrize('zsize', np.arange(10, 20, 3))
@pytest.mark.parametrize('conc', [1e2, 0.001, 2])
@pytest.mark.ci
def test_convert_concentration_units(zsize, conc):
    from .materials import BN, Agchain

    ang_to_cm = 1. * 10 ** (-8)

    atoms = BN.copy()
    cell = atoms.get_cell()
    atoms.set_cell([cell[0], cell[1], [0, 0, zsize]])
    volume = atoms.get_volume() / zsize
    ref_conc = conc / (volume * (ang_to_cm ** 2))

    conc = convert_concentration_units(conc, atoms)

    assert ref_conc == pytest.approx(conc)

    try:
        convert_concentration_units(conc, Agchain)
        assert False
    except NotImplementedError:
        assert True


@pytest.mark.parametrize('p0', [1e3, 1e-4])
@pytest.mark.parametrize('n0', [1e3, 2.3e2])
@pytest.mark.ci
def test_calculate_delta(p0, n0):
    from asr.charge_neutrality import (calculate_delta,
                                       check_delta_zero)

    conc_list = [1e-2, 2e-2, 2e-2]
    charge_list = [0, 1, -1]

    delta = calculate_delta(conc_list, charge_list, n0, p0)
    ref_delta = n0 - p0

    assert delta == pytest.approx(ref_delta)
    if delta == pytest.approx(0):
        assert check_delta_zero(delta, conc_list, n0, p0)
    else:
        assert not check_delta_zero(delta, conc_list, n0, p0)


@pytest.mark.parametrize('eform', [-1, 0, 1])
@pytest.mark.ci
def test_calculate_defect_concentration(eform):
    from asr.charge_neutrality import calculate_defect_concentration

    conc = calculate_defect_concentration(eform, 1, 1, 300)
    if eform == 0:
        assert conc == pytest.approx(1)
    elif eform < 0:
        assert conc > 1
    elif eform > 0:
        assert conc < 1


@pytest.mark.parametrize('gap', np.arange(0, 1.1, 0.5))
@pytest.mark.ci
def test_initialize_scf_loop(gap):
    from asr.charge_neutrality import initialize_scf_loop

    E, d, i, maxsteps, E_step, epsilon, converged = initialize_scf_loop(gap)

    assert E == pytest.approx(0)
    assert d == pytest.approx(1)
    assert i == pytest.approx(0)
    assert maxsteps == pytest.approx(1000)
    assert E_step == pytest.approx(gap / 10.)
    assert epsilon == pytest.approx(1e-12)
    assert not converged


@pytest.mark.parametrize('concentration', [1e-12, 3e-12, 1])
@pytest.mark.parametrize('delta', [1e-25, 1e-13, 1e-5])
@pytest.mark.parametrize('E_step', [1e-13, 1e-11])
@pytest.mark.ci
def test_check_convergence(concentration, delta, E_step):
    from asr.charge_neutrality import check_convergence, check_delta_zero

    n0 = 1
    p0 = 1
    epsilon = 1e-12
    conc_list = [n0 + p0 + concentration]
    delta_zero = check_delta_zero(delta, conc_list, n0, p0)
    if delta < concentration * 1e-12:
        ref_delta_zero = True
    else:
        ref_delta_zero = False
    assert delta_zero == ref_delta_zero

    convergence = check_convergence(delta, conc_list, n0, p0, E_step, epsilon)

    if E_step < epsilon or delta_zero:
        assert convergence
    else:
        assert not convergence


@pytest.mark.parametrize('old', [1, 0.5])
@pytest.mark.parametrize('new', [1, 0.5])
@pytest.mark.parametrize('d', [-1, 1])
@pytest.mark.ci
def test_update_scf_parameters(old, new, d):
    from asr.charge_neutrality import update_scf_parameters

    E_step = 10
    newstep, newd, olddelta = update_scf_parameters(old, new,
                                                    E_step, d)

    if new > old:
        assert newstep == pytest.approx(E_step / 10.)
        assert newd == pytest.approx(d * -1)
    else:
        assert newstep == pytest.approx(E_step)
        assert newd == pytest.approx(d)
    assert olddelta == pytest.approx(new)


@pytest.mark.parametrize('gap', np.arange(0.1, 1.01, 0.1))
@pytest.mark.ci
def test_get_dopability_type(gap):
    from asr.charge_neutrality import get_dopability_type
    energies = np.arange(0, gap, gap / 10.)

    for energy in energies:
        dopability = get_dopability_type(energy, gap)
        if energy < 0.25 * gap:
            assert dopability == 'p-type'
        elif energy > 0.75 * gap:
            assert dopability == 'n-type'
        else:
            assert dopability == 'intrinsic'


@pytest.mark.parametrize('energy', [0, 0.5])
@pytest.mark.parametrize('size', [0, 0.1])
@pytest.mark.parametrize('d', [-1, 1])
@pytest.mark.ci
def test_get_new_sample_point(energy, size, d):
    from asr.charge_neutrality import get_new_sample_point
    assert get_new_sample_point(energy, size, d) == pytest.approx(
        energy + size * d)
