import pytest
import numpy as np


@pytest.mark.parametrize('ef', np.arange(-5, 3, 2))
@pytest.mark.parametrize('emin', [-0.8, -0.2, 1])
@pytest.mark.parametrize('emax', [0.13, 0.5, -0.1])
@pytest.mark.ci
def test_return_erange_states(ef, emin, emax):
    from asr.get_wfs import return_erange_states
    import numpy as np

    erange = (emin, emax)
    evs = np.arange(-5, 3, 0.1)
    ref_evs = evs - ef

    states = return_erange_states(evs, ef, erange)
    ref_states = np.where((erange[1] > ref_evs)
                          & (ref_evs > erange[0]))[0]

    assert all(states == ref_states)

    # for inverted energy range there should be no
    # states returned
    if emin > emax:
        assert len(states) == 0


@pytest.mark.parametrize('gap', np.arange(0, 2.01, 20))
@pytest.mark.ci
def test_get_above_below(gap):
    from asr.get_wfs import get_above_below

    evs = np.arange(-5, 5.5, 20)
    ef = 0
    vbm = ef - gap / 2.
    cbm = ef + gap / 2.

    above_below = get_above_below(evs, ef, vbm, cbm)
    if gap >= 1:
        ref = (True, True)
    else:
        ref = (False, False)

    assert above_below[0] == ref[0]
    assert above_below[1] == ref[1]


@pytest.mark.parametrize('formula', ['MoS2', 'MoSe2', 'MoTe2'])
@pytest.mark.ci
def test_get_reference_index(formula):
    from asr.get_wfs import get_reference_index
    from ase.build import mx2

    atoms = mx2(formula)
    atoms = atoms.repeat((3, 3, 1))
    ref_index = get_reference_index(0, atoms)

    assert (ref_index == 15 or ref_index == 10)
