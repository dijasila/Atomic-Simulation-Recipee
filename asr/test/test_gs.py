import pytest
from pytest import approx
from .conftest import test_materials, get_webcontent


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
@pytest.mark.parametrize("gap", [0, 1])
@pytest.mark.parametrize("fermi_level", [0.5, 1.5])
def test_gs(separate_folder, usemocks, atoms, gap, fermi_level):
    from gpaw import GPAW as GPAWMOCK
    GPAWMOCK.set_property(gap=gap, fermi_level=fermi_level)

    from asr.gs import calculate, main
    from ase.io import write

    write('structure.json', atoms)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 2, "gamma": True},
        },
    )

    results = main()
    assert results.get("gaps_nosoc").get("efermi") == approx(fermi_level)
    assert results.get("efermi") == approx(fermi_level)
    if gap >= fermi_level:
        assert results.get("gap") == approx(gap)
    else:
        assert results.get("gap") == approx(0)

    content = get_webcontent('database.db')
    resultgap = results.get("gap")
    assert f"<td>Bandgap</td><td>{resultgap:0.2f}eV</td>" in content, content
    assert f"<td>Fermilevel</td><td>{fermi_level:0.3f}eV</td>" in \
        content, content
