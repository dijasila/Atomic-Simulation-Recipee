import pytest
from ase.build import bulk
from asr.structureinfo import main


@pytest.mark.ci
def test_cod_id(asr_tmpdir):
    atoms = bulk('Si')
    atoms.write('structure.json')
    result = main()
    assert result['spacegroup'] == 'Fd-3m'
