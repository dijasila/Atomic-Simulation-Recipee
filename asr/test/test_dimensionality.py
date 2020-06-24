from asr.dimensionality import main as dimensionality


def test_dimensionality(asr_tmpdir, test_material):
    nd = sum(test_material.pbc)

    results = dimensionality(test_material)

    interval = results['k_intervals'][0]
    assert interval['dimtype'] == f'{nd}D'


def test_dimensionality_cli(asr_tmpdir, test_material):
    nd = sum(test_material.pbc)
    test_material.write('structure.json')
    results = dimensionality.cli(args=[])

    interval = results['k_intervals'][0]
    assert interval['dimtype'] == f'{nd}D'
