"""Module for determining magnetic state."""
from asr.core import command


def get_magstate(calc):
    """Determine the magstate of calc."""
    magmoms = calc.get_property('magmoms', allow_calculation=False)

    if abs(magmoms).max() < 0.1:
        return 'nm'

    if (abs(magmoms).max() >= 0.1 and
       abs(magmoms.max() - magmoms.min()) < 0.02):
        return 'fm'

    return 'afm'


def webpanel(row, key_descriptions):
    """Webpanel for magnetic state."""
    rows = [['Magnetic state', row.magstate]]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Electronic properties', ''],
                             'rows': rows}]],
               'sort': 0}
    return [summary]


@command('asr.magstate',
         requires=['gs.gpw'],
         webpanel=webpanel,
         dependencies=['asr.gs@calculate'])
def main():
    """Determine magnetic state."""
    from gpaw import GPAW
    calc = GPAW('gs.gpw', txt=None)
    magstate = get_magstate(calc)

    results = {'magstate': magstate.upper(),
               'is_magnetic': magstate != 'nm'}

    return results


if __name__ == '__main__':
    main.cli()
