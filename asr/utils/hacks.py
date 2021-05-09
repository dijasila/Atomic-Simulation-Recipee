def gs_xcname_from_row(row):
    # Remove this and use RowInfo
    return RowInfo(row).gs_xcname()


class RowInfo:
    def __init__(self, row):
        self.row = row

    @property
    def gsdata(self):
        return self.row.data['results-asr.gs@calculate.json']

    def gs_xcname(self):
        data = self.gsdata
        if not hasattr(data, 'metadata'):
            # Old (?) compatibility hack
            return 'PBE'
        params = data.metadata.params
        if 'calculator' not in params:
            # What are the rules for when this piece of data exists?
            # Presumably the calculation used ASR defaults.
            return 'PBE'
        # If the parameters are present, but were not set, we are using
        # GPAW's default which is LDA.
        return params['calculator'].get('xc', 'LDA')

    def have_evac(self):
        return self.get_evac() is not None

    def get_evac(self, default=None):
        return self.row.get('evac', default)

    def evac_or_efermi(self):
        # We should probably be getting this data from GS results, not row
        evac = self.get_evac()
        if evac is not None:
            return EnergyReference('evac', evac, 'vacuum level', 'vac')

        efermi = self.row.get('efermi')
        return EnergyReference('efermi', efermi, 'Fermi level', 'F')


class EnergyReference:
    def __init__(self, key, value, prose_name, abbreviation):
        self.key = key
        self.value = value
        self.prose_name = prose_name
        self.abbreviation = abbreviation

    def mpl_plotlabel(self):
        return rf'$E - E_\mathrm{{{self.abbreviation}}}$ [eV]'
