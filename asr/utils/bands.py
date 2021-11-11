import numpy as np
from typing import Union
from gpaw import GPAW


def bs_from_gpw(fname):
    from gpaw import GPAW
    calc = GPAW(fname, txt=None)
    return calc.band_structure()


def bs_from_json(fname, soc):
    from ase.io.jsonio import read_json
    from ase.spectrum.band_structure import BandStructure
    read = read_json(fname)
    if isinstance(read, BandStructure):
        return read
    if soc:
        evs = read['eigenvalues_soc'] - read['evac']
        bs = BandStructure(read['path'],
                           evs[np.newaxis],
                           read['efermi_soc'] - read['evac'])
    else:
        evs = read['eigenvalues_nosoc'] - read['evac']
        bs = BandStructure(read['path'],
                           evs[np.newaxis],
                           read['efermi_nosoc'] - read['evac'])
    return bs


def calculate_evac(calc):
    '''Obtain vacuum level from a GPAW calculator'''
    evac = np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]
    return evac


def get_all_eigenvalues(calc, all_bz: bool = True):
    if all_bz:
        bzmap = calc.get_bz_to_ibz_map()
        bz = calc.get_bz_k_points()
        return bz, np.array([calc.get_eigenvalues(kpt=i) for i in bzmap])
    else:
        bz = calc.get_ibz_k_points()
        return bz, np.array([calc.get_eigenvalues(kpt=i) for i in range(len(bz))])


'''
def find_band_extrema(calc):
    kpts, cb, vb = get_cb_vb_surface(calc, all_bz=True)
    bzmap = calc.get_bz_to_ibz_map()
    kx = kpts[:, 0]
    ky = kpts[:, 1]
    uniq_kx, counts_x = np.unique(kpts[:, 0], return_counts=True)
    splitter = []
    indx = 0
    for i, c in enumerate(counts_x):
        indx += c
        splitter.append(indx)
    kpt_indexes_split = np.split(list(range(len(kpts))), splitter[:-1])
    print(kpt_indexes_split)
    kpts_split = np.split(kpts, splitter[:-1])
    for kpts, indexes in zip(kpts_split, kpt_indexes_split):
'''


def get_band_edges(calc):
    from ase.dft.bandgap import bandgap
    gap, vbm_k, cbm_k = bandgap(calc)
    vbm = calc.get_eigenvalues(kpt=vbm_k[1])[vbm_k[2]]
    cbm = calc.get_eigenvalues(kpt=cbm_k[1])[cbm_k[2]]
    return (vbm, cbm)


def get_gw_soc(dft, gw, nbands: int=10):
    import pickle
    from asr.utils import fermi_level
    from ase.dft.bandgap import bandgap
    from gpaw.spinorbit import soc_eigenstates

    gwresults = pickle.load(open(gw, 'rb'))
    qp = gwresults['qp'][0]

    lb, ub = max(dft.wfs.nvalence // 2 - nbands // 2 , 0),        \
                 dft.wfs.nvalence // 2 + nbands // 2

    soc = soc_eigenstates(dft, eigenvalues=qp[np.newaxis],
                          n1 = lb, n2 = ub)

    qp_soc = soc.eigenvalues()
    efermi_soc = fermi_level(dft, eigenvalues=qp_soc[np.newaxis],
                             nelectrons=(dft.get_number_of_electrons() - 2 * lb),
                             nspins=2)

    return qp_soc, efermi_soc


def find_local_band_extrema(calc):
    '''Find local extrema for both the conduction and valence bands.'''

    def split_kpts(kpts, column):
        kpts_tmp = np.array([tuple(i) for i in kpts], 
                            dtype = np.dtype([('x', float), ('y', float), ('z', float)]))
    
        if column == 0:
            order = ('x', 'y', 'z')
        if column == 1:
            order = ('y', 'x', 'z')
    
        # Sorting kpts along x/y direction
        sorter = np.argsort(kpts_tmp, order=order)
        kpts_sorted = kpts[sorter]
    
        # Grouping indexes of kpoints that share same kx/ky component
        uniq, counts = np.unique(kpts_sorted[:, column], return_counts=True)
        splitter = []
        indx = 0
        # This is necessary to relate np.unique to np.split
        for i, c in enumerate(counts):
            indx += c
            splitter.append(indx)
        return np.split(sorter, splitter[:-1])

    '''
    def find_extrema_bandwise(kpts, band, indexes):
        split_indexes_x = split_kpts(kpts, 1)
        split_indexes_x = split_kpts(kpts, 0)
        for index_group in indexes:
            group_energies = energies[index_group]
    '''

    def find_local_extrema_1D(energies, index_group):
        group_energies = energies[index_group] 
        maxima = []
        minima = []

        # Check if the first and/or last points are local extrema
        if group_energies[0] > group_energies[1]:
            maxima.append(index_group[0])
        if group_energies[0] < group_energies[1]:
            minima.append(index_group[0])

        if group_energies[-1] > group_energies[-2]:
            maxima.append(index_group[-1])
        if group_energies[-1] < group_energies[-2]:
            minima.append(index_group[-1])

        # Search for extrema among the other points
        for i in np.arange(1, len(group_energies) - 1):
            if group_energies[i] > group_energies[i-1] and group_energies[i] > group_energies[i+1]:
                maxima.append(index_group[i])
            if group_energies[i] < group_energies[i-1] and group_energies[i] < group_energies[i+1]:
                minima.append(index_group[i])
        
        return maxima, minima



    kpts, cb, vb = get_cb_vb_surface(calc, all_bz=True)
    split_indexes_x = split_kpts(kpts, 0)
    split_indexes_y = split_kpts(kpts, 1)

    maxima_x = []
    minima_x = []
    maxima_y = []
    minima_y = []
    for index_group in split_indexes_x:
        maxima_x += find_local_extrema_1D(vb, index_group)[0]
        minima_x += find_local_extrema_1D(cb, index_group)[1]
    for index_group in split_indexes_y:
        maxima_y += find_local_extrema_1D(vb, index_group)[0]
        minima_y += find_local_extrema_1D(cb, index_group)[1]

    print(maxima_x)
    print(maxima_y)
    print(set(maxima_x) & set(maxima_y))

        


    

    '''
    kpts_tmp = np.array([tuple(i) for i in kpts], dtype = np.dtype([('0', float), ('1', float), ('2', float)]))
    sort_x = np.argsort(kpts_tmp, order=('0', '1', '2'))
    sort_y = np.argsort(kpts_tmp, order=('2', '1', '0'))
    print(kpts[sort_x])
    print(kpts[sort_y])
    '''


    '''
    for kx in kx_vals
    sorted_kx = sorted(bz, key = lambda x: x[0])
    sorted_ky = sorted(bz, key = lambda x: x[1])
    print(sorted_ky)
    '''


def get_cb_vb_surface(calc: Union[GPAW, None] = None, ef = None, eigenvalues = None, kpts = None, all_bz: bool = True, soc: bool = False):
    '''Returns the surface formed by the edge states of CB and VB
       calculated on a 2D k-point grid
    '''
    if calc:
        ef = calc.get_fermi_level()
        if all_bz:
            kpts, eigenvalues = get_all_eigenvalues(calc, all_bz=True)
        else:
            kpts, eigenvalues = get_all_eigenvalues(calc, all_bz=False)

    if not isinstance(eigenvalues, np.ndarray):
        eigenvalues = np.asarray(eigenvalues)

    cb = []
    vb = []
    for ev in eigenvalues:
        cbi = ev[ev - ef > 0]
        vbi = ev[ev - ef < 0]
        cb.append(cbi.min())
        vb.append(vbi.max())

    return kpts, np.asarray(cb), np.asarray(vb)


def plot_cb_vb_surface(calc, all_bz: bool = True, title: str = '', soc: bool = False, mark_edges: bool = True):
    '''shows the brillouin zone in a 3D plot'''
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    bz, cb, vb = get_cb_vb_surface(calc, all_bz=all_bz, soc=soc)
    bz = bz[:, 0:2]
    vbm_pos = np.argmax(vb)
    cbm_pos = np.argmin(cb)
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_trisurf(bz[:,0], bz[:,1], cb, cmap='winter')
    ax.plot_trisurf(bz[:,0], bz[:,1], vb, cmap='summer')
    if mark_edges:
        ax.plot(bz[vbm_pos, 0], bz[vbm_pos, 1], vb[vbm_pos], marker='o', zorder=3)
        ax.plot(bz[cbm_pos, 0], bz[cbm_pos, 1], cb[cbm_pos], marker='o', zorder=4)

    ax.set_title(title)
    ax.set_xlabel('$k_x$', labelpad=20)
    ax.set_ylabel('$k_y$', labelpad=20)
    ax.set_zlabel('Energy (eV)', labelpad=20)
    plt.show()


def direct_gap(calc):
    '''Obtain lowest direct transition on a 2D k-point grid'''
    cb, vb = get_cb_vb_surface(calc)
    dir_gaps = cb - vb
    return dir_gaps.min()


def is_gap_direct(calc):
    '''Return True if the gap is direct'''
    ibz = calc.get_ibz_k_points()
    cb, vb = get_cb_vb_surface(calc)
    cbm_k = ibz[np.argmin(cb)]
    vbm_k = ibz[np.argmax(vb)]
    return np.allclose(cbm_k, vbm_k)


def multiplot(*toplot,
              reference=None,
              ylim=None,
              title=None,
              xtitle=None,
              ytitle=None,
              labels=None,
              hlines=None,
              styles=None,
              colors=None,
              fermiline=True,
              customticks=None,
              show=True,
              legend=True,
              soc=False,
              fontsize1=24,
              fontsize2=22,
              loc='best'):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import warnings
    warnings.filterwarnings('ignore')

    if not colors:
        colors = [cm.Set2(i) for i in range(len(toplot))]

    ax = plt.figure(figsize=(12, 9)).add_subplot(111)

    items = []
    for tp in toplot:
        if isinstance(tp, str):
            items.append(Bands(tp, soc))
        elif isinstance(tp, Bands):
            items.append(tp)
        else:
            msg = 'Please provide either Bands objects or filenames'
            raise ValueError(msg)

    kpts, specpts, lbls = items[0].get_kpts_and_labels(normalize=True)
    for spec in specpts[1:-1]:
        ax.axvline(spec, color='#bbbbbb')
    if customticks:
        lbls = list(customticks)
    ticks = [lab.replace('G', r'$\Gamma$') for lab in lbls]
    ax.set_xticklabels(ticks, fontsize=fontsize2)
    ax.set_xticks(specpts)
    ax.set_xlim([kpts[0], kpts[-1]])

    allfermi = []

    for (index, item), color in zip(enumerate(items), colors):
        if reference == "evac":
            ref = item.calculate_evac()
            if not ytitle:
                ytitle = r'$\mathrm{E-E_{vac}}$'
        elif reference == "ef":
            '''Useful only for plotting single bandstructure'''
            ref = item.get_efermi()
        elif isinstance(reference, float):
            ref = reference
        else:
            ref = 0.0
            ytitle = r'$\mathrm{E-E_{vac}}$'

        energies = item.get_energies() - ref
        efermi = item.get_efermi() - ref
        allfermi.append(efermi)

        try:
            lbl = labels[index]
        except (TypeError, IndexError):
            lbl = item._basename
        try:
            style = styles[index]
        except (TypeError, IndexError):
            color = colors[index]
            style = dict(ls='-', color=color, lw=2.0)

        kpts, _, _ = item.get_kpts_and_labels(normalize=True)
        ax.plot(kpts, energies[0], **style, label=lbl)
        for band in energies[1:]:
            ax.plot(kpts, band, **style)
        if fermiline:
            ax.axhline(efermi, color='#bbbbbb', ls='--', lw=2.0)

    if hlines:
        for val in hlines:
            ax.axhline(val, color='#bbbbbb', ls='--', lw=2.0)

    if title:
        plt.title(title, pad=10, fontsize=fontsize1)
    if not ylim:
        ylim = [min(allfermi) - 4, max(allfermi) + 4]

    ax.set_ylim(ylim)
    ax.set_xlabel(xtitle, fontsize=fontsize1, labelpad=8)
    ax.set_ylabel(ytitle, fontsize=fontsize1, labelpad=8)
    plt.yticks(fontsize=fontsize2)
    ax.xaxis.set_tick_params(width=3, length=10)
    ax.yaxis.set_tick_params(width=3, length=10)
    plt.setp(ax.spines.values(), linewidth=3)

    if legend:
        plt.legend(loc=loc, fontsize=fontsize2 - 2)

    if show:
        plt.show()


class Bands:
    def __init__(self,
                 filename,
                 soc=False,
                 gs='results-asr.gs.json'):

        from os.path import splitext
        self.filename = filename
        self._basename, self._ext = splitext(filename)

        print(f'Reading stuff from file {filename}...')

        if self._ext == '.gpw':
            if soc:
                raise NotImplementedError('SOC not implemented yet for bandstructures from .gpw files')
            self.bandstructure = bs_from_gpw(self.filename)

        if self._ext == '.json':
            self.bandstructure = bs_from_json(self.filename, soc)

        '''
        if gw:
            gsdct = read_json(gs)
            gsdata = gsdct["kwargs"]["data"]
            evac = gsdata["evac"]
            dct = read_json(self.filename)
            data = dct["kwargs"]["data"]
            path = data['bandstructure']['path']
            ef = data['efermi_gw_soc'] - evac1
            e1 = data1['bandstructure']['e_int_mk'] - evac1
            vbm1 = data1["vbm_gw"] - evac1
            cbm1 = data1["cbm_gw"] - evac1
            edg1 = get_edges(e1, ef1)
            x1, X1, labels1 = path1.get_linear_kpoint_axis()
        '''

        print("Done")

    def get_energies(self):
        return self.bandstructure.energies.T

    def get_efermi(self):
        return self.bandstructure.reference

    def get_kpts_and_labels(self, normalize=False):
        x, X, lbls = self.bandstructure.get_labels()
        if normalize:
            delta = x[-1] - x[0]
            return (x / delta, X / delta, lbls)
        else:
            return (x, X, lbls)

    def get_evac(self):
        errmsg = 'A .gpw file is required!'
        assert (self._ext == '.gpw'), errmsg
        evac = calculate_evac(self.filename)
        self.evac = evac
        return evac

    def calculate_SOC(self):
        from gpaw.spinorbit import soc_eigenstates
        from gpaw import GPAW
        errmsg = 'A .gpw file is required!'
        assert (self._ext == '.gpw'), errmsg
        calc = GPAW(self.filename, txt='-')
        spin = soc_eigenstates(calc)
        esoc = spin.eigenvalues().T
        efsoc = spin.fermi_level
        self.bandstructure.energies_soc = esoc
        self.bandstructure.reference_soc = efsoc
        return esoc, efsoc

    def plot(self, *args, **kwargs):
        multiplot(self, *args, **kwargs)

    def dump_to_json(self, filename=None):
        from ase.io.jsonio import write_json
        if not filename:
            filename = f'{self._basename}.json'
        dct = self.bandstructure.__dict__.copy()
        try:
            dct['path'] = dct.pop('_path')
            dct['energies'] = dct.pop('_energies')
            dct['reference'] = dct.pop('_reference')
        except KeyError:
            pass
        write_json(filename, dct)

    def get_band_edges(self, reference=0.0):
        en = self.get_energies()
        ef = self.get_efermi()
        allcb = en[(en - ef) > 0]
        allvb = en[(en - ef) < 0]
        cbm = allcb.min()
        vbm = allvb.max()
        if reference == 'evac':
            try:
                ref = self.bandstructure.evac
            except AttributeError:
                print('you have to calculate the vacuum level first!')
                return 0
        elif reference == 'ef':
            ref = self.bandstructure.reference
        elif isinstance(reference, float):
            ref = reference
        return vbm - ref, cbm - ref

    def get_band_edges_at_kpt(self, kpt):
        import numpy as np
        en_T = self.get_energies().T[0]
        ef = self.get_efermi()
        x, X, labels = self.get_kpts_and_labels()
        spec = X[labels.index(kpt)]
        indx_spec = np.where(x == spec)[0][0]
        en_kpt = en_T[indx_spec]
        en_vb = en_kpt[(en_kpt - ef) < 0]
        en_cb = en_kpt[(en_kpt - ef) > 0]
        return en_vb.max(), en_cb.min()

    def get_homo_lumo_gap(self, reference=0.0):
        vbm, cbm = self.get_band_edges(reference=reference)
        return cbm - vbm

    def get_direct_gap(self):
        en_T = self.get_energies().T[0] - self.get_efermi()
        dyn_gap = np.zeros(en_T.shape[0])
        for i, en in enumerate(en_T):
            lumo = en[en > 0].min()
            homo = en[en < 0].max()
            dyn_gap[i] = lumo - homo
        return dyn_gap.min()

    def get_transition(self, kpt1, kpt2):
        edges1 = self.get_band_edges_at_kpt(kpt1)
        edges2 = self.get_band_edges_at_kpt(kpt2)
        return max(edges2) - min(edges1)
