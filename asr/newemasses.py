from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import List
from gpaw import GPAW
from ase.units import Ha, Bohr
from asr.utils.gpw2eigs import gpw2eigs, calc2eigs
from asr.magnetic_anisotropy import get_spin_axis, get_spin_index
from asr.core import command, option, DictStr, ASRResult, prepare_result


class NoGapError(Exception):
    pass

class FittingError(Exception):
    pass

class PBCError(Exception):
    pass

class BT(Enum):
    vb = 'vb'
    cb = 'cb'
    

def check_pbc(gpw):
    calc = GPAW(gpw, txt=None)
    atoms = calc.get_atoms()

    ndim = atoms.pbc.sum()
    if ndim == 1:
        good = not pbc[0] and not pbc[1] and pbc[2]
        expected = '(False, False, True)'            
    elif ndim == 2:
        good = pbc[0] and pbc[1] and pbc[2]
        expected = '(True, True, False)'
    else:
        good = True

    if not good:
        raise PBCError(f'The effective mass calculation assumes PBC: {expected}') 

@command(module='asr.newemasses',
         requires=['gs.gpw', 'results-asr.magnetic_anisotropy.json'],
         dependencies=['asr.structureinfo',
                       'asr.magnetic_anisotropy',
                       'asr.gs'],
         creates=['em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw'])
@option('--gpwfname', type=str, help='GS fname')
@option('--erange1', type=float, help='First erange')
@option('--nkpts1', type=int, help='First nkpts')
@option('--erange2', type=float, help='Final erange')
@option('--nkpts2', type=int, help='Final nkpts')
@option('--soc', type=bool, help='Use SOC')
def refine(gpwfname: str = 'gs.gpw',
           erange1: float = 250e-3,
           nkpts1: int = 19,
           erange2: float = 10e-3,
           nkpts2: int = 15,
           soc: bool = True):
    # Final nonsc circle is larger than previously
    # so we can expand fit range if necessary
    # Write gpw files as before
    assert soc
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    from asr.magnetic_anisotropy import get_spin_axis
    import os.path

    check_pbc(gpwfname)

    # ! Convert from eV to Ha
    erange1 = erange1 / Ha
    erange2 = erange2 / Ha

    theta, phi = get_spin_axis()
    eigenvalues, efermi = gpw2eigs(gpw=gpwfname, soc=soc,
                                   theta=theta, phi=phi)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[np.newaxis]

    gap, skn1, skn2 = bandgap(eigenvalues=eigenvalues, efermi=efermi,
                        output=None)

    if not gap > 0:
        raise NoGapError('Gap was zero. efermi: {efermi}, theta: {theta}, phi: {phi}')

    for bt in BT:
        refined_name = get_name(soc=soc, bt=bt)
        if os.path.exists(refined_name):
            continue
        
        skn = skn1 if bt == BT.vb else skn2
        prelim_refine = preliminary_refine(gpw=gpwfname,
                                           e_skn=eigenvalues,
                                           efermi=efermi,
                                           theta=theta,
                                           phi=phi,
                                           skn=skn,
                                           erange=erange1, nkpts=nkpts1)

        nonsc_sphere(gpw=prelim_refine, savename=refined_name,
                     soc=soc, bt=bt, theta=theta, phi=phi,
                     erange=erange2, nkpts=nkpts2)





def preliminary_refine(gpw,
                       e_skn,
                       efermi,
                       theta,
                       phi,
                       skn,
                       erange, nkpts):
    #OBS QUESTION: soc not used here

    calc = GPAW(gpw, txt=None)
    atoms = calc.get_atoms()
    
    nkpts = nkpts + (1 - (nkpts % 2))
    # Ensure that we include the preliminary band extremum (BE)
    assert nkpts % 2 != 0

    cell_cv = atoms.get_cell()
    ndim = atoms.pbc.sum()
    ksphere = kptsinsphere(cell_cv, npoints=nkpts,
                           erange=erange, m=1.0,
                           dimensionality=ndim)

    
    k_kc = calc.get_bz_k_points()
    newk_kc = k_kc[skn[1]] + ksphere

    fname = '_refined'
    calc.set(kpts=newk_kc,
              symmetry='off',
              txt=fname + '.txt',
              fixdensity=True)

    atoms.get_potential_energy()
    
    calc.write(fname + '.gpw')

    return fname + '.gpw'


def kptsinsphere(cell_cv, npoints, erange, m, dimensionality):
    from ase.dft.kpoints import kpoint_convert

    a = np.linspace(-1, 1, npoints)
    X, Y, Z = np.meshgrid(a, a, a, indexing='ij')


    na = np.logical_and
    if dimensionality == 1:
        indices = na(Z**2 <= 1.0, na(X == 0, Y == 0))
    elif dimensionality == 2:
        indices = na(X**2 + Y**2 <= 1.0, Z == 0)
    else:
        indices = X**2 + Y**2 + Z**2 <= 1.0

    x, y, z = X[indices], Y[indices], Z[indices]

    kpts_kv = np.vstack([x, y, z]).T

    kr = np.sqrt(2.0 * m * erange)
    kpts_kv *= kr
    kpts_kv /= Bohr
    kpts_kc = kpoint_convert(cell_cv=cell_cv, ckpts_kv=kpts_kv)

    return kpts_kc


def nonsc_sphere(gpw, savename, soc, bt,
                 theta, phi,
                 erange, nkpts):
    # Do soc calc
    # Get best guess for location of minimum
    # Make nonsc sphere around
    from asr.utils.gpw2eigs import calc2eigs
    from ase.dft.bandgap import bandgap

    calc = GPAW(gpw, txt=None)
    atoms = calc.get_atoms()
    #OBS Presentation note: This should fix memory issue
    #OBS Can SOC calculation run in parallel?

    # Get BE location
    e_skn, efermi = calc2eigs(calc, soc=soc, theta=theta, phi=phi)
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]

    gap, skn1, skn2 = bandgap(eigenvalues=e_skn, efermi=efermi, output=None)

    if not gap > 0:
        raise NoGapError('Gap was zero in nonsc_sphere!')

    skn = skn1 if bt == BT.vb else skn2

    # Get kpts
    k_kc = calc.get_bz_k_points()
    cell_cv = atoms.get_cell()
    ndim = atoms.pbc.sum()
    nkpts = nkpts + (1 - nkpts % 2)
    assert nkpts % 2 != 0

    kcirc_kc = kptsinsphere(cell_cv, npoints=nkpts, erange=erange,
                            m=1.0, dimensionality=ndim)
    newk_kc = k_kc[skn[1]] + kcirc_kc

    calc.set(kpts=newk_kc,
             symmetry='off',
             txt=savename.replace('gpw', 'txt'),
             fixdensity=True)

    atoms.get_potential_energy()
    calc.write(savename)


@dataclass
class BandstructureData:
    kpts_kc: np.ndarray  # kpts in scaled coordinates
    kpts_kv: np.ndarray  # kpts in cartesian coordinates
    e_k: np.ndarray  # The energies in eV.
    # Should we keep it in eV? It is needed for plotting (eV)
    # and for validation/parabolicity (unclear what unit is best)
    spin_k: np.ndarray  # Spin-polarizations

    # Do we want to include extra bands? No that would introduce
    # a weird abstraction level-crossing

    # Error measures of the fit relative to the band over various energy ranges:
    maes: np.ndarray  # A list of (energy-range, mae) pairs
    mares: np.ndarray  # A list of (energy-range, mare) pairs


    def to_dict(self):
        return dict(kpts_kc=self.kpts_kc,
                    kpts_kv=self.kpts_kv,
                    e_k=self.e_k, spin_k=self.spin_k,
                    maes=self.maes, mares=self.mares)
    
    def from_dict(dct):
        return BandstructureData(**dct)

def make_bs_data(kpts_kc, kpts_kv, e_k, spin_k):
    return BandstructureData(kpts_kc, kpts_kv, e_k, spin_k, None, None)

@dataclass
class BandFit:
    k0_index: int  # Index into kpt-array around which to fit
    kpts_kv: np.ndarray  # Units: 1/Bohr
    eps_k: np.ndarray  # Units: Hartree
    bt: BT  # Bandtype
    band: int  # Keep index of band. Spin is always "0" because we always use SOC
    ndim: int  # Dimensionality of the system
    mass_n: np.ndarray  # List of effective masses
    dir_vn: np.ndarray  # Eigenvectors of the Hessian, i.e. directions of approx. constant curvature
    erange: float  # The erange used to make the fit
    fit_params: np.ndarray # The fitting params returned by numpy's lstsq

    bs_data: List[BandstructureData]
    # Convention: Indexing corresponds to indexing in mass array
    bs_erange: float
    bs_npoints: int

    
    def to_dict(self):
        dct = dict(k0_index=self.k0_index,
                   kpts_kv=self.kpts_kv,
                   eps_k=self.eps_k,
                   bt=self.bt.value,
                   band=self.band,
                   ndim=self.ndim,
                   mass_n=self.mass_n,
                   dir_vn=self.dir_vn,
                   erange=self.erange,
                   fit_params=self.fit_params,
                   bs_data=[bsd.to_dict() for bsd in self.bs_data],
                   bs_erange=self.bs_erange,
                   bs_npoints=self.bs_npoints)
        return dct

    def from_dict(dct):
        bt = dct.pop('bt')
        bt = BT[bt]
        bs_data = dct.pop('bs_data')
        bs_data = [BandstructureData.from_dict(bsd) for bsd in bs_data]
        return BandFit(**dct, bt=bt, bs_data=bs_data)


def make_bandfit(kpts_kv, eps_k, bt, band, ndim):
    assert bt in [BT.vb, BT.cb]
    k0_index = (np.argmin(eps_k) if bt == BT.cb
                else np.argmax(eps_k))

    return BandFit(k0_index, kpts_kv, eps_k, bt, band, ndim,
                   None, None, None, None, [], None, None)

    
def get_name(soc, bt):
    socstr = 'soc' if soc else 'nosoc'
    return f'em_circle_{bt.value}_{socstr}.gpw'


@command(module='asr.newemasses',
         requires=['em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw'],
         dependencies=['asr.newemasses@refine',
                       'asr.gs@calculate',
                       'asr.gs',
                       'asr.structureinfo',
                       'asr.magnetic_anisotropy'],
         creates=['fitdata.npy'])
@option('--gpwfilename', type=str, help='GS fname')
@option('--delta', type=float, help='delta')
def calculate_fits(gpwfilename: str = 'gs.gpw', delta: float = 0.1):
    # Identify relevant bands and BE locations
    # Separate data out into BandFit objects
    # For each band, perform fitting procedure
    # 

    # Raise error if gap is zero
    # check_gap(gpwfilename)

    bandfits = []
    for bt in BT:
        # Get the refined eigenvalues
        # refined_gpw_name = get_name(soc=True, bt=bt) + '.gpw'
        # calc = GPAW(refined_gpw_name, txt=None)
        calc = get_refined_calc(soc=True, bt=bt)
        # Then extract relevant bands and eigenvalues
        eps_ik, kpts_kv, band_i = get_bands(calc, delta=delta, bt=bt)
        ndim = calc.get_atoms().pbc.sum()

        # Put into BandFit object
        bandfits += [make_bandfit(kpts_kv, eps_k, bt, band, ndim)
                     for (band, eps_k) in zip(band_i, eps_ik)]

    # Then run fitting procedure    
    for bandit in bandfits:
        perform_fit(bandit) # Adds data to BandFit object

    result = convert_to_result(bandfits)
    
    np.save("fitdata.npy", result)
    # In new ASR we should just return the bandfits
    # return bandfits
    # or maybe result, if ASR cannot serialize arbitrary objs


def get_refined_calc(soc, bt):
    """Read calc produced by refine instruction."""
    assert soc == True  # Also type checks

    name = get_name(soc, bt)
    calc = GPAW(name, txt=None)

    return calc


def get_bands(calc, delta, bt):
    """Get bands from calc.

    Selects VB or CB depending on bt.
    Bands within delta are included.

    Converts everything to atomic units.

    Returns: eps_ik, kpts_kv, sn_i.
    i is here an index enumerating both spin and band.
    """
    from asr.utils.gpw2eigs import calc2eigs
    from asr.magnetic_anisotropy import get_spin_axis
    from ase.dft.bandgap import bandgap
    from ase.dft.kpoints import kpoint_convert
    from ase.units import Bohr, Ha

    theta, phi = get_spin_axis()
    e_skn, efermi = calc2eigs(calc, soc=True, theta=theta, phi=phi)
    assert e_skn.ndim == 2
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]

    gap, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn,
                                              efermi=efermi, output=None)
    if gap <= 0.0:
        raise NoGapError('Gap was zero inside calculate_fits!')
    
    s, k, n = (s1, k1, n1) if bt == BT.vb else (s2, k2, n2)
    
    e_ik, sn_i = get_be_indices(e_skn, s, k, n, bt, delta)
    kpts_kc = calc.get_bz_k_points()
    
    # Unit conversion
    # We the internal units to be atomic units!
    # Convert from eV (e_nk) and 1/Å (kpts) to atomic
    cell_cv = calc.get_atoms().get_cell()
    kpts_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=kpts_kc) * Bohr
    eps_ik = e_ik / Ha

    assert all(x[0] == 0 for x in sn_i)
    
    return eps_ik, kpts_kv, [x[1] for x in sn_i]
    

def get_be_indices(e_skn, s, k, n, bt, delta):
    """Get indices of bands close to BE.

    Get indices of bands within delta of band extremum (BE).

    The BE is assumed to be at s, k, n.

    bt indicates if it is a valence or conduction band.

    If VB we find bands below n, otherwise above."""
    be = e_skn[s, k, n]
    if bt == BT.vb:
        # All bands below VB and including VB
        b_sn = e_skn[:, k, :n+1]
        # Indices within delta
        bs, bn = np.where(b_sn >= be - delta)
    elif bt == BT.cb:
        # All bands above CB and including CB
        b_sn = e_skn[:, k, n:]
        # Indices within delta
        bs, bn = np.where(b_sn <= be + delta)
        # Need to adjust cb indices so they are
        # correct within the original eigenvalue
        # array
        bn += n
    else:
        raise ValueError(f'Incorrect bt: {bt}')

    sn_i = list(zip(bs, bn))
    e_ik = np.zeros((len(sn_i), e_skn.shape[1]))
    for i, (s, n) in enumerate(sn_i):
        e_ik[i, :] = e_skn[s, :, n]
    
    return e_ik, sn_i


def polynomial_fit(k0_index, kpts_kv, eps_k, erange, ndim):
    """Perform a fit around k0_index.

    The range of kpts that are used is determined
    by the erange parameter.
    """
    if erange is None:
        k_indices = np.ones(len(eps_k)).astype(bool)
    else:
        k_indices = np.abs(eps_k - eps_k[k0_index]) <= erange
    
    if k_indices.sum() < 1 + 2 * ndim:
        # Not enough points to do a reasonable fit 
        return [np.nan] * ndim, None, None

    k_kv = kpts_kv[k_indices, :]
    e_k = eps_k[k_indices]

    lstsq_solution, hessian, residuals = do_2nd_lstsq(k_kv, e_k)
    # This seems to be required for the mass determination
    # to be numerically stable enough:
    hessian[np.isclose(hessian, 0.0)] = 0.0

    curv_n, vec_vn = np.linalg.eigh(hessian)
    mass_n = []
    dir_nv = []
    for curv, vec_v in zip(curv_n, vec_vn.T):
        if not np.isnan(curv) and not np.allclose(curv, 0.0):
            mass_n.append(1.0 / curv)
            dir_nv.append(vec_v)
    mass_n = np.array(mass_n)
    dir_vn = np.array(dir_nv).T
            
    return mass_n, dir_vn, lstsq_solution


def do_2nd_lstsq(k_kv, e_k):
    kx_k = k_kv[:, 0]
    ky_k = k_kv[:, 1]
    kz_k = k_kv[:, 2]

    ones = np.ones(len(kx_k))

    A_kp = np.array([kx_k**2, ky_k**2, kz_k**2, kx_k * ky_k, kx_k * kz_k, ky_k * kz_k,
                     kx_k, ky_k, kz_k,
                     ones]).T

    sol, residuals, rank, singularvals = np.linalg.lstsq(A_kp, e_k, rcond=None)

    dxx = 2 * sol[0]
    dyy = 2 * sol[1]
    dzz = 2 * sol[2]
    dxy = sol[3]
    dxz = sol[4]
    dyz = sol[5]

    hessian = np.array([[dxx, dxy, dxz],
                        [dxy, dyy, dyz],
                        [dxz, dyz, dzz]])

    return sol, hessian, residuals
    

# Letting fitting function and eranges be input variables might make it
# easier to test?
# Also abstracts the actual logic of the function itself
# Makes it more modular, possible to change things easily
# Closer to a pure function/context-free function
def perform_fit(bandfit, fitting_fnc=polynomial_fit, eranges=[1e-3 / Ha, 5e-3 / Ha, None]):
    for erange in eranges:
        masses, dir_vn, fit_params = fitting_fnc(bandfit.k0_index,
                                                 bandfit.kpts_kv, bandfit.eps_k, erange,
                                                 bandfit.ndim) # Can this fail?

        expected_sign = -1 if bandfit.bt == BT.vb else 1
        if len(masses) == bandfit.ndim and all(np.sign(mass) == expected_sign for mass in masses):
            bandfit.erange = erange
            bandfit.fit_params = fit_params
            bandfit.mass_n = masses
            bandfit.dir_vn = dir_vn
            break
    else:
        band = bandfit.band
        raise FittingError(f'Could not construct any good fit for band {band}. Got masses: {masses}')


def convert_to_result(bandfits):
    """Convert a list of bandfits into intermediate result."""
    # Implementation should be trivial
    # Make a to_dict method on bandfits
    # Define ASRResult with list of bandfits
    # Maybe a recipe can only have one ASRResult,
    # so in that case use an intermediate result
    # We can easily refactor later.
    data = [bf.to_dict() for bf in bandfits]
    return data
    

@command(module='asr.newemasses',
         requires=['em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw',
                   'gs.gpw', 'results-asr.structureinfo.json',
                   'results-asr.gs.json',
                   'results-asr.magnetic_anisotropy.json',
                   'fitdata.npy'],
         dependencies=['asr.newemasses@calculate_fits',
                       'asr.newemasses@refine',
                       'asr.gs@calculate',
                       'asr.gs',
                       'asr.structureinfo',
                       'asr.magnetic_anisotropy'],
         creates=['bsdata.npy'])
@option('--fname', type=str, help='fname')
@option('--bs_erange', type=float, help='Energy in Ha')
@option('--bs_npoints', type=int, help='npts')
def calculate_bandstructures(fname: str = 'fitdata.npy',
                             bs_erange: float = 250e-3 / Ha,
                             bs_npoints: int = 91):
    # In new ASR this should just be a call to asr.emasses@calculate_fits
    # Or take bandfits as input?
    data = np.load(fname, allow_pickle=True)
    bandfits = [BandFit.from_dict(bf) for bf in data]

    # Get calc which has the density and is used to calculate
    # the bandstructure with fixdensity=True
    vb_calc = get_refined_calc(soc=True, bt=BT.vb)
    cb_calc = get_refined_calc(soc=True, bt=BT.cb)


    # Each BF has ndim directions for which we need to calculate the
    # bandstructures. 
    # So at top-level we want something like:
    for bf in bandfits:
        # ! This sets bandstructure data on BandFit object
        # ! This modifes calc so it probably cannot be used
        # ! later
        calc = vb_calc if bf.bt == BT.vb else cb_calc
        bf.bs_erange = bs_erange
        bf.bs_npoints = bs_npoints
        calc_bandstructure(bf, calc)

    # Now bandstructure is calculated and set
    # So we only have 3 logical steps here: load, read, calc?

    # return bandfits  # Do we need to return? Bit confused about how things will look with new ASR
    np.save('bsdata.npy', [bf.to_dict() for bf in bandfits])

    # Should be roughly equivalent to calculate_bs_along_emass_vecs
    # Should also do validation/parabolicity? Actually I would rather have it be a separate func.


def create_or_read_calc(bf, direction, calc):
    """Get eigenvalues in direction.
    
    Saves bandstructure calc in a file.
    Reads from file if file exists."""
    from pathlib import Path
    from gpaw.mpi import serial_comm

    # TODO Add bs_erange and bs_npoints here
    fname = f'em_bs_band={bf.band}_bt={bf.bt.value}_dir={direction}'
    fname = fname + f'_erange={bf.bs_erange}_npoints={bf.bs_npoints}.gpw'
    
    if not Path(fname).is_file():
        # ! Modifies calc, it probably cannot be used after
        calc_serial, k_kv = create_calc(bf, direction, fname, calc,
                                  bs_erange=bf.bs_erange, bs_npoints=bf.bs_npoints)
    else:
        from ase.dft.kpoints import kpoint_convert
        calc_serial = GPAW(fname, txt=None, communicator=serial_comm)
        k_kc = calc_serial.get_bz_k_points()
        cell_cv = calc_serial.get_atoms().get_cell()
        k_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=k_kc)
        

    return calc_serial, k_kv


def calc_bandstructure(bf: BandFit, calc,
                       createcalc_fn=create_or_read_calc,
                       eigscalc_fn=calc2eigs,
                       spinaxis_fn=get_spin_axis,
                       spinindex_fn=get_spin_index):
    # For each direction
    
    ##  Do a calc
    ###  If calc exists, read data
    ##  Set data 

    # return None

    for direction in range(bf.ndim):
        # This calculates or reads bandstructure along direction
        # It is a little ugly to return k_kv here, but it seems to
        # be the best way
        calc_serial, k_kv = createcalc_fn(bf, direction, calc)
        
        k_kc = calc_serial.get_bz_k_points()
        theta, phi = spinaxis_fn()
        e_km, _, s_kvm = eigscalc_fn(calc_serial, soc=True, return_spin=True,
                                   theta=theta, phi=phi)

        sz_km = s_kvm[:, spinindex_fn(), :]

        bsd = make_bs_data(k_kc, k_kv, e_km[:, bf.band], sz_km[:, bf.band])

        bf.bs_data.append(bsd)


def create_calc(bf, direction, fname, calc, bs_erange, bs_npoints):
    from asr.core import file_barrier
    from gpaw.mpi import serial_comm
    
    atoms = calc.get_atoms()
    cell_cv = atoms.get_cell()

    with file_barrier([fname]):
        k_kc, k_kv = get_kpts_for_bandstructure(bf, direction, bs_erange, bs_npoints,
                                                cell_cv, atoms.pbc)

        calc.set(kpts=k_kc, symmetry='off',
                 txt=fname.replace('gpw', 'txt'),
                 fixdensity=True)
        atoms.get_potential_energy()
        calc.write(fname)
        
    calc_serial = GPAW(fname, txt=None, communicator=serial_comm)

    return calc_serial, k_kv 


def get_kpts_for_bandstructure(bf, direction, bs_erange, bs_npoints, cell_cv, pbc):
    from ase.dft.kpoints import kpoint_convert

    kmax = np.sqrt(2.0 * abs(bf.mass_n[direction]) * bs_erange)
        
    kvec_v = bf.dir_vn[:, direction]

    # Stricly enforce 0 component along non-periodic directions
    # Sometimes fits have small non-zero components
    # Should this be done on k_kc?
    for i, pb in enumerate(pbc):
        if not pb:
            kvec_v[i] = 0.0

    k_kv = np.linspace(-1, 1, bs_npoints) * kmax * kvec_v.reshape(3, 1)
    k_kv = k_kv.T
    k_kv += bf.kpts_kv[bf.k0_index, :]
    k_kv /= Bohr
    assert k_kv.shape == (bs_npoints, 3)

    k_kc = kpoint_convert(cell_cv, ckpts_kv=k_kv)

    return k_kc, k_kv


@command(module='asr.newemasses',
         requires=['em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw',
                   'gs.gpw', 'results-asr.structureinfo.json',
                   'results-asr.gs.json',
                   'results-asr.magnetic_anisotropy.json',
                   'fitdata.npy',
                   'bsdata.npy'],
         dependencies=['asr.newemasses@calculate_bandstructures',
                       'asr.newemasses@calculate_fits',
                       'asr.newemasses@refine',
                       'asr.gs@calculate',
                       'asr.gs',
                       'asr.structureinfo',
                       'asr.magnetic_anisotropy'],
         creates=['paradata.npy'])
@option('--eranges', type=List[float], help='Eranges')
def calculate_parabolicities(eranges=[10e-3, 15e-3, 25e-3]):

    data = np.load('bsdata.npy', allow_pickle=True)
    bandfits = [BandFit.from_dict(d) for d in data]
    bandfits = calc_parabolicities(bandfits, eranges=eranges)
    np.save('paradata.npy', [bf.to_dict() for bf in bandfits])


def calc_parabolicities(bandfits: List[BandFit] = [],
                        eranges=[10e-3, 15e-3, 25e-3]):
    """Calculate MAE and MARE over energy ranges.

    bandfits are a list of bandfit-objects created by
    calculate_bandstructures (and other Instructions).
    
    eranges is a list of energy ranges over which to
    calculate the MAE and MARE.

    ! The units are eV.

    """    
    for bandfit in bandfits:
        for direction in range(bandfit.ndim):
            bsdata = bandfit.bs_data[direction]
            kpts_kv = bsdata.kpts_kv
            e_k = bsdata.e_k
            fit_e_k = get_model(bandfit.fit_params, kpts_kv)
            maes, mares = calc_errors(bandfit.bt, kpts_kv, e_k, fit_e_k, eranges)

            bsdata.maes = maes
            bsdata.mares = mares

    return bandfits


def calc_errors(bt, kpts_kv, e_k, fit_e_k, eranges):
    maes = np.zeros((len(eranges), 2))
    mares = np.zeros((len(eranges), 2))

    for i, erange in enumerate(eranges):
        reference_energy = np.min(e_k) if bt == BT.cb else np.max(e_k)
        indices = np.where(np.abs(e_k - reference_energy) < erange)

        mae = np.mean(np.abs(e_k[indices] - fit_e_k[indices]))
        mare = np.mean(np.abs((e_k[indices] - fit_e_k[indices])/e_k[indices]))

        maes[i, 0] = erange
        maes[i, 1] = mae
        mares[i, 0] = erange        
        mares[i, 1] = mare
    
    return maes, mares


def get_model(fit_params, kpts_kv):
    # ! Converts to eV
    kx_k = kpts_kv[:, 0]
    ky_k = kpts_kv[:, 1]
    kz_k = kpts_kv[:, 2]

    ones = np.ones(len(kx_k))

    A_kp = np.array([kx_k**2, ky_k**2, kz_k**2, kx_k * ky_k, kx_k * kz_k, ky_k * kz_k,
                     kx_k, ky_k, kz_k,
                     ones]).T

    return A_kp.dot(fit_params) * Ha


@command('asr.newemasses',
         requires=['em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw',
                   'gs.gpw', 'results-asr.structureinfo.json',
                   'results-asr.gs.json',
                   'results-asr.magnetic_anisotropy.json',
                   'fitdata.npy',
                   'bsdata.npy',
                   'paradata.npy'],
         dependencies=['asr.newemasses@calculate_parabolicities',
                       'asr.newemasses@calculate_bandstructures',
                       'asr.newemasses@calculate_fits',
                       'asr.newemasses@refine',
                       'asr.gs@calculate',
                       'asr.gs',
                       'asr.structureinfo',
                       'asr.magnetic_anisotropy'],
         creates=['finaldata.npy'])
def main():
    # refine()
    # subresults = calculate_fits()
    # subresults = calculate_bandstructures(subresults)
    # subresults = calculate_parabolicities(subresults)

    # result = make_asrresult(subresults)

    # return result
    data = np.load('paradata.npy', allow_pickle=True)
    bfs = [BandFit.from_dict(d) for d in data]
    np.save('finaldata.npy', [bf.to_dict() for bf in bfs])

def webpanel():
    raise NotImplementedError




"""
Are we missing anything?

We are not doing 3rd order fit. Is it necessary? Make a unit test where 2nd order will fail.
We are not doing weird checks on extremum type and BE location movement. Is this necessary?
We are not setting a lot of stuff that we currently have in the results. A lot of it can be calculated from fit_params, so maybe we don't need it?

Go through old code and double check that we have everything.
Go through old results and check if we have everything we want.


Next TODO: Implement get_bands. Exactly how data is structured will cascade and affect
everything.

"""

if __name__ == "__main__":
    main.cli()
