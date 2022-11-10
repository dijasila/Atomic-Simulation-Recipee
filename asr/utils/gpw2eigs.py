

def calc2eigs(calc, soc=True,
              return_spin=False,
              theta=0, phi=0, symmetry_tolerance=1e-7,
              width=None):
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc
    from ase.parallel import broadcast
    from .symmetry import restrict_spin_projection_2d
    from .calculator_utils import get_eigenvalues
    from .symmetry import _atoms2symmetry_gpaw

    dct = None
    bands = range(calc.get_number_of_bands())
    eps_nosoc_skn = get_eigenvalues(calc)[..., bands]
    efermi_nosoc = calc.get_fermi_level()
    dct = {'eps_nosoc_skn': eps_nosoc_skn,
           'efermi_nosoc': efermi_nosoc}
    if soc:
        occcalc = create_occ_calc({'width': 0.0})
        soc = soc_eigenstates(calc,
                              n1=bands[0],
                              n2=bands[-1] + 1,
                              theta=theta,
                              phi=phi,
                              occcalc=occcalc)
        eps_km = soc.eigenvalues()
        s_kvm = soc.spin_projections().transpose((0, 2, 1))
        efermi = soc.fermi_level

        symmetry = _atoms2symmetry_gpaw(calc.atoms,
                                        tolerance=symmetry_tolerance)
        # For magnetic systems we know some more about the spin
        # projections
        if calc.get_number_of_spins() == 1:
            # Inversion + time reversal symmetry forces degenerates spins
            if symmetry.has_inversion:
                s_kvm[:] = 0.0
            else:
                # For 2D we try to find materials where
                # spins are restricted to inplane spins
                if sum(calc.atoms.pbc) == 2:
                    bzk_kc = calc.get_bz_k_points()
                    for k, kpt in enumerate(bzk_kc):
                        s_vm = restrict_spin_projection_2d(kpt,
                                                           symmetry.op_scc,
                                                           s_kvm[k])
                        s_kvm[k] = s_vm
        dct.update({'eps_km': eps_km,
                    'efermi': efermi,
                    's_kvm': s_kvm})

    dct = broadcast(dct, root=0, comm=calc.world)
    if soc is None:
        raise NotImplementedError('soc=None is not implemented')

    if soc:
        out = (dct['eps_km'], dct['efermi'], dct['s_kvm'])
        if not return_spin:
            out = out[:2]
        return out
    else:
        if not return_spin:
            return dct['eps_nosoc_skn'], dct['efermi_nosoc']
        return dct['eps_nosoc_skn'], dct['efermi_nosoc'], dct['s_kvm']


def gpw2eigs(gpw, soc=True, return_spin=False,
             theta=0, phi=0, symmetry_tolerance=1e-7):
    """Give the eigenvalues w or w/o spinorbit coupling and the
    corresponding fermi energy.

    Parameters
    ----------
        gpw (str): gpw filename
        soc : None, bool
            use spinorbit coupling if None it returns both w and w/o
        bands : slice, list of ints or None
            None gives parameters.convergence.bands if possible else all bands

    Returns
    -------
    dict or e_skn, efermi
        containg eigenvalues and fermi levels w and w/o spinorbit coupling

    """
    from gpaw import GPAW
    from gpaw import mpi
    calc = GPAW(gpw, txt=None, communicator=mpi.serial_comm)
    return calc2eigs(calc, soc=soc, return_spin=return_spin,
                     theta=theta, phi=phi,
                     symmetry_tolerance=symmetry_tolerance)
