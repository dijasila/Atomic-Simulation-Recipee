from asr.core import command, argument, get_recipes, ASRResult
from asr.dimensionality import get_dimtypes
from asr.c2db.labels import label_explanation

# Style: "KVP: Long description !short description! [unit]

key_descriptions = {
    "berry": {
        "Topology": "KVP: Band topology !Topology!",
        "phi0_km": "Berry phase spectrum localized in k0",
        "s0_km": "Spin of berry phases localized in k0",
        "phi1_km": "Berry phase spectrum localized in k1",
        "s1_km": "Spin of berry phases localized in k1",
        "phi2_km": "Berry phase spectrum localized in k2",
        "s2_km": "Spin of berry phases localized in k3",
        "phi0_pi_km": "Berry phase spectrum at k2=pi localized in k0",
        "s0_pi_km": "Spin of berry at phases at k2=pi localized in k0",
    },
    "bse": {"E_B": "KVP: Exciton binding energy from BSE "
            "!Exc. bind. energy! [eV]"},
    "convex_hull": {
        "ehull": "KVP: Energy above convex hull [eV/atom]",
        "hform": "KVP: Heat of formation [eV/atom]",
        "thermodynamic_stability_level": "KVP: Thermodynamic stability level",
    },
    "defectinfo": {
        "defect_name": "KVP: Defect name {type}_{position} !Defect name!",
        "host_name": "KVP: Host formula",
        "charge_state": "KVP: Charge state"
    },
    "defect_symmetry": {
        "defect_pointgroup": "KVP: Defect point group !Point group!"
    },
    "magstate": {
        "magstate": "KVP: Magnetic state",
        "is_magnetic": "KVP: Material is magnetic !Magnetic!",
        "nspins": "KVP: Number of spins in calculator !n-spins!",
    },
    "gs": {
        "forces": "Forces on atoms [eV/Å]",
        "stresses": "Stress on unit cell [`eV/Å^{dim-1}`]",
        "etot": "KVP: Total energy !Tot. En.! [eV]",
        "evac": "KVP: Vacuum level !Vacuum level! [eV]",
        "evacdiff":
        "KVP: Vacuum level difference !Vacuum level difference! [eV]",
        "dipz": "KVP: Out-of-plane dipole along +z axis [e · Å/unit cell]",
        "efermi": "KVP: Fermi level !Fermi level! [eV]",
        "gap": "KVP: Band gap !Band gap! [eV]",
        "vbm": "KVP: Valence band maximum !Val. band maximum! [eV]",
        "cbm": "KVP: Conduction band minimum !Cond. band minimum! [eV]",
        "gap_dir": "KVP: Direct band gap !Dir. band gap! [eV]",
        "gap_dir_nosoc":
        "KVP: Direct gap w/o soc. !Dir. gap wo. soc.! [eV]",
        "gap_nosoc":
        "KVP: Gap w/o soc. !Gap wo. soc.! [eV]",
        "workfunction": "KVP: Work function (avg. if finite dipole) [eV]",
    },
    "gw": {
        "vbm_gw_nosoc": "Valence band maximum w/o soc. (G₀W₀) [eV]",
        "cbm_gw_nosoc": "Conduction band minimum w/o soc. (G₀W₀) [eV]",
        "gap_dir_gw_nosoc": "Direct gap w/o soc. (G₀W₀) [eV]",
        "gap_gw_nosoc": "Gap w/o soc. (G₀W₀) [eV]",
        "kvbm_nosoc": "k-point of G₀W₀ valence band maximum w/o soc",
        "kcbm_nosoc": "k-point of G₀W₀ conduction band minimum w/o soc",
        "vbm_gw": "KVP: Valence band maximum (G₀W₀) [eV]",
        "cbm_gw": "KVP: Conduction band minimum (G₀W₀) [eV]",
        "gap_dir_gw": "KVP: Direct band gap (G₀W₀) [eV]",
        "gap_gw": "KVP: Band gap (G₀W₀) [eV]",
        "kvbm": "k-point of G₀W₀ valence band maximum",
        "kcbm": "k-point of G₀W₀ conduction band minimum",
        "efermi_gw_nosoc": "Fermi level w/o soc. (G₀W₀) [eV]",
        "efermi_gw_soc": "Fermi level (G₀W₀) [eV]",
    },
    "hse": {
        "vbm_hse_nosoc": "Valence band maximum w/o soc. (HSE06) [eV]",
        "cbm_hse_nosoc": "Conduction band minimum w/o soc. (HSE06) [eV]",
        "gap_dir_hse_nosoc": "Direct gap w/o soc. (HSE06) [eV]",
        "gap_hse_nosoc": "Band gap w/o soc. (HSE06) [eV]",
        "kvbm_nosoc": "k-point of HSE06 valence band maximum w/o soc",
        "kcbm_nosoc": "k-point of HSE06 conduction band minimum w/o soc",
        "vbm_hse": "KVP: Valence band maximum (HSE06) [eV]",
        "cbm_hse": "KVP: Conduction band minimum (HSE06) [eV]",
        "gap_dir_hse": "KVP: Direct band gap (HSE06) [eV]",
        "gap_hse": "KVP: Band gap (HSE06) [eV]",
        "kvbm": "k-point of HSE06 valence band maximum",
        "kcbm": "k-point of HSE06 conduction band minimum",
        "efermi_hse_nosoc": "Fermi level w/o soc. (HSE06) [eV]",
        "efermi_hse_soc": "Fermi level (HSE06) [eV]",
    },
    "infraredpolarizability": {
        "alphax_lat": "KVP: Static lattice polarizability (x) [Å]",
        "alphay_lat": "KVP: Static lattice polarizability (y) [Å]",
        "alphaz_lat": "KVP: Static lattice polarizability (z) [Å]",
        "alphax": "KVP: Static total polarizability (x) [Å]",
        "alphay": "KVP: Static total polarizability (y) [Å]",
        "alphaz": "KVP: Static total polarizability (z) [Å]",
    },
    "magnetic_anisotropy": {
        "spin_axis": "KVP: Magnetic easy axis",
        "E_x": "KVP: Soc. total energy, x-direction [meV/unit cell]",
        "E_y": "KVP: Soc. total energy, y-direction [meV/unit cell]",
        "E_z": "KVP: Soc. total energy, z-direction [meV/unit cell]",
        "theta": "Easy axis, polar coordinates, theta [radians]",
        "phi": "Easy axis, polar coordinates, phi [radians]",
        "dE_zx":
        "KVP: Magnetic anisotropy (E<sub>z</sub> - E<sub>x</sub>) [meV/unit cell]",
        "dE_zy":
        "KVP: Magnetic anisotropy (E<sub>z</sub> - E<sub>y</sub>) [meV/unit cell]",
    },
    "exchange": {
        "J": "KVP: Nearest neighbor exchange coupling [meV]",
        "A": "KVP: Single-ion anisotropy (out-of-plane) [meV]",
        "lam": "KVP: Anisotropic exchange (out-of-plane) [meV]",
        "spin": "KVP: Maximum value of S_z at magnetic sites",
        "N_nn": "KVP: Number of nearest neighbors",
    },
    "pdos": {
        "pdos_nosoc":
        "Projected density of states w/o soc. !PDOS no soc!",
        "pdos_soc":
        "Projected density of states !PDOS!",
        "dos_at_ef_nosoc":
        "KVP: Density of states at the Fermi level w/o soc."
        "!DOS at ef no soc.! [states/(eV * unit cell)]",
        "dos_at_ef_soc":
        "KVP: Density of states at the Fermi level"
        " !DOS at ef! [states/(eV * unit cell)]",
    },
    "phonons": {
        "minhessianeig": "KVP: Minimum eigenvalue of Hessian [eV/Å²]",
        "dynamic_stability_phonons": "KVP: Phonon dynamic stability (low/high)",
    },
    "plasmafrequency": {
        "plasmafreq_vv": "Plasma frequency tensor [Hartree]",
        "plasmafrequency_x": "KVP: 2D plasma frequency (x)"
        "[`eV/Å^0.5`]",
        "plasmafrequency_y": "KVP: 2D plasma frequency (y)"
        "[`eV/Å^0.5`]",
    },
    "polarizability": {
        "alphax_el": "KVP: Static interband polarizability (x) [Å]",
        "alphay_el": "KVP: Static interband polarizability (y) [Å]",
        "alphaz_el": "KVP: Static interband polarizability (z) [Å]",
    },
    "relax": {
        "edft": "DFT total enrgy [eV]",
        "spos": "Array: Scaled positions",
        "symbols": "Array: Chemical symbols",
        "a": "Cell parameter a [Å]",
        "b": "Cell parameter b [Å]",
        "c": "Cell parameter c [Å]",
        "alpha": "Cell parameter alpha [deg]",
        "beta": "Cell parameter beta [deg]",
        "gamma": "Cell parameter gamma [deg]",
    },
    "stiffness": {
        "speed_of_sound_x": "KVP: Speed of sound (x) [m/s]",
        "speed_of_sound_y": "KVP: Speed of sound (y) [m/s]",
        "stiffness_tensor": "Stiffness tensor [`N/m^{dim-1}`]",
        "dynamic_stability_stiffness":
        "KVP: Stiffness dynamic stability (low/high)",
    },
    "structureinfo": {
        "cell_area": "KVP: Area of unit-cell [`Å²`]",
        "has_inversion_symmetry": "KVP: Material has inversion symmetry",
        "stoichiometry": "KVP: Stoichiometry",
        "spacegroup": "KVP: Space group",
        "spgnum": "KVP: Space group number",
        "pointgroup": "KVP: Point group",
        "crystal_type": "KVP: Crystal type",
    },
    "database.material_fingerprint": {
        'asr_id': 'KVP: Material unique ID',
        'uid': 'KVP: Unique identifier'
    },
    "dimensionality": {
        'dim_primary': 'KVP: Dim. with max. scoring parameter',
        'dim_primary_score': ('KVP: Dimensionality scoring parameter '
                              'of primary dimensionality.'),
        'dim_nclusters_0D': 'KVP: Number of 0D clusters.',
        'dim_nclusters_1D': 'KVP: Number of 1D clusters.',
        'dim_nclusters_2D': 'KVP: Number of 2D clusters.',
        'dim_nclusters_3D': 'KVP: Number of 3D clusters.',
        'dim_threshold_0D': 'KVP: 0D dimensionality threshold.',
        'dim_threshold_1D': 'KVP: 1D dimensionality threshold.',
        'dim_threshold_2D': 'KVP: 2D dimensionality threshold.',
        'dim_threshold_3D': 'KVP: 3D dimensionality threshold.',
    },
    "setinfo": {
        'first_class_material': (
            'KVP: A first class material marks a physical material. '
            '!First class material! [bool]'),
    },
    "info.json": {
        'class': 'KVP: Material class',
        'doi': 'KVP: Monolayer reported DOI',
        'icsd_id': 'KVP: Related ICSD id',
        'cod_id': 'KVP: Related COD id'
    },
    "emasses": {
        'emass_vb_dir1':
        'KVP: Valence band effective mass, direction 1 [`m_e`]',
        'emass_vb_dir2':
        'KVP: Valence band effective mass, direction 2 [`m_e`]',
        'emass_vb_dir3':
        'KVP: Valence band effective mass, direction 3 [`m_e`]',
        'emass_cb_dir1':
        'KVP: Conduction band effective mass, direction 1 [`m_e`]',
        'emass_cb_dir2':
        'KVP: Conduction band effective mass, direction 2 [`m_e`]',
        'emass_cb_dir3':
        'KVP: Conduction band effective mass, direction 3 [`m_e`]',
    },
    "c2db.labels": {
        "origin": f"KVP: {label_explanation}"
    },
    "database.fromtree": {
        "folder": "KVP: Path to collection folder",
    }
}


# Dimensionality key descrioptions:
for dimtype in get_dimtypes():
    key_descriptions['dimensionality'][f'dim_score_{dimtype}'] = \
        f'KVP: Dimensionality score of dimtype={dimtype}'

for i in range(6):
    for j in range(6):
        key_descriptions["stiffness"][f"c_{i}{j}"] = \
            f"KVP: Stiffness tensor, {i}{j}-component [`N/m^" + "{dim-1}`]"

# Piezoelectrictensor key_descriptions
piezokd = {}
for i in range(1, 4):
    for j in range(1, 7):
        key = 'e_{}{}'.format(i, j)
        name = 'Piezoelectric tensor'
        description = f'Piezoelectric tensor {i}{j}-component' + '[`\\text{Å}^{-1}`]'
        piezokd[key] = description

key_descriptions['piezoelectrictensor'] = piezokd

# Key descriptions like has_asr_gs_calculate
extras = {}
for recipe in get_recipes():
    key = 'has_' + recipe.name.replace('.', '_').replace('@', '_')
    extras[key] = f'{recipe.name} is calculated'
# Additional key description for spin coherence times of QPOD
extras['sct'] = 'Spin coherence time T2 [ms]'

key_descriptions['extra'] = extras


@command()
@argument('database', type=str)
def main(database: str) -> ASRResult:
    """Analyze database and set metadata.

    This recipe loops through all rows in a database and figures out what keys
    are present in the database. It also figures our what kind of materials
    (1D, 2D, 3D) are in the database. Then it saves those values in the
    database metadata under {"keys": ["etot", "gap", ...]}
    """
    from ase.db import connect

    db = connect(database)

    print('Row #')
    keys = set()
    for ir, row in enumerate(db.select(include_data=False)):
        if ir % 100 == 0:
            print(ir)
        keys.update(set(row.key_value_pairs.keys()))

    metadata = db.metadata
    metadata.update({'keys': sorted(list(keys))})
    db.metadata = metadata


if __name__ == '__main__':
    main.cli()
