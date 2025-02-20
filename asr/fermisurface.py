"""Fermi surfaces."""
from asr.core import command, ASRResult, prepare_result, option
from asr.database.browser import fig, make_panel_description, describe_entry


panel_description = make_panel_description(
    """The Fermi surface calculated with spin–orbit interactions. The expectation
value of S_i (where i=z for non-magnetic materials and otherwise is the
magnetic easy axis) indicated by the color code.""",
    articles=[
        'C2DB',
    ],
)


def bz_vertices(cell):
    import numpy as np
    from scipy.spatial import Voronoi
    icell = np.linalg.inv(cell) * 2 * np.pi
    ind = np.indices((3, 3)).reshape((2, 9)) - 1
    G = np.dot(icell, ind).T
    vor = Voronoi(G)
    bz1 = []
    for vertices, points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 not in vertices and 4 in points:
            normal = G[points].sum(0)
            normal /= (normal**2).sum()**0.5
            bz1.append((vor.vertices[vertices], normal))
    return bz1


def find_contours(eigs_nk, bzk_kv, s_nk=None):
    import numpy as np
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    minx = np.min(bzk_kv[:, 0:2])
    maxx = np.max(bzk_kv[:, 0:2])

    npoints = 1000

    xi = np.linspace(minx, maxx, npoints)
    yi = np.linspace(minx, maxx, npoints)

    zis = []
    for eigs_k in eigs_nk:
        zi = griddata((bzk_kv[:, 0], bzk_kv[:, 1]), eigs_k,
                      (xi[None, :], yi[:, None]), method='cubic')
        zis.append(zi)

    contours = []
    for n, zi in enumerate(zis):
        cs = plt.contour(xi, yi, zi, levels=[0])
        paths = cs.collections[0].get_paths()

        for path in paths:
            vertices = []
            for vertex in path.iter_segments(simplify=False):
                vertices.append(np.array((vertex[0][0],
                                          vertex[0][1],
                                          vertex[1], 0), float))
            vertices = np.array(vertices)
            if s_nk is not None:
                si = griddata((bzk_kv[:, 0], bzk_kv[:, 1]), s_nk[n],
                              vertices[:, :2], method='cubic')
                vertices[:, -1] = si
            contours.append(vertices)

    return contours


def webpanel(result, row, key_descriptions):

    panel = {'title': describe_entry('Fermi surface', panel_description),
             'columns': [[fig('fermi_surface.png')]],
             'plot_descriptions': [{'function': plot_fermi,
                                    'filenames': ['fermi_surface.png']}],
             'sort': 13}

    return [panel]


def plot_fermi(row, fname, sfs=1, dpi=200):
    from ase.geometry.cell import Cell
    from matplotlib import pyplot as plt
    from asr.utils.symmetry import c2db_symmetry_eps
    cell = Cell(row.cell)
    lat = cell.get_bravais_lattice(pbc=row.pbc, eps=c2db_symmetry_eps)
    plt.figure(figsize=(5, 4))
    ax = lat.plot_bz(vectors=False, pointstyle={'c': 'k', 'marker': '.'})
    add_fermi(row, ax=ax, s=sfs)
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)


def add_fermi(row, ax, s=0.25):
    from matplotlib import pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    verts = row.data['results-asr.fermisurface.json']['contours'].copy()
    normalize = colors.Normalize(vmin=-1, vmax=1)
    verts[:, :2] /= (2 * np.pi)
    im = ax.scatter(verts[:, 0], verts[:, 1], c=verts[:, -1],
                    s=s, cmap='viridis', marker=',',
                    norm=normalize, alpha=1, zorder=2)

    sdir = row.get('spin_axis', 'z')
    cbar = plt.colorbar(im, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params()
    cbar.set_label(r'$\langle S_{} \rangle $'.format(sdir))


@prepare_result
class Result(ASRResult):

    contours: list
    key_descriptions = {'contours': 'List of Fermi surface contours.'}

    formats = {"ase_webpanel": webpanel}


@command('asr.fermisurface',
         returns=Result,
         requires=['gs.gpw', 'results-asr.structureinfo.json'],
         dependencies=['asr.gs', 'asr.structureinfo'])
@option('--shift', help='Shift of Fermi level in eV.', type=float)
def main(shift: float = 0.0) -> Result:
    import numpy as np
    from gpaw import GPAW
    from asr.utils.gpw2eigs import gpw2eigs
    from gpaw.kpt_descriptor import to1bz
    from asr.magnetic_anisotropy import get_spin_axis, get_spin_index
    from ase.io import read
    atoms = read('structure.json')
    ndim = sum(atoms.pbc)
    assert ndim == 2, 'Fermi surface recipe only implemented for 2D systems.'
    theta, phi = get_spin_axis()
    eigs_km, ef, s_kvm = gpw2eigs('gs.gpw', return_spin=True,
                                  theta=theta, phi=phi,
                                  symmetry_tolerance=1e-2)
    ef += shift
    eigs_mk = eigs_km.T
    eigs_mk = eigs_mk - ef
    calc = GPAW('gs.gpw', txt=None)
    s_mk = s_kvm[:, get_spin_index()].T

    A_cv = calc.atoms.get_cell()
    B_cv = np.linalg.inv(A_cv).T * 2 * np.pi

    bzk_kc = calc.get_bz_k_points()
    bzk_kv = np.dot(bzk_kc, B_cv)

    contours = []
    selection = ~np.logical_or(eigs_mk.max(1) < 0, eigs_mk.min(1) > 0)
    eigs_mk = eigs_mk[selection, :]
    s_mk = s_mk[selection, :]

    n = 5
    N_xc = np.indices((n, n, 1)).reshape((3, n**2)).T - n // 2
    N_xc += np.array((0, 0, n // 2))
    N_xv = np.dot(N_xc, B_cv)

    eigs_mk = np.repeat(eigs_mk, n**2, axis=1)
    s_mk = np.repeat(s_mk, n**2, axis=1)
    tmpbzk_kv = (bzk_kv[:, np.newaxis] + N_xv[np.newaxis]).reshape(-1, 3)
    tmpcontours = find_contours(eigs_mk, tmpbzk_kv, s_nk=s_mk)

    # Only include contours with a part in 1st BZ
    for cnt in tmpcontours:
        k_kv = cnt.copy()
        k_kv = k_kv[:, :3]
        k_kv[:, 2] = 0
        k_kc = np.dot(k_kv, A_cv.T) / (2 * np.pi)
        inds_k = np.linalg.norm(to1bz(k_kc, A_cv) - k_kc, axis=1) < 1e-8
        if (inds_k).any():
            contours.append(cnt[inds_k, :])

    contours = np.concatenate(contours)
    data = {'contours': contours}
    return data


if __name__ == '__main__':
    main.cli()
