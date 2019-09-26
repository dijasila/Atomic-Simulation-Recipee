from asr.core import command


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


@command('asr.fermisurface',
         requires=['gs.gpw'])
def main():
    import numpy as np
    from gpaw import GPAW
    from c2db.utils import gpw2eigs, spin_axis
    from gpaw.kpt_descriptor import to1bz
    eigs_km, ef, s_kvm = gpw2eigs('gs.gpw', return_spin=True,
                                  optimal_spin_direction=True)
    eigs_mk = eigs_km.T
    eigs_mk -= ef
    calc = GPAW('gs.gpw', txt=None)
    s_mk = s_kvm[:, spin_axis()].T
    A_cv = calc.atoms.get_cell()
    B_cv = np.linalg.inv(A_cv).T * 2 * np.pi

    bzk_kc = calc.wfs.kd.bzk_kc
    bzk_kv = np.dot(bzk_kc, B_cv)

    contours = []
    selection = ~np.logical_or(eigs_mk.max(1) < 0, eigs_mk.min(1) > 0)
    eigs_mk = eigs_mk[selection, :]
    s_mk = s_mk[selection, :]
    bz2ibz_k = calc.wfs.kd.bz2ibz_k
    eigs_mk = eigs_mk[:, bz2ibz_k]
    s_mk = s_mk[:, bz2ibz_k]

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

    data = {'contours': contours}
    return data


if __name__ == '__main__':
    main.cli()
