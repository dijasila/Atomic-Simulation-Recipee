from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


class YFunctions:
    def __init__(self,
                 ndims: int,
                 lmax: int = 4):
        self.lmax = lmax
        self.ndims = ndims

        self.lm_j = [(-1, -1)]
        if ndims == 1:
            self.lm_j += [(0, 0)]
        elif ndims == 2:
            self.lm_j += [(0, 0)] + [(l, m)
                                     for l in range(2, lmax + 1, 2)
                                     for m in [0, 1]]
        else:
            self.lm_j += [(l, m)
                          for l in range(0, lmax + 1, 2)
                          for m in range(2 * l + 1)]

    def create_fit_from_coefs(self,
                              coef_j: np.ndarray,
                              kmin_v: np.ndarray) -> YFit:
        return YFit(self, coef_j, kmin_v)

    def fit_data(self,
                 k_iv: np.ndarray,
                 eig_i: np.ndarray) -> YFit:
        npoints, ndims = k_iv.shape
        assert ndims == self.ndims

        if npoints < 1.25 * len(self.lm_j):
            raise ValueError('Too few points!')

        def f(k_v):
            dk_iv = k_iv - k_v
            M_ji = self._calculate(dk_iv)
            coef_j = np.linalg.lstsq(M_ji.T, eig_i, rcond=None)[0]
            error_i = coef_j @ M_ji - eig_i
            return error_i, coef_j

        k0_v = k_iv[eig_i.argmin()].copy()

        result = minimize(
            lambda k_v: (f(k_v)[0]**2).sum(),
            x0=k0_v,
            method='Nelder-Mead')  # seems more robust than the default

        if not result.success:
            raise ValueError(result.message)

        kmin_v = result.x
        error_i, coef_j = f(kmin_v)
        max_error = abs(error_i).max()

        return YFit(self, coef_j, kmin_v), max_error

    def _calculate(self, k_xv: np.ndarray) -> np.ndarray:
        k2_x = (k_xv**2).sum(1)

        eps = 1e-12
        k2_x[k2_x < eps] = eps
        khat_xv = k_xv / (k2_x**0.5)[:, np.newaxis]

        M_jx = np.empty((len(self.lm_j), len(k2_x)))
        for j, (l, m) in enumerate(self.lm_j):
            if j == 0:
                M_jx[0] = 1.0
            else:
                f_x = Y(l, m, *khat_xv.T)
                M_jx[j] = f_x * k2_x
        return M_jx


class YFit:
    def __init__(self,
                 yfuncs,
                 coef_j: np.ndarray,
                 kmin_v: np.ndarray):
        self.yfuncs = yfuncs
        self.coef_j = coef_j
        self.kmin_v = kmin_v
        assert len(coef_j) == len(yfuncs.lm_j)
        self.emin = coef_j[0]

    def values(self, k_xv: np.ndarray) -> np.ndarray:
        M_jx = self.yfuncs._calculate(k_xv - self.kmin_v)
        return self.coef_j @ M_jx

    def hessian(self) -> np.ndarray:
        if self.yfuncs.ndims == 1:
            # 1, xx
            _, c00 = self.coef_j
            return np.array([[2 * c00]])

        if self.yfuncs.ndims == 1:
            # 1, xx+yy, xx-yy, 2xy
            _, c00, c20, c21 = self.coef_j[:4]
            return np.array([[2 * (c00 + c20), 2 * c21],
                             [2 * c21, 2 * (c00 - c20)]])

        _, c00, c20, c21, c22, c23, c24 = self.coef_j[:7]
        hess_vv = np.eye(3) * c00 * 2 * 0.28209479177387814
        hess_vv[0, 1] = c20 * 1.0925484305920792
        hess_vv[1, 2] = c21 * 1.0925484305920792
        hess_vv[2, 2] += c22 * 2 * 0.6307831305050401
        hess_vv[1, 1] -= c22 * 2 * 0.31539156525252005
        hess_vv[0, 0] -= c22 * 2 * 0.31539156525252005
        hess_vv[0, 2] = c23 * 1.0925484305920792
        hess_vv[0, 0] += c24 * 2 * 0.5462742152960396
        hess_vv[1, 1] -= c24 * 2 * 0.5462742152960396
        hess_vv[1, 0] = hess_vv[0, 1]
        hess_vv[2, 0] = hess_vv[0, 1]
        hess_vv[2, 1] = hess_vv[1, 2]
        return hess_vv

    def warping(self) -> float:
        analytic = 0.0
        non_analytic = 0.0
        for (l, m), c in zip(self.yfuncs.lm_j, self.coef_j):
            if l == -1:
                continue
            if l == 0 or l == 2:
                analytic += c**2
            else:
                non_analytic += c**2
        return non_analytic / analytic


def Y(l: int,
      m: int,
      x: np.ndarray,
      y: np.ndarray = None,
      z: np.ndarray = None) -> np.ndarray:
    if y is None:
        if l == 0:
            return np.ones_like(x)
        return x

    if z is None:
        # cubic harmonics:
        if l == 0:
            return np.ones_like(x)
        if m == 0:
            return 2**0.5 * ((x + 1j * y)**l).real
        assert m == 1
        return 2**0.5 * ((x + 1j * y)**l).imag

    result = 0.0
    for c, (i, j, k) in Y_L[l**2 + m]:
        result += c * x**i * y**j * z**k
    return result


Y_L = [
    # s, l=0:
    [(0.28209479177387814, (0, 0, 0))],
    # p, l=1:
    [(0.4886025119029199, (0, 1, 0))],
    [(0.4886025119029199, (0, 0, 1))],
    [(0.4886025119029199, (1, 0, 0))],
    # d, l=2:
    [(1.0925484305920792, (1, 1, 0))],
    [(1.0925484305920792, (0, 1, 1))],
    [(0.6307831305050401, (0, 0, 2)),
     (-0.31539156525252005, (0, 2, 0)),
     (-0.31539156525252005, (2, 0, 0))],
    [(1.0925484305920792, (1, 0, 1))],
    [(0.5462742152960396, (2, 0, 0)),
     (-0.5462742152960396, (0, 2, 0))],
    # f, l=3:
    [(-0.5900435899266435, (0, 3, 0)),
     (1.7701307697799304, (2, 1, 0))],
    [(2.890611442640554, (1, 1, 1))],
    [(-0.4570457994644658, (0, 3, 0)),
     (1.828183197857863, (0, 1, 2)),
     (-0.4570457994644658, (2, 1, 0))],
    [(0.7463526651802308, (0, 0, 3)),
     (-1.1195289977703462, (2, 0, 1)),
     (-1.1195289977703462, (0, 2, 1))],
    [(1.828183197857863, (1, 0, 2)),
     (-0.4570457994644658, (3, 0, 0)),
     (-0.4570457994644658, (1, 2, 0))],
    [(1.445305721320277, (2, 0, 1)),
     (-1.445305721320277, (0, 2, 1))],
    [(0.5900435899266435, (3, 0, 0)),
     (-1.7701307697799304, (1, 2, 0))],
    # g, l=4:
    [(2.5033429417967046, (3, 1, 0)),
     (-2.5033429417967046, (1, 3, 0))],
    [(-1.7701307697799307, (0, 3, 1)),
     (5.310392309339792, (2, 1, 1))],
    [(-0.9461746957575601, (3, 1, 0)),
     (-0.9461746957575601, (1, 3, 0)),
     (5.6770481745453605, (1, 1, 2))],
    [(-2.0071396306718676, (2, 1, 1)),
     (2.676186174229157, (0, 1, 3)),
     (-2.0071396306718676, (0, 3, 1))],
    [(0.6347132814912259, (2, 2, 0)),
     (-2.5388531259649034, (2, 0, 2)),
     (0.31735664074561293, (0, 4, 0)),
     (-2.5388531259649034, (0, 2, 2)),
     (0.31735664074561293, (4, 0, 0)),
     (0.8462843753216345, (0, 0, 4))],
    [(2.676186174229157, (1, 0, 3)),
     (-2.0071396306718676, (3, 0, 1)),
     (-2.0071396306718676, (1, 2, 1))],
    [(2.8385240872726802, (2, 0, 2)),
     (0.47308734787878004, (0, 4, 0)),
     (-0.47308734787878004, (4, 0, 0)),
     (-2.8385240872726802, (0, 2, 2))],
    [(1.7701307697799307, (3, 0, 1)),
     (-5.310392309339792, (1, 2, 1))],
    [(-3.755014412695057, (2, 2, 0)),
     (0.6258357354491761, (0, 4, 0)),
     (0.6258357354491761, (4, 0, 0))],
    # h, l=5:
    [(-6.5638205684017015, (2, 3, 0)),
     (3.2819102842008507, (4, 1, 0)),
     (0.6563820568401701, (0, 5, 0))],
    [(8.302649259524165, (3, 1, 1)),
     (-8.302649259524165, (1, 3, 1))],
    [(-3.913906395482003, (0, 3, 2)),
     (0.4892382994352504, (0, 5, 0)),
     (-1.467714898305751, (4, 1, 0)),
     (-0.9784765988705008, (2, 3, 0)),
     (11.741719186446009, (2, 1, 2))],
    [(-4.793536784973324, (3, 1, 1)),
     (-4.793536784973324, (1, 3, 1)),
     (9.587073569946648, (1, 1, 3))],
    [(-5.435359814348363, (0, 3, 2)),
     (0.9058933023913939, (2, 3, 0)),
     (-5.435359814348363, (2, 1, 2)),
     (3.6235732095655755, (0, 1, 4)),
     (0.45294665119569694, (4, 1, 0)),
     (0.45294665119569694, (0, 5, 0))],
    [(3.508509673602708, (2, 2, 1)),
     (-4.678012898136944, (0, 2, 3)),
     (1.754254836801354, (0, 4, 1)),
     (-4.678012898136944, (2, 0, 3)),
     (1.754254836801354, (4, 0, 1)),
     (0.9356025796273888, (0, 0, 5))],
    [(-5.435359814348363, (3, 0, 2)),
     (3.6235732095655755, (1, 0, 4)),
     (0.45294665119569694, (5, 0, 0)),
     (0.9058933023913939, (3, 2, 0)),
     (-5.435359814348363, (1, 2, 2)),
     (0.45294665119569694, (1, 4, 0))],
    [(-2.396768392486662, (4, 0, 1)),
     (2.396768392486662, (0, 4, 1)),
     (4.793536784973324, (2, 0, 3)),
     (-4.793536784973324, (0, 2, 3))],
    [(3.913906395482003, (3, 0, 2)),
     (-0.4892382994352504, (5, 0, 0)),
     (0.9784765988705008, (3, 2, 0)),
     (-11.741719186446009, (1, 2, 2)),
     (1.467714898305751, (1, 4, 0))],
    [(2.075662314881041, (4, 0, 1)),
     (-12.453973889286246, (2, 2, 1)),
     (2.075662314881041, (0, 4, 1))],
    [(-6.5638205684017015, (3, 2, 0)),
     (0.6563820568401701, (5, 0, 0)),
     (3.2819102842008507, (1, 4, 0))],
    # i, l=6:
    [(4.099104631151485, (5, 1, 0)),
     (-13.663682103838287, (3, 3, 0)),
     (4.099104631151485, (1, 5, 0))],
    [(11.83309581115876, (4, 1, 1)),
     (-23.66619162231752, (2, 3, 1)),
     (2.366619162231752, (0, 5, 1))],
    [(20.182596029148968, (3, 1, 2)),
     (-2.0182596029148967, (5, 1, 0)),
     (2.0182596029148967, (1, 5, 0)),
     (-20.182596029148968, (1, 3, 2))],
    [(-7.369642076119388, (0, 3, 3)),
     (-5.527231557089541, (2, 3, 1)),
     (2.7636157785447706, (0, 5, 1)),
     (22.108926228358165, (2, 1, 3)),
     (-8.29084733563431, (4, 1, 1))],
    [(-14.739284152238776, (3, 1, 2)),
     (14.739284152238776, (1, 1, 4)),
     (1.842410519029847, (3, 3, 0)),
     (0.9212052595149235, (5, 1, 0)),
     (-14.739284152238776, (1, 3, 2)),
     (0.9212052595149235, (1, 5, 0))],
    [(2.9131068125936572, (0, 5, 1)),
     (-11.652427250374629, (0, 3, 3)),
     (5.8262136251873144, (2, 3, 1)),
     (-11.652427250374629, (2, 1, 3)),
     (2.9131068125936572, (4, 1, 1)),
     (4.660970900149851, (0, 1, 5))],
    [(5.721228204086558, (4, 0, 2)),
     (-7.628304272115411, (0, 2, 4)),
     (-0.9535380340144264, (2, 4, 0)),
     (1.0171072362820548, (0, 0, 6)),
     (-0.9535380340144264, (4, 2, 0)),
     (5.721228204086558, (0, 4, 2)),
     (-0.3178460113381421, (0, 6, 0)),
     (-7.628304272115411, (2, 0, 4)),
     (-0.3178460113381421, (6, 0, 0)),
     (11.442456408173117, (2, 2, 2))],
    [(-11.652427250374629, (3, 0, 3)),
     (4.660970900149851, (1, 0, 5)),
     (2.9131068125936572, (5, 0, 1)),
     (5.8262136251873144, (3, 2, 1)),
     (-11.652427250374629, (1, 2, 3)),
     (2.9131068125936572, (1, 4, 1))],
    [(7.369642076119388, (2, 0, 4)),
     (-7.369642076119388, (0, 2, 4)),
     (-0.46060262975746175, (2, 4, 0)),
     (-7.369642076119388, (4, 0, 2)),
     (0.46060262975746175, (4, 2, 0)),
     (-0.46060262975746175, (0, 6, 0)),
     (0.46060262975746175, (6, 0, 0)),
     (7.369642076119388, (0, 4, 2))],
    [(7.369642076119388, (3, 0, 3)),
     (-2.7636157785447706, (5, 0, 1)),
     (5.527231557089541, (3, 2, 1)),
     (-22.108926228358165, (1, 2, 3)),
     (8.29084733563431, (1, 4, 1))],
    [(2.522824503643621, (4, 2, 0)),
     (5.045649007287242, (0, 4, 2)),
     (-30.273894043723452, (2, 2, 2)),
     (-0.5045649007287242, (0, 6, 0)),
     (-0.5045649007287242, (6, 0, 0)),
     (5.045649007287242, (4, 0, 2)),
     (2.522824503643621, (2, 4, 0))],
    [(2.366619162231752, (5, 0, 1)),
     (-23.66619162231752, (3, 2, 1)),
     (11.83309581115876, (1, 4, 1))],
    [(-10.247761577878714, (4, 2, 0)),
     (-0.6831841051919143, (0, 6, 0)),
     (0.6831841051919143, (6, 0, 0)),
     (10.247761577878714, (2, 4, 0))],
    # j, l=7:
    [(14.850417383016522, (2, 5, 0)),
     (4.950139127672174, (6, 1, 0)),
     (-24.75069563836087, (4, 3, 0)),
     (-0.7071627325245963, (0, 7, 0))],
    [(-52.91921323603801, (3, 3, 1)),
     (15.875763970811402, (1, 5, 1)),
     (15.875763970811402, (5, 1, 1))],
    [(-2.5945778936013015, (6, 1, 0)),
     (2.5945778936013015, (4, 3, 0)),
     (-62.26986944643124, (2, 3, 2)),
     (4.670240208482342, (2, 5, 0)),
     (6.226986944643123, (0, 5, 2)),
     (31.13493472321562, (4, 1, 2)),
     (-0.5189155787202603, (0, 7, 0))],
    [(41.513246297620825, (3, 1, 3)),
     (12.453973889286246, (1, 5, 1)),
     (-41.513246297620825, (1, 3, 3)),
     (-12.453973889286246, (5, 1, 1))],
    [(-18.775072063475285, (2, 3, 2)),
     (-0.4693768015868821, (0, 7, 0)),
     (0.4693768015868821, (2, 5, 0)),
     (2.3468840079344107, (4, 3, 0)),
     (-12.516714708983523, (0, 3, 4)),
     (37.55014412695057, (2, 1, 4)),
     (1.4081304047606462, (6, 1, 0)),
     (9.387536031737643, (0, 5, 2)),
     (-28.162608095212928, (4, 1, 2))],
    [(13.27598077334948, (3, 3, 1)),
     (6.63799038667474, (1, 5, 1)),
     (-35.402615395598616, (3, 1, 3)),
     (21.24156923735917, (1, 1, 5)),
     (-35.402615395598616, (1, 3, 3)),
     (6.63799038667474, (5, 1, 1))],
    [(-0.4516580379125865, (0, 7, 0)),
     (10.839792909902076, (0, 5, 2)),
     (-1.3549741137377596, (2, 5, 0)),
     (-1.3549741137377596, (4, 3, 0)),
     (-21.679585819804153, (0, 3, 4)),
     (-21.679585819804153, (2, 1, 4)),
     (5.781222885281108, (0, 1, 6)),
     (-0.4516580379125865, (6, 1, 0)),
     (21.679585819804153, (2, 3, 2)),
     (10.839792909902076, (4, 1, 2))],
    [(-11.471758521216831, (2, 0, 5)),
     (1.0925484305920792, (0, 0, 7)),
     (-11.471758521216831, (0, 2, 5)),
     (28.67939630304208, (2, 2, 3)),
     (-2.3899496919201733, (6, 0, 1)),
     (-7.16984907576052, (4, 2, 1)),
     (14.33969815152104, (4, 0, 3)),
     (-2.3899496919201733, (0, 6, 1)),
     (-7.16984907576052, (2, 4, 1)),
     (14.33969815152104, (0, 4, 3))],
    [(10.839792909902076, (1, 4, 2)),
     (-0.4516580379125865, (7, 0, 0)),
     (21.679585819804153, (3, 2, 2)),
     (-1.3549741137377596, (5, 2, 0)),
     (-0.4516580379125865, (1, 6, 0)),
     (-21.679585819804153, (3, 0, 4)),
     (-1.3549741137377596, (3, 4, 0)),
     (5.781222885281108, (1, 0, 6)),
     (-21.679585819804153, (1, 2, 4)),
     (10.839792909902076, (5, 0, 2))],
    [(10.620784618679584, (2, 0, 5)),
     (-10.620784618679584, (0, 2, 5)),
     (3.31899519333737, (6, 0, 1)),
     (3.31899519333737, (4, 2, 1)),
     (-17.701307697799308, (4, 0, 3)),
     (-3.31899519333737, (0, 6, 1)),
     (-3.31899519333737, (2, 4, 1)),
     (17.701307697799308, (0, 4, 3))],
    [(-1.4081304047606462, (1, 6, 0)),
     (0.4693768015868821, (7, 0, 0)),
     (18.775072063475285, (3, 2, 2)),
     (-0.4693768015868821, (5, 2, 0)),
     (12.516714708983523, (3, 0, 4)),
     (-2.3468840079344107, (3, 4, 0)),
     (28.162608095212928, (1, 4, 2)),
     (-37.55014412695057, (1, 2, 4)),
     (-9.387536031737643, (5, 0, 2))],
    [(10.378311574405206, (4, 0, 3)),
     (-3.1134934723215615, (0, 6, 1)),
     (15.56746736160781, (2, 4, 1)),
     (-62.26986944643124, (2, 2, 3)),
     (10.378311574405206, (0, 4, 3)),
     (-3.1134934723215615, (6, 0, 1)),
     (15.56746736160781, (4, 2, 1))],
    [(-2.5945778936013015, (1, 6, 0)),
     (-62.26986944643124, (3, 2, 2)),
     (-0.5189155787202603, (7, 0, 0)),
     (31.13493472321562, (1, 4, 2)),
     (2.5945778936013015, (3, 4, 0)),
     (6.226986944643123, (5, 0, 2)),
     (4.670240208482342, (5, 2, 0))],
    [(2.6459606618019005, (6, 0, 1)),
     (-2.6459606618019005, (0, 6, 1)),
     (-39.68940992702851, (4, 2, 1)),
     (39.68940992702851, (2, 4, 1))],
    [(0.7071627325245963, (7, 0, 0)),
     (-14.850417383016522, (5, 2, 0)),
     (24.75069563836087, (3, 4, 0)),
     (-4.950139127672174, (1, 6, 0))],
    # k, l=8:
    [(-5.831413281398639, (1, 7, 0)),
     (40.81989296979047, (3, 5, 0)),
     (-40.81989296979047, (5, 3, 0)),
     (5.831413281398639, (7, 1, 0))],
    [(-2.9157066406993195, (0, 7, 1)),
     (61.22983945468571, (2, 5, 1)),
     (-102.04973242447618, (4, 3, 1)),
     (20.409946484895237, (6, 1, 1))],
    [(7.452658724833596, (3, 5, 0)),
     (-3.1939965963572554, (1, 7, 0)),
     (44.715952349001576, (1, 5, 2)),
     (7.452658724833596, (5, 3, 0)),
     (-149.05317449667191, (3, 3, 2)),
     (-3.1939965963572554, (7, 1, 0)),
     (44.715952349001576, (5, 1, 2))],
    [(31.04919559888297, (2, 5, 1)),
     (-3.449910622098108, (0, 7, 1)),
     (13.799642488392433, (0, 5, 3)),
     (17.24955311049054, (4, 3, 1)),
     (-137.99642488392433, (2, 3, 3)),
     (-17.24955311049054, (6, 1, 1)),
     (68.99821244196217, (4, 1, 3))],
    [(-1.9136660990373229, (3, 5, 0)),
     (-1.9136660990373229, (1, 7, 0)),
     (45.927986376895745, (1, 5, 2)),
     (-76.54664396149292, (1, 3, 4)),
     (1.9136660990373229, (7, 1, 0)),
     (1.9136660990373229, (5, 3, 0)),
     (-45.927986376895745, (5, 1, 2)),
     (76.54664396149292, (3, 1, 4))],
    [(18.528992329433162, (4, 3, 1)),
     (3.705798465886632, (2, 5, 1)),
     (-49.41064621182176, (2, 3, 3)),
     (-3.705798465886632, (0, 7, 1)),
     (24.70532310591088, (0, 5, 3)),
     (-19.764258484728707, (0, 3, 5)),
     (11.117395397659896, (6, 1, 1)),
     (-74.11596931773265, (4, 1, 3)),
     (59.29277545418611, (2, 1, 5))],
    [(-0.912304516869819, (7, 1, 0)),
     (-2.7369135506094566, (5, 3, 0)),
     (27.36913550609457, (5, 1, 2)),
     (-2.7369135506094566, (3, 5, 0)),
     (54.73827101218914, (3, 3, 2)),
     (-72.98436134958551, (3, 1, 4)),
     (-0.912304516869819, (1, 7, 0)),
     (27.36913550609457, (1, 5, 2)),
     (-72.98436134958551, (1, 3, 4)),
     (29.193744539834206, (1, 1, 6))],
    [(-3.8164436064572986, (6, 1, 1)),
     (-11.449330819371895, (4, 3, 1)),
     (30.53154885165839, (4, 1, 3)),
     (-11.449330819371895, (2, 5, 1)),
     (61.06309770331678, (2, 3, 3)),
     (-36.63785862199006, (2, 1, 5)),
     (-3.8164436064572986, (0, 7, 1)),
     (30.53154885165839, (0, 5, 3)),
     (-36.63785862199006, (0, 3, 5)),
     (6.978639737521917, (0, 1, 7))],
    [(0.31803696720477487, (8, 0, 0)),
     (1.2721478688190995, (6, 2, 0)),
     (-10.177182950552796, (6, 0, 2)),
     (1.9082218032286493, (4, 4, 0)),
     (-30.53154885165839, (4, 2, 2)),
     (30.53154885165839, (4, 0, 4)),
     (1.2721478688190995, (2, 6, 0)),
     (-30.53154885165839, (2, 4, 2)),
     (61.06309770331678, (2, 2, 4)),
     (-16.283492720884475, (2, 0, 6)),
     (0.31803696720477487, (0, 8, 0)),
     (-10.177182950552796, (0, 6, 2)),
     (30.53154885165839, (0, 4, 4)),
     (-16.283492720884475, (0, 2, 6)),
     (1.1631066229203195, (0, 0, 8))],
    [(-3.8164436064572986, (7, 0, 1)),
     (-11.449330819371895, (5, 2, 1)),
     (30.53154885165839, (5, 0, 3)),
     (-11.449330819371895, (3, 4, 1)),
     (61.06309770331678, (3, 2, 3)),
     (-36.63785862199006, (3, 0, 5)),
     (-3.8164436064572986, (1, 6, 1)),
     (30.53154885165839, (1, 4, 3)),
     (-36.63785862199006, (1, 2, 5)),
     (6.978639737521917, (1, 0, 7))],
    [(0.912304516869819, (2, 6, 0)),
     (-13.684567753047284, (2, 4, 2)),
     (0.4561522584349095, (0, 8, 0)),
     (-13.684567753047284, (0, 6, 2)),
     (36.492180674792756, (0, 4, 4)),
     (-14.596872269917103, (0, 2, 6)),
     (-0.4561522584349095, (8, 0, 0)),
     (-0.912304516869819, (6, 2, 0)),
     (13.684567753047284, (6, 0, 2)),
     (13.684567753047284, (4, 2, 2)),
     (-36.492180674792756, (4, 0, 4)),
     (14.596872269917103, (2, 0, 6))],
    [(-3.705798465886632, (5, 2, 1)),
     (-18.528992329433162, (3, 4, 1)),
     (49.41064621182176, (3, 2, 3)),
     (-11.117395397659896, (1, 6, 1)),
     (74.11596931773265, (1, 4, 3)),
     (-59.29277545418611, (1, 2, 5)),
     (3.705798465886632, (7, 0, 1)),
     (-24.70532310591088, (5, 0, 3)),
     (19.764258484728707, (3, 0, 5))],
    [(-4.784165247593307, (4, 4, 0)),
     (-1.9136660990373229, (2, 6, 0)),
     (57.40998297111968, (2, 4, 2)),
     (0.4784165247593307, (0, 8, 0)),
     (-11.481996594223936, (0, 6, 2)),
     (19.13666099037323, (0, 4, 4)),
     (-1.9136660990373229, (6, 2, 0)),
     (57.40998297111968, (4, 2, 2)),
     (-114.81996594223936, (2, 2, 4)),
     (0.4784165247593307, (8, 0, 0)),
     (-11.481996594223936, (6, 0, 2)),
     (19.13666099037323, (4, 0, 4))],
    [(17.24955311049054, (3, 4, 1)),
     (-17.24955311049054, (1, 6, 1)),
     (68.99821244196217, (1, 4, 3)),
     (31.04919559888297, (5, 2, 1)),
     (-137.99642488392433, (3, 2, 3)),
     (-3.449910622098108, (7, 0, 1)),
     (13.799642488392433, (5, 0, 3))],
    [(-7.452658724833596, (2, 6, 0)),
     (0.5323327660595426, (0, 8, 0)),
     (-7.452658724833596, (0, 6, 2)),
     (111.78988087250394, (2, 4, 2)),
     (7.452658724833596, (6, 2, 0)),
     (-111.78988087250394, (4, 2, 2)),
     (-0.5323327660595426, (8, 0, 0)),
     (7.452658724833596, (6, 0, 2))],
    [(-20.409946484895237, (1, 6, 1)),
     (102.04973242447618, (3, 4, 1)),
     (-61.22983945468571, (5, 2, 1)),
     (2.9157066406993195, (7, 0, 1))],
    [(0.7289266601748299, (0, 8, 0)),
     (-20.409946484895237, (2, 6, 0)),
     (51.02486621223809, (4, 4, 0)),
     (-20.409946484895237, (6, 2, 0)),
     (0.7289266601748299, (8, 0, 0))]]
