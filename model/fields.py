"""Quasi-steady reaction-diffusion fields solved with warm-started red-black SOR.

Each field obeys  lap(c) - k(x) c + src(x) = 0  on the lattice (D = 1 site^2 per unit),
with zero-flux (Neumann) boundaries. Oxygen is pinned to 1 at vessel sites.
"""
import numpy as np
from numba import njit

from .params import (P_O_HYP, P_CCL2_SECRETE, P_K_CCL2, P_M1_CHEMO, P_M2_SUPP, P_IFN_SECRETE, P_K_IFN, P_K_O_TISSUE, P_K_O_TUMOR, P_K_C, P_CHEMO_SECRETE, P_CHEMO_APOP,
                     P_K_S, P_SUPP_SECRETE)

VESSEL = 5


@njit(cache=True)
def _sor(c, k, src, pinned, pin_val, sweeps, omega):
    nx, ny = c.shape
    for _ in range(sweeps):
        for color in range(2):
            for i in range(nx):
                im = i - 1 if i > 0 else 0
                ip = i + 1 if i < nx - 1 else nx - 1
                for j in range((i + color) % 2, ny, 2):
                    if pinned[i, j]:
                        c[i, j] = pin_val
                        continue
                    jm = j - 1 if j > 0 else 0
                    jp = j + 1 if j < ny - 1 else ny - 1
                    s = c[im, j] + c[ip, j] + c[i, jm] + c[i, jp]
                    new = (s + src[i, j]) / (4.0 + k[i, j])
                    c[i, j] += omega * (new - c[i, j])


@njit(cache=True)
def build_coeffs(state, P, k_o, k_c, src_c, k_s, src_s, vessel):
    nx, ny = state.shape
    kc = P[P_K_C]
    ks = P[P_K_S]
    for i in range(nx):
        for j in range(ny):
            st = state[i, j]
            live = st == 1 or st == 2
            vessel[i, j] = st == VESSEL
            k_o[i, j] = P[P_K_O_TUMOR] if live else P[P_K_O_TISSUE]
            k_c[i, j] = kc
            k_s[i, j] = ks
            # sources scaled by decay so bulk steady-state ~ secretion value
            if live:
                src_c[i, j] = P[P_CHEMO_SECRETE] * kc
                src_s[i, j] = P[P_SUPP_SECRETE] * ks
            elif st == 4:
                src_c[i, j] = P[P_CHEMO_APOP] * kc
                src_s[i, j] = 0.0
            else:
                src_c[i, j] = 0.0
                src_s[i, j] = 0.0


class Fields:
    def __init__(self, nx, ny):
        self.O = np.ones((nx, ny))
        self.C = np.zeros((nx, ny))
        self.S = np.zeros((nx, ny))
        self.G = np.zeros((nx, ny))  # IFN-gamma
        self.Q = np.zeros((nx, ny))  # CCL2-like macrophage chemoattractant
        self.k_q = np.zeros((nx, ny))
        self.src_q = np.zeros((nx, ny))
        self.k_g = np.zeros((nx, ny))
        self.src_g = np.zeros((nx, ny))
        self.k_o = np.zeros((nx, ny))
        self.k_c = np.zeros((nx, ny))
        self.k_s = np.zeros((nx, ny))
        self.src_c = np.zeros((nx, ny))
        self.src_s = np.zeros((nx, ny))
        self.zero = np.zeros((nx, ny))
        self.vessel = np.zeros((nx, ny), dtype=np.bool_)
        self.nopin = np.zeros((nx, ny), dtype=np.bool_)

    def update(self, state, P, sweeps, omega=1.7, tact=None, m1=None, m2=None):
        build_coeffs(state, P, self.k_o, self.k_c, self.src_c, self.k_s, self.src_s, self.vessel)
        if m1 is not None:
            self.src_c += m1 * (P[P_M1_CHEMO] * P[P_K_C])
            self.src_s += m2 * (P[P_M2_SUPP] * P[P_K_S])
        live = (state == 1) | (state == 2)
        self.k_q[:] = P[P_K_CCL2]
        np.multiply(live & (self.O < P[P_O_HYP]), P[P_CCL2_SECRETE] * P[P_K_CCL2], out=self.src_q)
        _sor(self.O, self.k_o, self.zero, self.vessel, 1.0, sweeps, omega)
        _sor(self.C, self.k_c, self.src_c, self.nopin, 0.0, sweeps, omega)
        _sor(self.S, self.k_s, self.src_s, self.nopin, 0.0, sweeps, omega)
        _sor(self.Q, self.k_q, self.src_q, self.nopin, 0.0, sweeps, omega)
        if tact is not None:
            self.k_g[:] = P[P_K_IFN]
            np.multiply(tact, P[P_IFN_SECRETE] * P[P_K_IFN], out=self.src_g)
            _sor(self.G, self.k_g, self.src_g, self.nopin, 0.0, sweeps, omega)
