"""Numba kernels for tumor cells (per-site arrays) and CD8 T cells (agent list + occupancy layer).

Tumor state codes: 0 empty, 1 proliferating, 2 quiescent, 3 necrotic, 4 apoptotic, 5 vessel.
"""
import numpy as np
from numba import njit

from .params import (P_DT, P_T_DIV, P_T_DEATH, P_O_Q, P_O_N, P_MU, P_MUT_SD, P_VESSEL_REGRESS,
                     P_T_CLEAR, P_T_NECRO_CLEAR, P_T_LIFE, P_T_LIFE_EX, P_MOVE_PROB, P_TUMOR_DRAG,
                     P_S_MOTILITY, P_CHI, P_K_KILL, P_P_BLOCK, P_S_BLOCK, P_E_KILL, P_E_PDL1,
                     P_P_PROLIF, P_MAX_DIV, P_R_RECRUIT)

EMPTY, PROLIF, QUIES, NECRO, APOP, VESSEL = 0, 1, 2, 3, 4, 5

# Moore neighbourhood (8) followed by the radius-2 ring (16): 24 offsets
_DI = []
_DJ = []
for _r in (1, 2):
    for _a in range(-_r, _r + 1):
        for _b in range(-_r, _r + 1):
            if max(abs(_a), abs(_b)) == _r:
                _DI.append(_a)
                _DJ.append(_b)
DI = np.array(_DI, dtype=np.int64)
DJ = np.array(_DJ, dtype=np.int64)


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(cache=True)
def tumor_step(state, lineage, antigen, pdl1, timer, born, perm, O, P, step_i, ctr):
    """One asynchronous sweep over all sites in random order. ctr[1] = next lineage id."""
    nx, ny = state.shape
    n = nx * ny
    dt = P[P_DT]
    p_div = dt / P[P_T_DIV]
    p_death = dt / P[P_T_DEATH]
    # Fisher-Yates reshuffle of update order
    for k in range(n - 1, 0, -1):
        r = np.random.randint(0, k + 1)
        tmp = perm[k]
        perm[k] = perm[r]
        perm[r] = tmp
    cand = np.empty(24, dtype=np.int64)
    for k in range(n):
        idx = perm[k]
        i = idx // ny
        j = idx % ny
        st = state[i, j]
        if st == EMPTY:
            continue
        if st == APOP:
            timer[i, j] += dt
            if timer[i, j] > P[P_T_CLEAR]:
                state[i, j] = EMPTY
                lineage[i, j] = 0
            continue
        if st == NECRO:
            if np.random.random() < dt / P[P_T_NECRO_CLEAR]:
                state[i, j] = EMPTY
                lineage[i, j] = 0
            continue
        if st == VESSEL:
            # vessel regresses when engulfed by tumor
            if np.random.random() < P[P_VESSEL_REGRESS]:
                nt = 0
                for q in range(8):
                    a = i + DI[q]
                    b = j + DJ[q]
                    if 0 <= a < nx and 0 <= b < ny:
                        s2 = state[a, b]
                        if s2 == PROLIF or s2 == QUIES or s2 == NECRO:
                            nt += 1
                if nt >= 7:
                    state[i, j] = EMPTY
            continue
        # live tumor cell
        if born[i, j] == step_i:
            continue
        o = O[i, j]
        if o < P[P_O_N]:
            state[i, j] = NECRO
            timer[i, j] = 0.0
            continue
        if np.random.random() < p_death:
            state[i, j] = APOP
            timer[i, j] = P[P_T_CLEAR] * 0.5  # quiet death: shorter flash
            continue
        if o < P[P_O_Q]:
            state[i, j] = QUIES
            continue
        # space available within radius 2?
        nc = 0
        for q in range(8):
            a = i + DI[q]
            b = j + DJ[q]
            if 0 <= a < nx and 0 <= b < ny and state[a, b] == EMPTY:
                cand[nc] = q
                nc += 1
        if nc == 0:
            # Moore full: look at ring 2
            for q in range(8, 24):
                a = i + DI[q]
                b = j + DJ[q]
                if 0 <= a < nx and 0 <= b < ny and state[a, b] == EMPTY:
                    cand[nc] = q
                    nc += 1
        if nc == 0:
            state[i, j] = QUIES
            continue
        state[i, j] = PROLIF
        if np.random.random() < p_div:
            q = cand[np.random.randint(0, nc)]
            a = i + DI[q]
            b = j + DJ[q]
            state[a, b] = PROLIF
            born[a, b] = step_i
            timer[a, b] = 0.0
            if np.random.random() < P[P_MU]:
                lineage[a, b] = ctr[1]
                ctr[1] += 1
                antigen[a, b] = _clip01(antigen[i, j] + np.random.normal() * P[P_MUT_SD])
                pdl1[a, b] = _clip01(pdl1[i, j] + np.random.normal() * P[P_MUT_SD])
            else:
                lineage[a, b] = lineage[i, j]
                antigen[a, b] = antigen[i, j]
                pdl1[a, b] = pdl1[i, j]


@njit(cache=True)
def _add_tcell(x, y, e, d, tx, ty, talive, texh, tkills, tage, tdiv, tgrid, ctr, cap):
    n = ctr[0]
    if n >= cap:
        return
    tx[n] = x
    ty[n] = y
    talive[n] = 1
    texh[n] = e
    tkills[n] = 0
    tage[n] = 0.0
    tdiv[n] = d
    tgrid[x, y] = n
    ctr[0] = n + 1


@njit(cache=True)
def tcell_step(state, antigen, pdl1, timer, O, C, S, tx, ty, talive, texh, tkills, tage, tdiv,
               tgrid, P, ctr, kill_x, kill_y, cap):
    """ctr[0] = number of T cell slots in use, ctr[2] = kills recorded this frame."""
    nx, ny = state.shape
    dt = P[P_DT]
    n = ctr[0]
    order = np.random.permutation(n)
    w = np.empty(8, dtype=np.float64)
    cand = np.empty(9, dtype=np.int64)
    for oi in range(n):
        t = order[oi]
        if talive[t] == 0:
            continue
        x = tx[t]
        y = ty[t]
        tage[t] += dt
        exhausted = texh[t] >= 1.0
        life = P[P_T_LIFE_EX] if exhausted else P[P_T_LIFE]
        if np.random.random() < dt / life:
            talive[t] = 0
            tgrid[x, y] = -1
            continue
        if not exhausted:
            # gather live tumor targets: own site (index 8) + Moore neighbours
            nc = 0
            for q in range(9):
                if q == 8:
                    a = x
                    b = y
                else:
                    a = x + DI_M[q]
                    b = y + DJ_M[q]
                if 0 <= a < nx and 0 <= b < ny:
                    s2 = state[a, b]
                    if s2 == PROLIF or s2 == QUIES:
                        cand[nc] = q
                        nc += 1
            if nc > 0:
                q = cand[np.random.randint(0, nc)]
                if q == 8:
                    a = x
                    b = y
                else:
                    a = x + DI_M[q]
                    b = y + DJ_M[q]
                p = pdl1[a, b]
                texh[t] += P[P_E_PDL1] * p * dt
                pk = (P[P_K_KILL] * antigen[a, b] * (1.0 - P[P_P_BLOCK] * p)
                      * (1.0 - P[P_S_BLOCK] * min(S[x, y], 1.0)))
                if np.random.random() < pk:
                    state[a, b] = APOP
                    timer[a, b] = 0.0
                    tkills[t] += 1
                    texh[t] += P[P_E_KILL]
                    if ctr[2] < kill_x.shape[0]:
                        kill_x[ctr[2]] = a
                        kill_y[ctr[2]] = b
                        ctr[2] += 1
                    # clonal expansion
                    if tdiv[t] < P[P_MAX_DIV] and np.random.random() < P[P_P_PROLIF]:
                        start = np.random.randint(0, 8)
                        for qq in range(8):
                            q2 = (start + qq) % 8
                            a2 = x + DI_M[q2]
                            b2 = y + DJ_M[q2]
                            if 0 <= a2 < nx and 0 <= b2 < ny and tgrid[a2, b2] < 0:
                                tdiv[t] += 1
                                _add_tcell(a2, b2, texh[t] * 0.5, tdiv[t], tx, ty, talive,
                                           texh, tkills, tage, tdiv, tgrid, ctr, cap)
                                break
                # engaged with a target: stay put this step
                continue
        # movement: biased random walk up the chemokine gradient
        mp = P[P_MOVE_PROB] * (1.0 - P[P_S_MOTILITY] * min(S[x, y], 1.0))
        if np.random.random() >= mp:
            continue
        ch = C[x, y]
        tot = 0.0
        for q in range(8):
            a = x + DI_M[q]
            b = y + DJ_M[q]
            if 0 <= a < nx and 0 <= b < ny and tgrid[a, b] < 0:
                wq = np.exp(P[P_CHI] * (C[a, b] - ch) / (ch + 0.05))
                s2 = state[a, b]
                if s2 == PROLIF or s2 == QUIES or s2 == NECRO:
                    wq *= P[P_TUMOR_DRAG]
                w[q] = wq
                tot += wq
            else:
                w[q] = 0.0
        if tot <= 0.0:
            continue
        r = np.random.random() * tot
        acc = 0.0
        for q in range(8):
            acc += w[q]
            if r < acc and w[q] > 0.0:
                a = x + DI_M[q]
                b = y + DJ_M[q]
                tgrid[x, y] = -1
                tgrid[a, b] = t
                tx[t] = a
                ty[t] = b
                break


DI_M = DI[:8].copy()
DJ_M = DJ[:8].copy()


@njit(cache=True)
def recruit(state, C, vx, vy, tx, ty, talive, texh, tkills, tage, tdiv, tgrid, P, ctr, cap):
    nx, ny = state.shape
    p0 = P[P_R_RECRUIT] * P[P_DT]
    for v in range(vx.shape[0]):
        x = vx[v]
        y = vy[v]
        if state[x, y] != VESSEL:
            continue
        if np.random.random() < p0 * C[x, y]:
            if tgrid[x, y] < 0:
                _add_tcell(x, y, 0.0, 0, tx, ty, talive, texh, tkills, tage, tdiv, tgrid, ctr, cap)
            else:
                start = np.random.randint(0, 8)
                for qq in range(8):
                    q = (start + qq) % 8
                    a = x + DI_M[q]
                    b = y + DJ_M[q]
                    if 0 <= a < nx and 0 <= b < ny and tgrid[a, b] < 0:
                        _add_tcell(a, b, 0.0, 0, tx, ty, talive, texh, tkills, tage, tdiv,
                                   tgrid, ctr, cap)
                        break


@njit(cache=True)
def compact(tx, ty, talive, texh, tkills, tage, tdiv, tgrid, ctr):
    n = ctr[0]
    m = 0
    for t in range(n):
        if talive[t]:
            if m != t:
                tx[m] = tx[t]
                ty[m] = ty[t]
                talive[m] = 1
                texh[m] = texh[t]
                tkills[m] = tkills[t]
                tage[m] = tage[t]
                tdiv[m] = tdiv[t]
                tgrid[tx[m], ty[m]] = m
            m += 1
    for t in range(m, n):
        talive[t] = 0
    ctr[0] = m
