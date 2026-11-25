"""Simulation driver: builds initial conditions, steps the model, writes snapshots + time series."""
import json
import os
import shutil
import time

import numpy as np
import yaml
import zarr
from scipy import ndimage

from . import agents as A
from .fields import Fields
from .outcome import classify
from .params import P_DT, P_E_PDL1, P_IFN_HALF, P_PDL1_IFN, P_P_BLOCK, P_T_PRIME, load_config, param_array


def _init(cfg, rng):
    g = cfg["grid"]
    nx, ny = g["nx"], g["ny"]
    state = np.zeros((nx, ny), np.int8)
    # vessels: random, but not inside the seed region
    vmask = rng.random((nx, ny)) < g["vessel_density"]
    ii, jj = np.indices((nx, ny))
    r2 = (ii - nx // 2) ** 2 + (jj - ny // 2) ** 2
    vmask &= r2 > (g["seed_radius"] + 3) ** 2
    state[vmask] = A.VESSEL
    seed = r2 <= g["seed_radius"] ** 2
    state[seed] = A.PROLIF
    lineage = np.zeros((nx, ny), np.int32)
    lineage[seed] = 1
    ini = cfg["init"]
    antigen = np.zeros((nx, ny))
    pdl1 = np.zeros((nx, ny))
    antigen[seed] = np.clip(rng.normal(ini["antigen0"], ini["antigen_sd"], seed.sum()), 0, 1)
    pdl1[seed] = np.clip(rng.normal(ini["pdl1_0"], ini["pdl1_sd"], seed.sum()), 0, 1)
    vx, vy = np.nonzero(vmask)
    return state, lineage, antigen, pdl1, vx.astype(np.int64), vy.astype(np.int64)


def run(config_path, seed=1, name=None, out_root="runs", overrides=None, verbose=True):
    cfg = load_config(config_path)
    if overrides:
        for k, v in overrides.items():
            sec, key = k.split(".")
            cfg.setdefault(sec, {})[key] = v
    name = name or f"{cfg.get('name', 'run')}_s{seed}"
    out = os.path.join(out_root, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    cfg["seed"] = seed
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    rng = np.random.default_rng(seed)
    A.seed_rng(seed)
    P = param_array(cfg)
    rc = cfg["run"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state, lineage, antigen, pdl1, vx, vy = _init(cfg, rng)
    timer = np.zeros((nx, ny))
    born = np.full((nx, ny), -1, np.int64)
    perm = np.arange(nx * ny, dtype=np.int64)

    cap = int(rc["t_cap"])
    tx = np.zeros(cap, np.int64); ty = np.zeros(cap, np.int64)
    talive = np.zeros(cap, np.int8); texh = np.zeros(cap); tkills = np.zeros(cap, np.int64)
    tage = np.zeros(cap); tdiv = np.zeros(cap, np.int64)
    tgrid = np.full((nx, ny), -1, np.int64)
    ctr = np.zeros(4, np.int64)
    ctr[1] = 2  # next lineage id
    kill_x = np.zeros(20000, np.int64); kill_y = np.zeros(20000, np.int64)

    # macrophages: tissue-resident population seeded on empty tissue sites
    mc = cfg.get("macrophages") or {}
    mcap = int(mc.get("cap", 20000))
    mx = np.zeros(mcap, np.int64); my = np.zeros(mcap, np.int64)
    malive = np.zeros(mcap, np.int8); mpol = np.zeros(mcap); mage = np.zeros(mcap)
    mgrid = np.full((nx, ny), -1, np.int64)
    mctr = np.zeros(2, np.int64)
    res = (state == A.EMPTY) & (rng.random((nx, ny)) < float(mc.get("resident_density", 0.0)))
    for x, y in zip(*np.nonzero(res)):
        A._add_mac(int(x), int(y), float(mc.get("m0", 0.3)), mx, my, malive, mpol, mage, mgrid, mctr, mcap)
    m1map = np.zeros((nx, ny)); m2map = np.zeros((nx, ny)); menc = np.zeros((nx, ny), np.uint8)
    A.mac_maps(mx, my, malive, mpol, mctr, m1map, m2map, menc)

    fields = Fields(nx, ny)
    fields.update(state, P, 400, m1=m1map, m2=m2map)  # settle initial fields
    tact = np.zeros((nx, ny))

    dt = P[P_DT]
    n_steps = int(round(rc["days"] * 24 / dt))
    fe = rc["frame_every"]
    n_frames = n_steps // fe + 1

    store = zarr.open_group(os.path.join(out, "snap.zarr"), mode="w")
    z_state = store.create_array("state", shape=(n_frames, nx, ny), chunks=(1, nx, ny), dtype="i1")
    z_lin = store.create_array("lineage", shape=(n_frames, nx, ny), chunks=(1, nx, ny), dtype="i4")
    z_t = store.create_array("tcell", shape=(n_frames, nx, ny), chunks=(1, nx, ny), dtype="i1")
    z_m = store.create_array("mac", shape=(n_frames, nx, ny), chunks=(1, nx, ny), dtype="u1")
    z_o = store.create_array("oxygen", shape=(n_frames, nx, ny), chunks=(1, nx, ny), dtype="f2")
    kills_all = []  # (frame, x, y)

    ts = {k: [] for k in ["day", "tumor_live", "prolif", "quies", "necrotic", "apoptotic",
                          "t_active", "t_exhausted", "t_in_tumor", "t_margin", "kills_cum",
                          "antigen_mean", "pdl1_mean", "pdl1_eff_mean", "n_lineages", "radius_um",
                          "mac_total", "mac_m1", "mac_m2", "mac_pol_mean", "vessels"]}
    kills_cum = 0

    def snapshot(frame, step):
        nonlocal kills_cum
        tl = np.zeros((nx, ny), np.int8)
        n = ctr[0]
        alive = talive[:n] == 1
        ex = texh[:n] >= 1.0
        tl[tx[:n][alive & ~ex], ty[:n][alive & ~ex]] = 1
        tl[tx[:n][alive & ex], ty[:n][alive & ex]] = 2
        z_state[frame] = state
        z_lin[frame] = lineage
        z_t[frame] = tl
        A.mac_maps(mx, my, malive, mpol, mctr, m1map, m2map, menc)
        z_m[frame] = menc
        nm = mctr[0]
        pol = mpol[:nm][malive[:nm] == 1]
        ts["mac_total"].append(int(pol.size))
        ts["mac_m1"].append(int((pol < 0.5).sum()))
        ts["mac_m2"].append(int((pol >= 0.5).sum()))
        ts["mac_pol_mean"].append(round(float(pol.mean()), 4) if pol.size else None)
        ts["vessels"].append(int((state == A.VESSEL).sum()))
        z_o[frame] = fields.O.astype(np.float16)
        k = ctr[2]
        for q in range(k):
            kills_all.append((frame, int(kill_x[q]), int(kill_y[q])))
        kills_cum += int(k)
        ctr[2] = 0
        live = (state == A.PROLIF) | (state == A.QUIES)
        mass = live | (state == A.NECRO) | (state == A.APOP)
        nlive = int(live.sum())
        tum_region = ndimage.binary_fill_holes(ndimage.binary_closing(mass, iterations=2))
        margin = ndimage.binary_dilation(tum_region, iterations=4) & ~tum_region
        act = tl == 1
        ts["day"].append(round(step * dt / 24, 4))
        ts["tumor_live"].append(nlive)
        ts["prolif"].append(int((state == A.PROLIF).sum()))
        ts["quies"].append(int((state == A.QUIES).sum()))
        ts["necrotic"].append(int((state == A.NECRO).sum()))
        ts["apoptotic"].append(int((state == A.APOP).sum()))
        ts["t_active"].append(int(act.sum()))
        ts["t_exhausted"].append(int((tl == 2).sum()))
        ts["t_in_tumor"].append(int((act & tum_region).sum()))
        ts["t_margin"].append(int((act & margin).sum()))
        ts["kills_cum"].append(kills_cum)
        ts["antigen_mean"].append(round(float(antigen[live].mean()), 4) if nlive else None)
        ts["pdl1_mean"].append(round(float(pdl1[live].mean()), 4) if nlive else None)
        if nlive:
            g = fields.G[live]
            pe = np.minimum(1.0, pdl1[live] + P[P_PDL1_IFN] * g / (g + P[P_IFN_HALF]))
            ts["pdl1_eff_mean"].append(round(float(pe.mean()), 4))
        else:
            ts["pdl1_eff_mean"].append(None)
        ts["n_lineages"].append(int(len(np.unique(lineage[live]))) if nlive else 0)
        area = tum_region.sum()
        ts["radius_um"].append(round(float(np.sqrt(area / np.pi) * 15.0), 1))
        return nlive

    t0 = time.time()
    snapshot(0, 0)
    frame = 1
    t_prime = P[P_T_PRIME]
    # anti-PD-1: from treatment day, PD-L1 inhibition of killing and PD-L1-driven exhaustion
    # are both reduced by the drug efficacy (antibody occupies PD-1 on T cells)
    tr = cfg.get("treatment") or {}
    t_treat = float(tr["day"]) * 24 if tr.get("day") is not None else None
    treated = False
    for step in range(1, n_steps + 1):
        if t_treat is not None and not treated and step * dt >= t_treat:
            eff = float(tr.get("drug_eff", 0.9))
            P[P_P_BLOCK] *= (1.0 - eff)
            P[P_E_PDL1] *= (1.0 - eff)
            treated = True
            if verbose:
                print(f"  >> anti-PD-1 started day {step*dt/24:.1f} (efficacy {eff})", flush=True)
        A.tumor_step(state, lineage, antigen, pdl1, timer, born, perm, fields.O, P, step, ctr)
        if step * dt >= t_prime:
            A.recruit(state, fields.C, vx, vy, tx, ty, talive, texh, tkills, tage, tdiv, tgrid,
                      P, ctr, cap)
        A.tcell_step(state, antigen, pdl1, timer, fields.O, fields.C, fields.S, fields.G, tx, ty, talive,
                     texh, tkills, tage, tdiv, tgrid, P, ctr, kill_x, kill_y, cap)
        A.mac_recruit(state, fields.Q, vx, vy, mx, my, malive, mpol, mage, mgrid, P, mctr, mcap)
        A.mac_step(state, lineage, timer, fields.O, fields.Q, fields.S, fields.G, mx, my, malive, mpol,
                   mage, mgrid, P, mctr, ctr, kill_x, kill_y, step, born)
        if step % 50 == 0:
            A.compact(tx, ty, talive, texh, tkills, tage, tdiv, tgrid, ctr)
            A.mac_compact(mx, my, malive, mpol, mage, mgrid, mctr)
        if step % rc["field_every"] == 0:
            A.active_tcell_map(tx, ty, talive, texh, ctr, tact)
            A.mac_maps(mx, my, malive, mpol, mctr, m1map, m2map, menc)
            fields.update(state, P, rc["field_sweeps"], tact=tact, m1=m1map, m2=m2map)
            # vessel list changes (regression, M2 angiogenesis)
            vx, vy = (a.astype(np.int64) for a in np.nonzero(state == A.VESSEL))
        if step % fe == 0:
            nlive = snapshot(frame, step)
            frame += 1
            if verbose and frame % 96 == 0:
                print(f"  day {step*dt/24:5.1f}  tumor {nlive:6d}  T {ts['t_active'][-1]:5d}"
                      f"/{ts['t_exhausted'][-1]:4d}  Mac {ts['mac_m1'][-1]}/{ts['mac_m2'][-1]}  kills {kills_cum:6d}  "
                      f"[{time.time()-t0:5.1f}s]", flush=True)
    kills = np.array(kills_all, np.int32).reshape(-1, 3)
    np.save(os.path.join(out, "kills.npy"), kills)
    store.attrs["n_frames"] = frame
    store.attrs["frame_hours"] = fe * dt
    outcome = classify(ts["tumor_live"])
    meta = {"name": name, "seed": seed, "outcome": outcome, "n_frames": frame,
            "frame_hours": fe * dt, "runtime_s": round(time.time() - t0, 1),
            "scenario": cfg.get("name"), "description": cfg.get("description", ""),
            "treat_day": (t_treat / 24 if t_treat is not None else None),
            "drug_eff": (float(tr.get("drug_eff", 0.9)) if t_treat is not None else None)}
    with open(os.path.join(out, "timeseries.json"), "w") as f:
        json.dump({"meta": meta, "series": ts}, f)
    if verbose:
        print(f"done {name}: outcome={outcome} runtime={meta['runtime_s']}s")
    return out
