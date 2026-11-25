"""Collect rendered runs into dashboard/data/ and write dashboard/data.js (works from file:// too).

Usage: python dashboard/build.py   (from the tumor_abm folder)
"""
import glob
import json
import os
import shutil

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "dashboard", "data")

SCENARIOS = {
    "hot": {"group": "natural", "title": "Hot", "tag": "Inflamed",
            "blurb": "Highly antigenic tumor with a strong chemokine signal and little suppression. "
                     "T cells arrive in large numbers, gather at the margin and break the tumor into fragments."},
    "cold": {"group": "natural", "title": "Cold", "tag": "Immune desert",
             "blurb": "Poorly antigenic tumor with a weak chemokine signal in a suppressive (TGF-β-like) environment. "
                      "Few T cells arrive, they move and kill poorly, and the tumor keeps growing."},
    "edited": {"group": "natural", "title": "Immunoediting", "tag": "Equilibrium → escape",
               "blurb": "Intermediate inflammation with late priming. The tumor is knocked back and held, "
                        "then regrows from clones with lower antigenicity."},
    "pdl1_high": {"group": "therapy", "title": "PD-L1-high", "tag": "Untreated",
                  "blurb": "Inflamed tumor with high PD-L1 (0.65). T cells infiltrate but are blocked and exhausted, "
                           "and the tumor escapes."},
    "pdl1_high_rx": {"group": "therapy", "title": "PD-L1-high + anti-PD-1", "tag": "Day 14",
                     "blurb": "Same tumor, anti-PD-1 from day 14 (blocks 90% of PD-1/PD-L1 signalling). "
                              "T cells resume killing and the tumor is eliminated."},
    "pdl1_high_late": {"group": "therapy", "title": "PD-L1-high + late anti-PD-1", "tag": "Day 24",
                       "blurb": "Treatment starts at day 24, when the tumor is about 20k cells. T cells already inside "
                                "the tumor resume killing and clear it."},
    "cold_rx": {"group": "therapy", "title": "Cold + anti-PD-1", "tag": "Day 14",
                "blurb": "Cold tumor with the same treatment. PD-L1 was not the main barrier (low antigen, few T cells, "
                         "suppression), so the tumor still escapes."},
}
GROUPS = {"natural": "Natural history", "therapy": "Anti-PD-1 therapy"}
KEY_PARAMS = [
    ("init", "antigen0", "Founder antigenicity"),
    ("init", "pdl1_0", "Founder PD-L1"),
    ("params", "chemo_secrete", "Chemokine secretion"),
    ("params", "supp_secrete", "Suppressor (TGF-β) secretion"),
    ("params", "k_kill", "T-cell kill strength"),
    ("params", "t_prime", "Priming delay (h)"),
    ("params", "mu", "Mutation prob / division"),
    ("params", "mut_sd", "Mutation step size"),
    ("treatment", "day", "Anti-PD-1 start day"),
]
SERIES = ["day", "tumor_live", "prolif", "quies", "necrotic", "t_active", "t_exhausted",
          "t_in_tumor", "t_margin", "kills_cum", "antigen_mean", "pdl1_mean", "n_lineages", "radius_um"]
STRIDE = 2  # keep every 2nd frame (1 h resolution) for charts


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    runs = []
    for rd in sorted(glob.glob(os.path.join(RUNS, "*_s*"))):
        if not os.path.exists(os.path.join(rd, "video.mp4")):
            continue
        name = os.path.basename(rd)
        with open(os.path.join(rd, "timeseries.json")) as f:
            ts = json.load(f)
        with open(os.path.join(rd, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        scen = cfg.get("name")
        if scen not in SCENARIOS:
            continue
        dst = os.path.join(OUT, name)
        os.makedirs(dst)
        videos = {}
        for mode, fn in [("state", "video.mp4"), ("lineage", "video_lineage.mp4")]:
            if os.path.exists(os.path.join(rd, fn)):
                shutil.copy(os.path.join(rd, fn), dst)
                poster = fn.replace(".mp4", "_poster.jpg")
                if os.path.exists(os.path.join(rd, poster)):
                    shutil.copy(os.path.join(rd, poster), dst)
                videos[mode] = {"src": f"data/{name}/{fn}", "poster": f"data/{name}/{poster}"}
        s = ts["series"]
        series = {k: s[k][::STRIDE] for k in SERIES}
        runs.append({
            "name": name, "scenario": scen, "seed": cfg["seed"], "outcome": ts["meta"]["outcome"],
            "frame_hours": ts["meta"]["frame_hours"], "n_frames": ts["meta"]["n_frames"],
            "videos": videos, "series": series,
            "peak_tumor": max(s["tumor_live"]), "final_tumor": s["tumor_live"][-1],
            "peak_t": max(s["t_active"]), "total_kills": s["kills_cum"][-1],
            "treat_day": ts["meta"].get("treat_day"),
        })
    params = {}
    for scen in SCENARIOS:
        cfgs = [r for r in runs if r["scenario"] == scen]
        if not cfgs:
            continue
        with open(os.path.join(RUNS, cfgs[0]["name"], "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        params[scen] = {key: (cfg.get(sec) or {}).get(key) for sec, key, _ in KEY_PARAMS}
    data = {"scenarios": SCENARIOS, "groups": GROUPS, "runs": runs, "params": params,
            "param_labels": [[k, lbl] for _, k, lbl in KEY_PARAMS],
            "fps": 30, "grid_mm": 300 * 15 / 1000}
    with open(os.path.join(ROOT, "dashboard", "data.js"), "w") as f:
        f.write("window.ABM_DATA = ")
        json.dump(data, f)
        f.write(";\n")
    print(f"wrote {len(runs)} runs:", ", ".join(f"{r['name']}({r['outcome']})" for r in runs))


if __name__ == "__main__":
    main()
