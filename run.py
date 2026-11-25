"""CLI: python run.py config/hot.yaml --seed 1 [--name hot_s1] [--days 30] [--set params.k_kill=0.2]"""
import argparse

from model.sim import run

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--name")
ap.add_argument("--out", default="runs")
ap.add_argument("--days", type=float)
ap.add_argument("--treat", type=float, help="start anti-PD-1 on this day")
ap.add_argument("--set", action="append", default=[], help="section.key=value")
a = ap.parse_args()
ov = {}
if a.days:
    ov["run.days"] = a.days
if a.treat is not None:
    ov["treatment.day"] = a.treat
for s in a.set:
    k, v = s.split("=")
    ov[k] = float(v)
run(a.config, seed=a.seed, name=a.name, overrides=ov, out_root=a.out)
