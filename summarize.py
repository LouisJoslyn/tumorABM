"""One-line outcome summary per run:  python summarize.py [runs_dir] [name_filter]"""
import glob
import json
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
filt = sys.argv[2] if len(sys.argv) > 2 else ""
for r in sorted(glob.glob(f"{root}/*/timeseries.json")):
    d = json.load(open(r)); s = d["series"]; m = d["meta"]
    if filt not in m["name"]:
        continue
    mx = lambda k: max([x for x in s.get(k, []) if x is not None], default=None)
    print(f'{m["name"]:22s} {m["outcome"]:12s}', [s["tumor_live"][k] for k in range(0, len(s["day"]), 192)],
          "Tmax", mx("t_active"), "PDL1eff", mx("pdl1_eff_mean"), "M1/M2 end",
          (s.get("mac_m1") or [None])[-1], (s.get("mac_m2") or [None])[-1])
