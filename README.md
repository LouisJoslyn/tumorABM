# Tumor–Immune ABM

Stochastic 2D agent-based model of tumor growth and CD8 T-cell killing (Python + Numba).
The design is in [SPEC.md](SPEC.md).

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install --only-binary=:all: numba llvmlite
.venv/bin/pip install --prefer-binary numpy scipy zarr matplotlib pyyaml imageio imageio-ffmpeg
```

## Run a simulation
```bash
.venv/bin/python run.py config/hot.yaml --seed 1                  # -> runs/hot_s1/
.venv/bin/python run.py config/cold.yaml --seed 2 --days 30
.venv/bin/python run.py config/hot.yaml --name test --set params.k_kill=0.05 --set init.antigen0=0.6
.venv/bin/python run.py config/pdl1_high.yaml --treat 14                 # anti-PD-1 from day 14
./batch.sh runs "hot cold" "1 2 3"                                      # configs x seeds in parallel + summary
.venv/bin/python summarize.py runs                                      # one-line outcome per run
```

## Render
```bash
./render_all.sh                                                   # every video the dashboard uses
```
```bash
.venv/bin/python -m viz.render runs/hot_s1                        # video.mp4 + poster
.venv/bin/python -m viz.render runs/hot_s1 --mode lineage         # colour by clone
.venv/bin/python -m viz.render runs/hot_s1 --stills 10,20,30      # PNG checks
```

## Dashboard
```bash
.venv/bin/python dashboard/build.py
python3 dashboard/serve.py                                        # open http://localhost:8777 (supports video seeking)
```
(`dashboard/index.html` also opens directly from disk; the data is in `data.js`.)
