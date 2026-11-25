#!/bin/zsh
# Render every scenario video used by the dashboard.
cd "$(dirname "$0")"
for r in hot_s1 hot_s2 hot_s3 cold_s1 cold_s2 edited_s1 edited_s3; do
  .venv/bin/python -m viz.render runs/$r --workers 8
done
.venv/bin/python -m viz.render runs/edited_s1 --mode lineage --workers 8
.venv/bin/python -m viz.render runs/hot_s2 --mode lineage --workers 8
echo ALL_DONE
