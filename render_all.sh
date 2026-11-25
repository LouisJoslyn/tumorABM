#!/bin/zsh
# Render every video the dashboard uses (run after the simulations in runs/).
cd "$(dirname "$0")"
for r in hot_s1 hot_s2 hot_s3 cold_s1 edited_s2 edited_s3 pdl1_high_s1 pdl1_high_rx_s1 pdl1_high_late_s1 cold_rx_s1; do
  .venv/bin/python -m viz.render runs/$r --workers 8 || echo "FAILED $r"
done
for r in edited_s3 hot_s2; do
  .venv/bin/python -m viz.render runs/$r --mode lineage --workers 8 || echo "FAILED $r lineage"
done
echo ALL_DONE
