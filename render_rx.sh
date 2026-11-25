#!/bin/zsh
# Render the anti-PD-1 scenario videos.
cd "$(dirname "$0")"
for r in pdl1_high_s1 pdl1_high_rx_s1 pdl1_high_late_s1 cold_rx_s1; do
  .venv/bin/python -m viz.render runs/$r --workers 8 || echo "FAILED $r"
done
echo ALL_DONE
