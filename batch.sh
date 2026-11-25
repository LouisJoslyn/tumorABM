#!/bin/zsh
# Run configs x seeds in parallel into a runs folder:  ./batch.sh <out_root> "<cfg1 cfg2 ...>" "<seeds>" [extra run.py args]
cd "$(dirname "$0")"
out=$1; cfgs=(${=2}); seeds=(${=3}); shift 3
mkdir -p $out
for c in $cfgs; do for s in $seeds; do
  .venv/bin/python run.py config/$c.yaml --seed $s --out $out "$@" > $out/${c}_s$s.log 2>&1 &
done; done
wait
.venv/bin/python summarize.py $out
