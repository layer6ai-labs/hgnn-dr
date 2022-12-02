#!/bin/bash
set -e

__DIR__=$(dirname $(readlink -f "$0"))
save_metrics="$__DIR__/../data/baseline_metrics.xlsx"

base_seed=1234
base_split=4567

rm -f $save_metrics
for i in {0..9}; do
    split=$(($base_split + $i))
    seed=$(($base_seed + $i))

    clear
    echo -e "\n$i: SPLIT $split, SEED $seed\n================================\n"

    python $__DIR__/build_dataset.py --network_split $split --random_state $seed
    python $__DIR__/build_models.py --network_split $split --random_state $seed --save_metrics_xlsx $save_metrics
done
