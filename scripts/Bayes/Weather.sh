#!/bin/bash
# Bayesian HP search for TimeTucker on Weather
# Weather has 21 channels at 10-min sampling, period_len=144 (=24h) is common.
# Per agreed spec, r_p == period_len (so r_p=144 here unless user changes period_len).
model_name=TimeTucker
root_path_name=../dataset/
data_path_name=weather.csv
data_name=Weather
seq_len=720
period_len=4
enc_in=21
gpu=${GPU:-0}

dir=./logs/Bayes
mkdir -p "$dir"

for pred_len in 96 192 336 720; do
    python -u run_bayes.py \
        --model "$model_name" \
        --data "$data_name" \
        --root_path "$root_path_name" \
        --data_path "$data_path_name" \
        --features M \
        --seq_len "$seq_len" \
        --pred_len "$pred_len" \
        --period_len "$period_len" \
        --enc_in "$enc_in" \
        --train_epochs 10 \
        --patience 3 \
        --use_revin 1 \
        --share_factors 1 \
        --use_orthogonal 1 \
        --init_points 15 \
        --n_iter 45 \
        --gpu "$gpu" \
        --retrain 1 \
        --retrain_epochs 30 \
        --retrain_patience 5 \
        --result_dir "$dir" \
        > "$dir/${data_name}_${seq_len}_${pred_len}.log" 2>&1
done
