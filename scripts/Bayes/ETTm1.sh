#!/bin/bash
model_name=TimeTucker
root_path_name=../dataset/
data_path_name=ETTm1.csv
data_name=ETTm1
seq_len=720
period_len=4
enc_in=7
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
