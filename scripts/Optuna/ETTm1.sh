#!/bin/bash

if [ ! -d "./logs/Optuna" ]; then
    mkdir -p ./logs/Optuna
fi

model_name=TimeTucker
data=ETTm1
data_path=ETTm1.csv
enc_in=7
seq_len=720
period_len=4
gpu=${GPU:-0}
num_workers=6
trials_per_worker=10

pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"; do
    export IS_OPTUNA_MASTER=0

    for worker_id in $(seq 1 $num_workers); do
        export CUDA_VISIBLE_DEVICES=$gpu

        python -u run_longExp.py \
          --is_training 1 \
          --model $model_name \
          --root_path ../dataset/ \
          --data_path $data_path \
          --data $data \
          --model_id ${data}_Optuna_${pred_len}_worker${worker_id} \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in $enc_in \
          --period_len $period_len \
          --use_period_norm 1 \
          --use_revin 1 \
          --share_factors 1 \
          --use_orthogonal 1 \
          --use_hyperParam_optim \
          --n_jobs 1 \
          --optuna_trial_num $trials_per_worker \
          --train_epochs 30 \
          --patience 5 \
          --gpu 0 \
          > logs/Optuna/${data}_${pred_len}_worker${worker_id}.log 2>&1 &

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched Worker $worker_id on GPU $gpu (PID: $!, Logs: logs/Optuna/${data}_${pred_len}_worker${worker_id}.log)"

        sleep 2
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for $num_workers workers to complete for pred_len=$pred_len..."
    wait
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker search complete for pred_len=$pred_len."

    export IS_OPTUNA_MASTER=1
    export CUDA_VISIBLE_DEVICES=$gpu

    python -u run_longExp.py \
      --is_training 1 \
      --model $model_name \
      --root_path ../dataset/ \
      --data_path $data_path \
      --data $data \
      --model_id ${data}_Optuna_${pred_len}_master \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in $enc_in \
      --period_len $period_len \
      --use_period_norm 1 \
      --use_revin 1 \
      --share_factors 1 \
      --use_orthogonal 1 \
      --use_hyperParam_optim \
      --n_jobs 1 \
      --optuna_trial_num 0 \
      --train_epochs 30 \
      --patience 5 \
      --gpu 0 \
      > logs/Optuna/${data}_${pred_len}_master.log 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final evaluation complete for pred_len=$pred_len."
    echo ""
done

echo "All Optuna distributed searches finished successfully!"
