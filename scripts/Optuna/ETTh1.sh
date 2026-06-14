#!/bin/bash

if [ ! -d "./logs/Optuna" ]; then
    mkdir -p ./logs/Optuna
fi

model_name=TimeTucker
data=ETTh1
data_path=ETTh1.csv
enc_in=7
seq_len=720
num_workers=6
trials_per_worker=10  

pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"; do
    
    for worker_id in $(seq 1 $num_workers); do
        
        
        export CUDA_VISIBLE_DEVICES=0 
        
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
          --period_len 24 \
          --use_hyperParam_optim \
          --n_jobs 1 \
          --optuna_trial_num $trials_per_worker \
          --train_epochs 30 \
          --patience 5 \
          > logs/Optuna/${data}_${pred_len}_worker${worker_id}.log 2>&1 &
          
        echo "Launched Worker $worker_id on GPU 0 (Logs: logs/Optuna/${data}_${pred_len}_worker${worker_id}.log)"
        
        sleep 2
    done

    wait
    
    echo ">>> Completed hyper-parameter search for pred_len=$pred_len !"
    echo ""
done

echo " All Optuna distributed searches finished successfully on A6000!"