#!/bin/bash
# 这个脚本安全地启动 4 个并行工作流（Workers）共同推进 Optuna 进度

if [ ! -d "./logs/Optuna" ]; then
    mkdir -p ./logs/Optuna
fi

model_name=TimeTucker
trial_num=50   # 50次探索，如果有4个Worker，每个Worker大概负责分摊 12-13 次
data=traffic
enc_in=862

# 要搜索的预测长度
pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"; do
    echo ">>> Starting Distributed Optuna for $data | Pred: $pred_len"
    
    # 拉起 4 个独立的并行 Worker（假设你要跑 4 并发）
    for worker_id in {1..4}; do
        # 这里的技巧：如果有多张卡，可以通过分配 CUDA_VISIBLE_DEVICES 来实现多卡并行
        # 例如：gpu_id=$(( (worker_id - 1) % 2 )) 使用卡 0 和卡 1 交替
        export CUDA_VISIBLE_DEVICES=0 
        
        # 将 n_jobs 严格设为 1，依赖 SQLite 数据库协调进度
        python -u run_longExp.py \
          --is_training 1 \
          --model $model_name \
          --data custom \
          --data_path ${data}.csv \
          --model_id ${data}_Optuna_${pred_len}_worker${worker_id} \
          --pred_len $pred_len \
          --seq_len 720 \
          --enc_in $enc_in \
          --period_len 24 \
          --use_hyperParam_optim \
          --n_jobs 1 \
          --optuna_trial_num $trial_num \
          --train_epochs 30 \
          --patience 5 \
          --batch_size 64 \
          > logs/Optuna/${data}_${pred_len}_worker${worker_id}.log 2>&1 &
    done
    
    # 核心指令：挂起主脚本，等待这 4 个 worker 把当前的 pred_len 调参全部跑完，再进入下一个 pred_len
    wait
    echo ">>> Completed pred_len=$pred_len"
done
