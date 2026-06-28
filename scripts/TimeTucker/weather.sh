model_name=TimeTucker
root_path_name=../dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=Weather
seq_len=720
gpu=${GPU:-0}

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/${model_id_name}" ]; then
    mkdir ./logs/${model_id_name}
fi
dir=./logs/${model_id_name}


for pred_len in 96 192 336 720; do
python -u run_longExp.py \
    --is_training 1 \
    --orthogonal_weight 0.12 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$model_name" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --period_len 4 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 3 \
    --r_n 8 \
    --r_c 20 \
    --r_p 4 \
    --use_revin 1 \
    --share_factors 1 \
    --gpu $gpu \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --dropout 0.4 \
    > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"
done
