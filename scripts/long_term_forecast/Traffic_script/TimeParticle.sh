export CUDA_VISIBLE_DEVICES=0

model_name=TimeParticle

batch_size=16

d_model=128

learning_rate=0.001


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --n_layer 1 \
  --top_k 1 \
  --period_list  24 \
  --d_state 16 \
  --expand 2 \
  --d_conv 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model $d_model \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --n_layer 1 \
  --top_k 1 \
  --period_list 24  \
  --d_state 16 \
  --expand 2 \
  --d_conv 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model $d_model \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --n_layer 1 \
  --top_k 1 \
  --period_list 24  \
  --d_state 16 \
  --expand 2 \
  --d_conv 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model $d_model \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --n_layer 1 \
  --top_k 1 \
  --period_list 24  \
  --d_state 16 \
  --expand 2 \
  --d_conv 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model $d_model \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1