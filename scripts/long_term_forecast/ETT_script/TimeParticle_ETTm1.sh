export CUDA_VISIBLE_DEVICES=0

model_name=TimeParticle

d_model=64
expand=2
learning_rate=0.01
batch_size=128
patience=10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --n_layer 1 \
  --top_k 1 \
  --period_list 48 \
  --d_state 16 \
  --expand $expand \
  --d_conv 4 \
  --d_model $d_model \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --n_layer 1 \
  --top_k 1 \
  --period_list 48 \
  --d_state 16 \
  --expand $expand \
  --d_conv 4 \
  --d_model $d_model \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --n_layer 1 \
  --top_k 1 \
  --period_list 48 \
  --d_state 16 \
  --expand $expand \
  --d_conv 4 \
  --d_model $d_model \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --n_layer 1 \
  --top_k 1 \
  --period_list 48 \
  --d_state 16 \
  --expand $expand \
  --d_conv 4 \
  --d_model $d_model \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1