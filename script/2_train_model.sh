export CUDA_VISIBLE_DEVICES=1

train_epochs=3 
batch_size=32
goal=max_roi_30
test_year=2026
train_period=3

data_category=Top100
model_name=iTransformer
train_test_split="$((test_year - train_period))_$((test_year - 1))"
loss=ConcordanceCorrelation
data=Dataset_Individual_Seq_Norm

  CUDA_LAUNCH_BLOCKING=1 python -u model/run.py \
      --Notes tmp \
      --task_name pct_to_prev \
      --is_training 1 \
      --root_path data/ \
      --output_path results/ \
      --model_id max_price \
      --model $model_name \
      --data $data \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 13 \
      --dec_in 7 \
      --c_out 13 \
      --des 'Exp' \
      --itr 1 \
      --wandb_project_name ""\
      --individual \
      --batch_size ${batch_size} \
      --train_epochs $train_epochs \
      --use_norm 2 \
      --result_file_name ${data_category}_${model_name}_${train_test_split}_${loss}_${data} \
      --decomp_method "moving_avg" \
      --decomp_kernel_size 25 \
      --train_test_split ${train_test_split} \
      --category ${data_category} \
      --n_features 13 \
      --loss ${loss} \
      --goal $goal 