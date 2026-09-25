


run_script=ft.py
ds_config=ds_offload_stage2_config.json


model=Qwen/Qwen2.5-Math-7B-Instruct

lang=sample6langs
dataset_name=prm_shepherd


deepspeed --num_nodes 1 --num_gpus 4 \
    $run_script \
    --ds_config $ds_config \
    --model_name_or_path $model \
    --lang $lang \
    --dataset $dataset_name \
    --per_device_train_batch_size 7 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5

