#!/usr/bin/env bash
gpu=0

main_dir="/home/cli/FunctionalSpecializationInMHA"
data_dir="${main_dir}/data"

cd ${main_dir}

# mnli, ag, ag2pairs, qnli, sst-2, ...
task_names="qnli:sst-2"
data_dirs="${data_dir}/QNLI_data:${data_dir}/SST-2_data"

# model_path="${data_dir}/bert-base-uncased"
# model_path="${data_dir}/bert-large-uncased"
model_path="${data_dir}/roberta-base"

# tasks="bert-base-uncased/${task_names}"
# tasks="bert-large-uncased/${task_names}"
tasks="roberta-based/${task_names}"

seed=0

# train_data_num=800000
train_data_num=10000
# train_data_num=120000:80000

max_train_steps=1000
checkpointing_steps=250
train_options="--max_train_steps ${max_train_steps} --checkpointing_steps ${checkpointing_steps} --eval_checkpointing_step --save_model "

# num_train_epochs=3
# train_options="--num_train_epochs ${num_train_epochs} --eval_epoch --save_model "

train_batch_size=32
gradient_accumulation_steps=1
# train_batch_size=64

# max_length=512
max_length=256

suffix=""
# suffix="-epoch1"
prefix="${tasks}/seed${seed}${suffix}"

# learning_rate=3e-5
learning_rate=2e-5
# learning_rate=5e-5


part=""

model_dir="log/$prefix"
log_file="log/${prefix}.log"

mkdir -p $model_dir

function run_multitasks_train () {
    CUDA_VISIBLE_DEVICES=$gpu python -u ./src/run_ntasks_classifier.py \
    --task_names $task_names \
    --do_train \
    --do_eval \
    --seed $seed \
    --do_lower_case \
    $1 \
    --data_dirs $data_dirs \
    --train_data_num $train_data_num \
    --model_name_or_path $model_path \
    --max_length $max_length \
    --per_device_train_batch_size $train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --pad_to_max_length \
    --per_device_eval_batch_size 8 \
    --output_dir $model_dir >>${log_file} 2>&1
}

function run_multitasks_dry_run () {
    CUDA_VISIBLE_DEVICES=$gpu python -u ./src/run_ntasks_classifier.py \
    --task_names $task_names \
    --dry_run \
    --seed $seed \
    --do_lower_case \
    --eval_pruned \
    --prune_percent $prune_percent \
    --data_dirs $data_dirs \
    --model_name_or_path $model_path \
    --max_length $max_length \
    --per_device_train_batch_size $train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 2 \
    --per_device_eval_batch_size 8 \
    --output_dir $model_dir >>${log_file} 2>&1
}


# run_multitasks_dry_run

if [ ! -e $model_dir/pytorch_model.bin ]
then
    echo "Begin to train model."
    run_multitasks_train "${train_options}"
fi
