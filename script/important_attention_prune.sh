#!/usr/bin/env bash
gpu=0

main_dir="/home/cli/FunctionalSpecializationInMHA"
data_dir="${main_dir}/data"

cd ${main_dir}

# base_model_path="${data_dir}/bert-base-uncased"
# base_model_path="${data_dir}/bert-large-uncased"
base_model_path="${data_dir}/roberta-base"

task_names="qnli:sst-2"
data_dirs="${data_dir}/QNLI_data:${data_dir}/SST-2_data"

# tasks="bert-large-uncased/${task_names}"
# tasks="bert-base-uncased/${task_names}"
tasks="roberta-based/${task_names}"

seed=0

prune_percent="20 30 40"
# prune_percent=`seq 5 5 100`
# prune_percent="5 10 15 20 25 30 35 40 45 50"

prune_tasks="all"
# prune_tasks="qnli"

claculate_importance_num=10000
# claculate_importance_num=5000000

random_prune_number=0
# random_prune_number=3

exact_pruning=False
# exact_pruning=True

acending_pruning=False
# acending_pruning=True

# same_div=False
same_div=True

# max_length=512
max_length=256

claculate_importance_batch_size=32
eval_batch_size=64

# suffix=""
# suffixs=(epoch3)
suffixs=(step_500)
# suffixs=(step_500 step_1000)


part=""

function run_multitasks_eval () {
    CUDA_VISIBLE_DEVICES=$gpu python -u ./src/run_ntasks_classifier.py \
    --task_names $task_names \
    --do_eval \
    --seed $seed \
    --do_lower_case \
    $1 \
    --data_dirs $data_dirs \
    --train_data_num $claculate_importance_num \
    --model_name_or_path $base_model_path \
    --max_length $max_length \
    --per_device_train_batch_size $claculate_importance_batch_size \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size $eval_batch_size \
    --output_dir $model_dir >>${log_file} 2>&1
}


for suffix in "${suffixs[@]}"
do

prune_other_options=""
prune_prefix=""

if [ "${same_div}" == "True" ];
then
prune_other_options="${prune_other_options} --same_div"
prune_prefix="${prune_prefix}same_div_"
fi


if [ "${acending_pruning}" == "True" ];
then
prune_other_options="${prune_other_options} --prune_acending"
prune_prefix="${prune_prefix}acending_"
fi

if [ "${exact_pruning}" == "True" ];
then
prune_other_options="${prune_other_options} --exact_pruning"
prune_prefix="${prune_prefix}exact_"
fi

prefix="${tasks}/seed${seed}/${prune_prefix}prune-${suffix}"

model_dir="log/$prefix"
log_file="log/${prefix}.log"
mkdir -p $model_dir

model_path="log/${tasks}/seed${seed}/${suffix}/pytorch_model.bin"

echo "${tasks}-seed${seed}-${suffix}: Begin to prune model."

prune_options="--do_prune --eval_pruned --prune_tasks ${prune_tasks} --prune_percent ${prune_percent} --random_prune_number ${random_prune_number} --init_model ${model_path} $prune_other_options"
run_multitasks_eval "$prune_options"

cat ${log_file} | grep "Average D-score:"
done