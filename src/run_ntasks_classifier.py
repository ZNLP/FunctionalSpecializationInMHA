# coding=utf-8

from logger import logger
import os
import json

import datasets
import torch
import numpy as np

import classifier_args

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoTokenizer
)

from transformers.utils.versions import require_version
from model.modeling_bert import BertForMultiTasksClassification, replace_bert_attn
replace_bert_attn()

from model.modeling_roberta import RobertaForMultiTasksClassification, replace_roberta_attn
replace_roberta_attn()

import data_processors_ntasks as data
import pruning
from classifier_training import multiTasksTraining, set_seeds, multiTasksLinearEvaluation
from classifier_eval import calculate_head_importance, task_eval, multiTasksEval, get_random_mask, eval_metrics_dict, multiTasksPredict
from classifier_scoring import Metrics, d_score

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

type2model = {
    "bert": BertForMultiTasksClassification,
    "roberta" : RobertaForMultiTasksClassification,
}

def prepare_dry_run(args):
    # args.no_cuda = True
    args.per_device_train_batch_size = 24
    args.per_device_eval_batch_size = 8
    args.do_train = True
    args.num_train_epochs = 2
    args.do_eval = True
    args.do_prune = True
    args.do_anal = False
    # args.output_dir = tempfile.mkdtemp()
    return args

def print_matrix(matrix:np.ndarray,format_str:str="{:.7f}"):
    """print matrix for save and compare result easily

    Args:
        matrix (np.array): matrix to print
    """
    raw, col = matrix.shape
    for r in range(raw):
        for c in range(col):
            print(format_str.format(matrix[r][c]),end="\t")
        # end of this line
        print()

def main():
    # Arguments
    parser = classifier_args.get_ntasks_parser()
    classifier_args.training_args(parser)
    classifier_args.fp16_args(parser)
    classifier_args.pruning_args(parser)
    classifier_args.eval_args(parser)
    classifier_args.analysis_args(parser)
    classifier_args.head_specific_training_args(parser)
    args = parser.parse_args()

    # Sanity checks
    if args.task_names is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    # ==== CHECK ARGS AND SET DEFAULTS ====
    if args.dry_run:
        args = prepare_dry_run(args)

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: "
            f"{args.gradient_accumulation_steps}, should be >= 1"
        )

    if args.n_retrain_steps_after_pruning > 0 and args.retrain_pruned_heads:
        raise ValueError(
            "--n_retrain_steps_after_pruning and --retrain_pruned_heads are "
            "mutually exclusive"
        )


    # ==== SETUP DEVICE ====
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )

    logger.info("Initial Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        logger.info(f"\t{attr}={value}")

    # Make one log on every process with the configuration for debugging.
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # ==== PREPARE DATA ====
    # Get the datasets from tasks defined

    data_processor = data.MultiProcessor()
    data_split = json.loads(args.data_split) if args.data_split is not None else {"train":"train","validation":"validation"}
    raw_dataset_list = data_processor.get_examples(tasks=args.task_names, data_dirs=args.data_dirs, split=data_split, data_num=args.train_data_num)    

    # ==== SETUP EXPERIMENT ====
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    config.taskid2labelnum = data.taskid2labelnum

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    model = type2model[config.model_type].from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        state_dict= torch.load(args.init_model) if args.init_model is not None else None
    )

    # Preprocessing the datasets
    num_training_samples = 0
    num_step_per_epoch = 0

    train_dataloaders = []
    eval_dataloaders = []
    
    label_list = []
    label2id_list = []
    id2label_list = []

    for idx, raw_dataset in enumerate(raw_dataset_list):
        # print(f"{idx}:{data_processor.task_ids[idx]}")
        # print(f"{idx}")
        train_dataloader, eval_dataloader, labels, label2id, id2label  = data.convert2loader(
                                                                                raw_datasets=raw_dataset,
                                                                                dataset_id=data_processor.task_ids[idx],
                                                                                tokenizer=tokenizer,
                                                                                accelerator=accelerator,
                                                                                args=args
                                                                                )
        train_dataloaders.append(train_dataloader)
        eval_dataloaders.append(eval_dataloader)
        label_list.append(labels)
        label2id_list.append(label2id)
        id2label_list.append(id2label)

        if args.do_train or args.do_prune:
            num_training_samples += len(raw_dataset["train"])
            num_step_per_epoch += len(train_dataloader)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    
    # reuse for eval and prune
    # metric = load_metric("accuracy")


    if args.do_train or args.do_prune:
        for idx, train_dataloader in enumerate(train_dataloaders):
            train_dataloaders[idx] = accelerator.prepare(train_dataloader)

    if args.do_eval or args.do_predict:
        for idx, eval_dataloader in enumerate(eval_dataloaders):
            eval_dataloaders[idx] = accelerator.prepare(eval_dataloader)


    if args.do_train:
        if args.linear_eval:
            multiTasksLinearEvaluation(
                data_loaders=train_dataloaders,
                model=model,
                task_ids=data_processor.task_ids,
                output_dir=args.output_dir
            )

            multiTasksEval(model=model,
                       eval_dataloaders=eval_dataloaders,
                       task_ids=data_processor.task_ids,
                       accelerator=accelerator,
                       save_result=True,
                       output_dir=args.output_dir)
        else:
            multiTasksTraining(train_dataloaders=train_dataloaders,
                                num_training_samples=num_training_samples,
                                optimizer=optimizer,
                                model=model,
                                accelerator=accelerator,
                                num_step_per_epoch=num_step_per_epoch,
                                args=args,
                                eval_dataloaders=eval_dataloaders,
                                task_ids=data_processor.task_ids)
    

    if args.do_eval:
        base_results = multiTasksEval(model=model,
                       eval_dataloaders=eval_dataloaders,
                       task_ids=data_processor.task_ids,
                       accelerator=accelerator,
                       save_result=True,
                       output_dir=args.output_dir)
    
    if args.do_predict:
        multiTasksPredict(
                    model=model,
                    eval_dataloaders=eval_dataloaders,
                    id2label_dicts=id2label_list,
                    task_ids=data_processor.task_ids,
                    accelerator=accelerator,
                    output_dir=args.output_dir
                    )

    if args.do_prune:
        if int(args.random_prune_number) > 0: 
            random_prune_matrix = np.zeros((len(args.prune_percent),args.random_prune_number,len(data_processor.task_ids)))
            
            logger.info("***** Random Prune *****")
            for prune_idx, prune_percent in enumerate(args.prune_percent):
                for random_idx in range(args.random_prune_number):
                    random_head_mask = get_random_mask(
                        num_layer=model.ptm.config.num_hidden_layers,
                        num_head=model.ptm.config.num_attention_heads,
                        percentage=int(prune_percent)/100.0)
                    
                    logger.info(f"Evaluating following random prune mask (Percentage:{prune_percent}%, Time-{random_idx})")
                    print_matrix(random_head_mask.numpy(),format_str="{:.1f}")
                    
                    eval_results = multiTasksEval(model=model,
                        eval_dataloaders=eval_dataloaders,
                        task_ids=data_processor.task_ids,
                        accelerator=accelerator,
                        save_result=False,
                        head_mask=random_head_mask)
                    
                    for task_idx, eval_result in enumerate(eval_results):
                        # default accuracy
                        eval_main_result = eval_result["main"]
                        random_prune_matrix[prune_idx][random_idx][task_idx] = eval_main_result
            logger.info("***** Random Prune done! *****")

        # Parse pruning descriptor
        to_prune = pruning.parse_head_pruning_descriptors(
            args.attention_mask_heads,
            reverse_descriptors=args.reverse_head_mask,
        )
        # Determine the number of heads to prune, e.g. prune_sequence = [7,7,7,7,8,...]
        prune_sequence = pruning.determine_pruning_sequence(
            args.prune_number,
            args.prune_percent,
            model.ptm.config.num_hidden_layers,
            model.ptm.config.num_attention_heads,
            args.at_least_x_heads_per_layer,
        )

        args.prune_tasks = args.prune_tasks.lower()

        # prune matrix for save, shape: (#prune percentage, #tasks, #tasks)
        prune_matrix = np.zeros((len(args.prune_percent),len(data_processor.task_ids),len(data_processor.task_ids)))

        for idx, task_id in enumerate(data_processor.task_ids):
            task_name = data.taskid2taskname[task_id]
            
            # if not prune this task
            if args.prune_tasks != "all" and task_name not in args.prune_tasks:
                logger.info(f"Skip the prune test for task {task_name}.")
                continue
            
            logger.info(f"***** Prune task {task_name} *****")
            to_prune = {}
            
            # matrix to indicate the head mask and calculate the model's output 
            head_mask = torch.ones((model.ptm.config.num_hidden_layers,model.ptm.config.num_attention_heads),dtype=torch.int8)

            for step, n_to_prune in enumerate(prune_sequence):

                if step == 0 or args.exact_pruning:
                    # Calculate importance scores for each layer
                    head_importance = calculate_head_importance(
                        model=model,
                        dataloader=eval_dataloaders[idx] if args.importance_on_eval else train_dataloaders[idx],
                        batch_size=args.per_device_train_batch_size,
                        accelerator=accelerator,
                        normalize_scores_by_layer=args.normalize_pruning_by_layer,
                        disable_progress_bar=args.no_progress_bars,
                        same_div=args.same_div,
                        head_mask=head_mask
                    )

                    logger.info("Head importance scores")
                    for layer in range(len(head_importance)):
                        layer_scores = head_importance[layer].cpu().data
                        logger.info("\t".join(f"{x:.6f}" for x in layer_scores))
                
                # Determine which heads to prune
                head_mask, to_prune = pruning.what_to_prune(
                    head_importance,
                    n_to_prune,
                    head_mask=head_mask,
                    to_prune={} if args.retrain_pruned_heads else to_prune,
                    at_least_x_heads_per_layer=args.at_least_x_heads_per_layer,
                    ascending=args.prune_acending
                )

                if args.eval_pruned:
                    # Print the pruning descriptor
                    logger.info("Evaluating following pruning strategy")
                    logger.info(pruning.to_pruning_descriptor(to_prune))

                    for eval_id, eval_task_id in enumerate(data_processor.task_ids):
                        current_task = data.taskid2taskname[eval_task_id]
                        if current_task in eval_metrics_dict:
                            eval_metric = Metrics(eval_metrics_dict[current_task])
                        else:
                            eval_metric = Metrics(eval_metrics_dict["default"])
                        
                        logger.info(f"***** Running evaluation for task:{current_task} *****")
                        logger.info(f"  Num examples = {len(eval_dataloaders[eval_id].dataset)}")

                        eval_result = task_eval(
                            model=model,
                            data_loader=eval_dataloaders[eval_id],
                            metric=eval_metric,
                            accelerator=accelerator,
                            is_regression=(data.taskid2labelnum[eval_task_id]==1),
                            head_mask=head_mask
                        )

                        main_eval_result = eval_result["main"]
                    
                        logger.info(f"***** Pruning eval results for task:{current_task} *****")
                        tot_pruned = sum(len(heads) for heads in to_prune.values())
                        logger.info(f"{tot_pruned}\t{main_eval_result}\t({eval_metric.main_eval_metric})")

                        prune_matrix[step][idx][eval_id] = main_eval_result

        for prune_percentage_id in range(len(args.prune_percent)):
            print(f"\n\nPrune matrix for {args.prune_percent[prune_percentage_id]}")
            print_matrix(prune_matrix[prune_percentage_id])
            d_s, avg_d = d_score(prune_matrix[prune_percentage_id], base_results)
            for p_t_i, d_t in enumerate(d_s):
                print(f"D-score of task {data.taskid2taskname[data_processor.task_ids[p_t_i]]}: {d_t:.2f}")
            print(f"Average D-score: {avg_d:.2f} (Prune top {args.prune_percent[prune_percentage_id]}% important attention heads)")

            if int(args.random_prune_number) > 0: 
                # print(f"Random prune performance:")
                # print_matrix(random_prune_matrix[prune_percentage_id])
                print(f"Random prune average performance:")
                print_matrix(np.average(random_prune_matrix[prune_percentage_id],axis=0).reshape(1,len(data_processor.task_ids)))


if __name__ == "__main__":
    main()
