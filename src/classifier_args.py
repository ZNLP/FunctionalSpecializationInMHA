
import argparse
from transformers import (
    SchedulerType
)

def get_ntasks_parser():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on multi-text classification task")

    # Required parameters
    parser.add_argument(
        "--task_names",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train."
    )

    parser.add_argument(
        "--data_dirs",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other "
        "data files) for the task."
    )

    parser.add_argument(
        "--data_split",
        default='{\"train\":\"train\",\"validation\":\"validation\"}',
        type=str,
        help="Define data split, default: '{\"train\":\"train\",\"validation\":\"validation\"}'"
    )

    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )

    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--train_data_num",
        default=None,
        type=str,
        help="Number of training samples used to train."
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--init_model", default=None, type=str,
        help="Path of model parameters load to eval or prune."
    )

    parser.add_argument(
        "--prune_tasks",
        default="all",
        type=str,
        help="The name of the task to eval pruned."
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model "
        "predictions and checkpoints will be written."
    )

    # Other parameters
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )

    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="Run all steps with a small model and sample data."
    )

    parser.add_argument(
        "--do_train",
        action='store_true',
        help="Whether to run training."
    )

    parser.add_argument(
        "--do_prune",
        action='store_true',
        help="Whether to run pruning."
    )

    parser.add_argument(
        "--prune_acending",
        action='store_true',
        help="Whether to prune in acending order of head importance(Default: descending)."
    )

    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run eval on the dev set."
    )

    parser.add_argument(
        "--do_predict",
        action='store_true',
        help="Whether to run predict on the dev/test set."
    )

    parser.add_argument(
        "--eval_epoch",
        action='store_true',
        help="Whether to run eval after training each epoch."
    )

    parser.add_argument(
        "--eval_checkpointing_step",
        action='store_true',
        help="Whether to run eval after training each checkpointing_steps.(checkpointing_steps must be set)"
    )

    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Whether not to use CUDA when available"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="random seed for initialization"
    )

    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Print data examples"
    )

    parser.add_argument(
        "--no-progress-bars",
        action='store_true',
        help="Disable progress bars"
    )

    parser.add_argument(
        "--feature_mode",
        action='store_true',
        help="Don't update the BERT weights."
    )

    parser.add_argument(
        "--toy_classifier",
        action='store_true',
        help="Toy classifier"
    )

    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Don't raise an error when the output dir exists"
    )

    parser.add_argument(
        "--toy_classifier_n_heads",
        default=1,
        type=int,
        help="Number of heads in the simple (non-BERT) sequence classifier"
    )
    
    # arguments for model hub
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_model_id", 
        type=str, 
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    
    return parser


def training_args(parser):
    train_group = parser.add_argument_group("Training")

    train_group.add_argument(
        "--training_sample_method",
        default="proportional",
        help="The sampling method to use.",
        choices=["proportional", "square_root", "annealed"],
    )

    train_group.add_argument(
        "--linear_eval",
        action='store_true',
        help="Whether to linearly evaluate pre-train model."
    )

    train_group.add_argument(
        "--save_model",
        action='store_true',
        help="Whether to save the parameters of model."
    )

    train_group.add_argument(
        "--froze_ptm",
        action='store_true',
        help="Whether to froze the parameters of pre-train models."
    )

    train_group.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="Initial learning rate (after the potential warmup period) to use."
    )

    train_group.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    train_group.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    train_group.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )

    train_group.add_argument(
        "--attn_dropout",
        default=0.1,
        type=float,
        help="Head dropout rate"
    )
    
    train_group.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )

    train_group.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    train_group.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear "
        "learning rate warmup for. "
        "E.g., 0.1 = 10%% of training."
    )
    train_group.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    train_group.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before "
        "performing a backward/update pass."
    )
    train_group.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    train_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

def head_specific_training_args(parser):
    specific_training_group = parser.add_argument_group("Specific training for attention heads.")

    specific_training_group.add_argument(
        "--do_specific_head_train",
        action='store_true',
        help="Whether to run specific training attention head for each task."
    )

    specific_training_group.add_argument(
        "--specific_train_start_proportion",
        default=0.9,
        type=float,
        help="Proportion of start point when perform specific training "
        "E.g., 0.1 = 10%% of training."
    )

    specific_training_group.add_argument(
        "--train_specific_head_proportion",
        default=0.3,
        type=float,
        help="Proportion of top important attention head trained for particular task when perform specific training."
        "E.g., 0.3 = 30%% of heads."
    )

    specific_training_group.add_argument(
        "--train_shared_important_head",
        action='store_true',
        help="Whether to train those important heads shared by tasks"
    )

    specific_training_group.add_argument(
        "--train_other_parameters",
        action='store_true',
        help="Whether to train other parameters other than attention head."
    )

def distill_args(parser):
    distill_group = parser.add_argument_group("Distilling")

    distill_group.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    
    distill_group.add_argument("--teacher_init",
                        default=None,
                        type=str,
                        help="The initial model of teacher.")

    distill_group.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")

    distill_group.add_argument('--aug_train',
                        action='store_true')
    
    distill_group.add_argument('--pred_distill',
                        action='store_true',
                        help="Distill the prediction layer.")
    
    distill_group.add_argument('--temperature',
                        type=float,
                        default=1.)

    


def pruning_args(parser):
    prune_group = parser.add_argument_group("Pruning")
    prune_group.add_argument(
        "--compute_head_importance_on_subset",
        default=1.0,
        type=float,
        help="Percentage of the training data to use for estimating "
        "head importance."
    )
    prune_group.add_argument(
        "--prune_percent",
        default=[50],
        type=float,
        nargs="*",
        help="Percentage of heads to prune."
    )
    prune_group.add_argument(
        "--prune_number",
        default=None,
        nargs="*",
        type=int,
        help="Number of heads to prune. Overrides `--prune_percent`"
    )
    prune_group.add_argument(
        "--random_prune_number",
        default=3,
        type=int,
        help="Number of random prune."
    )
    prune_group.add_argument(
        "--prune_reverse_order",
        action='store_true',
        help="Prune in reverse order of importance",
    )
    prune_group.add_argument(
        "--normalize_pruning_by_layer",
        action='store_true',
        help="Normalize importance score by layers for pruning"
    )
    prune_group.add_argument(
        "--actually_prune",
        action='store_true',
        help="Really prune (like, for real)"
    )
    prune_group.add_argument(
        "--at_least_x_heads_per_layer",
        type=int,
        default=0,
        help="Keep at least x attention heads per layer"
    )
    prune_group.add_argument(
        "--exact_pruning",
        action='store_true',
        help="Reevaluate head importance score before each pruning step."
    )
    prune_group.add_argument(
        "--same_div",
        action='store_true',
        help="The head importance scores in every layer are divided by same token number."
    )
    prune_group.add_argument(
        "--eval_pruned",
        action='store_true',
        help="Evaluate the network after pruning"
    )
    prune_group.add_argument(
        "--importance_on_eval",
        action='store_true',
        help="Evaluate the importance on eval dataset"
    )
    prune_group.add_argument(
        "--n_retrain_steps_after_pruning",
        type=int,
        default=0,
        help="Retrain the network after pruning for a fixed number of steps"
    )
    prune_group.add_argument(
        "--retrain_learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for retraining the network after pruning for a "
        "fixed number of steps"
    )
    prune_group.add_argument(
        "--retrain_pruned_heads",
        action='store_true',
        help="Retrain the pruned heads"
    )
    prune_group.add_argument(
        "--n_retrain_steps_pruned_heads",
        type=int,
        default=0,
        help="Retrain the pruned heads"
    )
    prune_group.add_argument(
        "--reinit_from_pretrained",
        action='store_true',
        help="Reinitialize the pruned head from the pretrained model"
    )
    prune_group.add_argument(
        "--no_dropout_in_retraining",
        action='store_true',
        help="Disable dropout when retraining heads"
    )
    prune_group.add_argument(
        "--only_retrain_val_out",
        action='store_true',
        help="Only retrain the value and output layers for attention heads"
    )


def eval_args(parser):
    eval_group = parser.add_argument_group("Evaluation")
    
    eval_group.add_argument(
        "--attention_mask_heads", default="", type=str, nargs="*",
        help="[layer]:[head1],[head2]..."
    )
    eval_group.add_argument(
        '--reverse_head_mask',
        action='store_true',
        help="Mask all heads except those specified by "
        "`--attention-mask-heads`"
    )
    eval_group.add_argument(
        '--save-attention-probs', default="", type=str,
        help="Save attention to file"
    )


def analysis_args(parser):
    anal_group = parser.add_argument_group("Analyzis")
    anal_group.add_argument(
        "--anal_data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the "
        "`diagnistic-full.tsv` file."
    )


def fp16_args(parser):
    fp16_group = parser.add_argument_group("FP16")
    fp16_group.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of"
        " 32-bit"
    )
    fp16_group.add_argument(
        '--loss_scale',
        type=float, default=0,
        help="Loss scaling to improve fp16 numeric stability. "
        "Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n"
    )
