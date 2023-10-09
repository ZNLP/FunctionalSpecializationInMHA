from tqdm import tqdm
from itertools import islice
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr
from classifier_scoring import Metrics

from logger import logger
import data_processors_ntasks as data
import os

eval_metrics_dict = {
    "default": {"main":"accuracy","metrics":["accuracy"]},
    "mrpc": {"main":"f1","metrics":["accuracy","f1"]},
    "qqp": {"main":"f1","metrics":["accuracy","f1"]},
    "sts-b": {"main":"spearman_r","metrics":["pearson_r","spearman_r"]}, 
    # "sts-b": {"main":"mean_metrics","metrics":["pearson_r","spearman_r"]}, 
    "cola": {"main":"matthews_r","metrics":["matthews_r"]},
}


def get_head_idx_from_str(head_str:str):
    str_part = head_str.split(",")
    layer = int(str_part[0].split("-")[1])
    index = int(str_part[1].split("-")[1])
    return layer, index

def get_random_mask(num_layer=12, num_head=12, percentage=0.3):
    head_mask = torch.ones((num_layer,num_head))
    num_head_masked = int(percentage * num_layer * num_head)
    head_idx = []
    for i in range(num_layer):
        for j in range(num_head):
            head_idx.append(f"L-{i},H-{j}")
    random.shuffle(head_idx)
    mask_done_num = 0
    for head_str in head_idx:
        layer_idx, num_idx = get_head_idx_from_str(head_str)
        layer_left = torch.sum(head_mask[layer_idx]).item()
        if layer_left <= 1:
            # we have to left a head
            continue
        mask_done_num += 1
        head_mask[layer_idx][num_idx] = 0
        if mask_done_num >= num_head_masked:
            break
    return head_mask


def calculate_head_importance(
        model,
        dataloader,
        batch_size,
        accelerator,
        device=None,
        normalize_scores_by_layer=True,
        disable_progress_bar=False,
        same_div=False,
        head_mask=None
):
    """Calculate head importance scores"""
    # Disable dropout
    model.eval()
    # Device
    device = device or next(model.parameters()).device

    n_prune_steps = len(dataloader)
    num_sample = len(dataloader.dataset)
    
    logger.info("***** Calculating head importance *****")
    logger.info(f"  Num examples = {num_sample}")
    logger.info(f"  Batch size = {batch_size}")
    logger.info(f"  Num steps = {n_prune_steps}")

    prune_iterator = tqdm(
        dataloader,
        desc="Iteration",
        disable=disable_progress_bar,
        total=n_prune_steps,
        miniters=int(n_prune_steps/100)
    )

    # Head importance tensor
    head_mask = head_mask.to(device) if head_mask is not None else None
    
    n_layers = model.ptm.config.num_hidden_layers
    n_heads = model.ptm.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    tot_tokens = 0

    for step, batch in enumerate(prune_iterator):
        # logger.info(f"input of batch: {batch}")
        batch["head_mask"] = head_mask

        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        for layer in range(model.ptm.config.num_hidden_layers):
            self_att = model.ptm.encoder.layer[layer].attention.self
            # ctx.shape:(bsz,n_heads,n_length,hidden_size)
            ctx = self_att.context_layer_val
            # grad_ctx.shape:(bsz,n_heads,n_length,hidden_size)
            grad_ctx = ctx.grad

            ctx = ctx.view(-1,n_heads,ctx.size(-2),ctx.size(-1))
            grad_ctx = grad_ctx.view(-1,n_heads,grad_ctx.size(-2),grad_ctx.size(-1))

            # logger.info(f"ctx.shape: {ctx.shape}")
            # logger.info(f"grad_ctx.shape: {grad_ctx.shape}")

            # Take the dot, which equals to for b,h,l,i : dot[b,h,l] = torch.dot(grad_ctx[b,h,l,:],ctx[b,h,l,:]) 
            dot = torch.einsum("bhli,bhli->bhl", [grad_ctx, ctx])
            head_importance[layer] += dot.abs().sum(-1).sum(0).detach()

        tot_tokens += batch["attention_mask"].float().detach().sum().data
    
    logger.info(f"++++Total tokens:{tot_tokens}")
    
    if same_div:
        head_importance /= tot_tokens
    else:
        # (Original) why the importance of the last layer is divided by subset_size(0~1) but the others divided by tot_tokens (may be hundards of and thounds of)
        head_importance[:-1] /= tot_tokens

    # Layerwise importance normalization
    if normalize_scores_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    return head_importance

def multiTasksEval(
    model,
    eval_dataloaders,
    task_ids,
    accelerator,
    save_result=False,
    output_dir=None,
    head_mask=None
):
    eval_results = []

    for idx, eval_dataloader in enumerate(eval_dataloaders):
        # print("data_loader:",eval_dataloader)

        current_task = data.taskid2taskname[task_ids[idx]]
        logger.info(f"***** Running evaluation for task:{current_task} *****")
        logger.info(f"  Num examples = {len(eval_dataloader.dataset)}")
        
        if current_task in eval_metrics_dict:
            metric = Metrics(eval_metrics_dict[current_task])
        else:
            metric = Metrics(eval_metrics_dict["default"])

        eval_result = task_eval(
            model=model,
            data_loader=eval_dataloader,
            metric=metric,
            accelerator=accelerator,
            is_regression=(data.taskid2labelnum[task_ids[idx]]==1),
            head_mask=head_mask
        )

        eval_results.append(eval_result)

        logger.info(f"***** Eval results for task:{current_task} *****")
        for key in sorted(eval_result.keys()):
            logger.info("%s = %s" % (key, str(eval_result[key])))

        # save eval result
        if save_result:
            output_eval_file = os.path.join(output_dir, f"eval_{current_task}.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(eval_result.keys()):
                    writer.write("%s = %s\n" % (key, str(eval_result[key])))
            logger.info(f"***** Save eval results for task:{current_task} in {output_eval_file} *****")

    return eval_results

def task_eval(
    model,
    data_loader,
    metric,
    accelerator,
    is_regression,
    disable_progress_bar=False,
    head_mask = None
):
    samples_seen = 0
    model.eval()

    eval_iterator = tqdm(
        data_loader,
        desc="Iteration",
        disable=disable_progress_bar,
        total=len(data_loader),
        miniters=int(len(data_loader)/100)
    )

    device = next(model.parameters()).device
    head_mask = head_mask.to(device) if head_mask is not None else None

    for step, batch in enumerate(eval_iterator):
        batch["head_mask"] = head_mask
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()

        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        
        # logger.info("logits:",outputs.logits)
        # logger.info(f"Test: predictions:{predictions},references:{references}")

        # print(f"accelerator.num_processes:{accelerator.num_processes}")

        if accelerator.num_processes > 1:
            if step == len(data_loader) - 1:
                predictions = predictions[: len(data_loader.dataset) - samples_seen]
                references = references[: len(data_loader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        
        metric.add_batch(
                predictions=predictions.detach().cpu().numpy(),
                references=references.detach().cpu().numpy(),
            )

        # print(f"predictions:{predictions.cpu()},references:{references.cpu()}")

    eval_metric = metric.compute()
    
    return eval_metric
 
def multiTasksPredict(
    model,
    eval_dataloaders,
    id2label_dicts,
    task_ids,
    accelerator,
    output_dir=None,
    head_mask=None
):

    for idx, eval_dataloader in enumerate(eval_dataloaders):
        # print("data_loader:",eval_dataloader)

        current_task = data.taskid2taskname[task_ids[idx]]

        logger.info(f"***** Running prediction for task:{current_task} *****")
        logger.info(f"  Num examples = {len(eval_dataloader.dataset)}")

        is_regression = (data.taskid2labelnum[task_ids[idx]]==1)
        if not is_regression:
            logger.info(f"  Id2labels = {id2label_dicts[idx]}")

        pred_result = task_predict(
            model=model,
            data_loader=eval_dataloader,
            accelerator=accelerator,
            is_regression=is_regression,
            head_mask=head_mask
        )

        # save eval result
        output_eval_file = os.path.join(output_dir, f"{current_task.upper()}.tsv")
        with open(output_eval_file, "w") as writer:
            writer.write("index\tprediction\n")
            for p_idx, p_i in enumerate(pred_result):
                if is_regression:
                    # for STS-B (0~5)
                    p_i = min(max(p_i,0),5)
                    writer.write(f"{p_idx}\t{p_i:.3f}\n")
                else:
                    writer.write(f"{p_idx}\t{id2label_dicts[idx][p_i]}\n")

        logger.info(f"***** Save prediction results for task:{current_task} in {output_eval_file} *****")


def task_predict(
    data_loader,
    model,
    is_regression,
    accelerator,
    disable_progress_bar=False,
    head_mask = None
):
    """Predict labels on a dataset"""

    model.eval()

    # Device
    device = next(model.parameters()).device
    head_mask = head_mask.to(device) if head_mask is not None else None

    predict_iterator = tqdm(
        data_loader,
        desc="Predicting labels",
        disable=disable_progress_bar,
        total=len(data_loader),
        miniters=int(len(data_loader)/100)
    )

    # Compute model predictions
    samples_seen = 0
    prediction_all = []
    for step, batch in enumerate(predict_iterator):
        batch["head_mask"] = head_mask
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        predictions = accelerator.gather(predictions)

        if accelerator.num_processes > 1:
            if step == len(data_loader) - 1:
                predictions = predictions[: len(data_loader.dataset) - samples_seen]
                references = references[: len(data_loader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]

        # Track predictions
        batch_predictions = predictions.detach().cpu().numpy()
        for pred in batch_predictions:
            prediction_all.append(pred)

    return prediction_all


def analyze_nli(anal_examples, predictions, labels_list):
    report = {
        "label": {},
        "lex_sem": {},
        "pred_arg_struct": {},
        "logic": {},
        "knowledge": {},
        "domain": {},
    }
    normalizers = {k: {} for k in report}
    for example, pred in zip(anal_examples, predictions):
        correct = float(example.label == labels_list[pred])
        for feature in report:
            values = getattr(example, feature)
            if values is not None:
                # Sometimes there are multiple values
                for value in values.split(";"):
                    # Record whether the model was correct on this particular
                    # value of the feature
                    if value not in report[feature]:
                        report[feature][value] = 0
                        normalizers[feature][value] = 0
                    report[feature][value] += correct
                    normalizers[feature][value] += 1
    # Normalize report
    for feature in report:
        Ns = normalizers[feature]
        report[feature] = {k: v / Ns[k] for k, v in report[feature].items()}

    return report
