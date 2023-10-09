import math
from logger import logger
import os
import random
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_scheduler
import torch
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from classifier_eval import multiTasksEval, calculate_head_importance
import data_processors_ntasks as data
import pruning

def weighted_sample(population, weights, k):
    return np.random.choice(population, size=k, replace=True, p=weights)

def proportional_sample(dataset_lens:list, current_epoch:int, total_epoch:int):
    batch_dataset_ids = []
    for d_i, d_len in enumerate(dataset_lens):
        for i in range(d_len):
            batch_dataset_ids.append(d_i)
    random.shuffle(batch_dataset_ids)
    return batch_dataset_ids

def alpha_sample(dataset_lens:list, alpha=1, k=100):
    weights = [d_len**alpha for d_len in dataset_lens]
    sum_w = sum(weights)
    weights = [w/sum_w for w in weights]
    population = [i for i in range(len(dataset_lens))]
    return weighted_sample(population, weights, k=k).tolist()

def square_root_sample(dataset_lens:list, current_epoch:int, total_epoch:int):
    return alpha_sample(dataset_lens, alpha=0.5, k=sum(dataset_lens))

def annealed_sample(dataset_lens:list, current_epoch:int, total_epoch:int):
    alpha = 1-0.8*(current_epoch-1)/(total_epoch-1)
    return alpha_sample(dataset_lens, alpha=alpha, k=sum(dataset_lens))

def renew_best_result(curr_state:str, new_result:list=None, old_result:dict=None)->dict:
    # at the begining of training
    if curr_state is None:
        best_result = {
            "state":None,
            "results":[],
            "performance": -1
        }
        return best_result
    
    if old_result["state"] is None:

        best_result = {
            "state":curr_state,
            "results": new_result,
            "performance": sum([r["main"] for r in new_result])
        }
        return best_result

    assert(len(new_result) == len(old_result["results"]))
    
    curr_performance = sum([r["main"] for r in new_result])
    if curr_performance > old_result["performance"]:
        best_result = {
            "state":curr_state,
            "results": new_result,
            "performance": curr_performance
        }
        return best_result
    
    return old_result


def set_seeds(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def mask_grad(p, mask):
    device = p.device
    grad_tensor = p.grad.data.cpu().numpy()
    grad_tensor = np.where(mask<1, 0, grad_tensor)
    p.grad.data = torch.from_numpy(grad_tensor).to(device)
    return grad_tensor

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

def get_layer_idx(p_name:str):
    # ptm.encoder.layer.11.attention.self.query.weight
    str_list = p_name.split(".")
    return int(str_list[str_list.index("layer")+1])

def generate_attention_head_mask(layer_mask, hidden_size=768, num_attention_heads=12):
    attention_head_size = int(hidden_size / num_attention_heads)

    attention_head_weight_zeros = torch.zeros((attention_head_size, hidden_size),dtype=torch.int8)
    attention_head_bias_zeros = torch.zeros((attention_head_size), dtype=torch.int8)

    weight_mask = torch.ones((hidden_size,hidden_size), dtype=torch.int8)
    bias_mask = torch.ones((hidden_size), dtype=torch.int8)

    for head_idx in range(num_attention_heads):
        if layer_mask[head_idx] == 0:
            weight_mask[head_idx*attention_head_size:(head_idx+1)*attention_head_size] = attention_head_weight_zeros
            bias_mask[head_idx*attention_head_size:(head_idx+1)*attention_head_size] = attention_head_bias_zeros

    return weight_mask, bias_mask

def ridge_regression(X, y, alpha=0.0, solver='cholesky'):
    clf = Ridge(alpha=alpha,solver=solver)
    rdg_res = clf.fit(X, y)

    regress_score = rdg_res.score(X, y)

    w = np.array(rdg_res.coef_, dtype=float)
    b = np.array(rdg_res.intercept_, dtype=float)

    logger.info(f"score of regression:{regress_score}")
    return w, b, regress_score

def linear_evaluation(data_loader, model, task_id, disable_progress_bar=False):
    task_name = data.taskid2taskname[task_id]
    num_label = data.taskid2labelnum[task_id]

    logger.info(f"***** Running linear evaluation for task:{task_name} *****")
    logger.info(f"  Num examples = {len(data_loader.dataset)}")
    
    model.eval()

    prune_iterator = tqdm(
        data_loader,
        desc="Iteration",
        disable=disable_progress_bar,
        total=len(data_loader),
        miniters=int(len(data_loader)/100)
    )

    device = next(model.parameters()).device

    sent_reps = []
    targets = []
    dataset_id = None

    # get representation of sentences in dataset and their labels
    for step, batch in enumerate(prune_iterator):
        targets.append(batch["labels"])
        dataset_id = batch["dataset_labels"][0].item()

        batch.pop("dataset_labels")
        batch.pop("labels")

        with torch.no_grad():
            outputs = model.ptm(**batch)
        
        sent_reps.append(outputs[1].detach().cpu())
    
    # convert target to one-hot representations
    sent_reps = torch.cat(sent_reps, axis=0).cpu().numpy()
    targets = torch.cat(targets,axis=0).cpu().numpy()

    logger.info(f"sent_reps.shape = {sent_reps.shape}")
    logger.info(f"targets.shape = {targets.shape}")

    # targets = targets.reshape(-1)
    one_hots_tgt = np.zeros((targets.shape[0], num_label), dtype=float)
    one_hots_tgt[np.arange(targets.shape[0]), targets] = 1.0
    
    # solve the parameters of regression problems
    w,b,_ = ridge_regression(X=sent_reps, y=one_hots_tgt, alpha=0, solver='cholesky')
    
    model.classifiers[str(dataset_id)].weight.data = torch.tensor(w,dtype=torch.float32,device=device)
    model.classifiers[str(dataset_id)].bias.data = torch.tensor(b,dtype=torch.float32,device=device)


def multiTasksLinearEvaluation(data_loaders, model, task_ids, output_dir=None, disable_progress_bar=False):
    for task_i, task_id in enumerate(task_ids):
        linear_evaluation(
            data_loader=data_loaders[task_i],
            model=model,
            task_id=task_id,
            disable_progress_bar=disable_progress_bar
        )
    
    # save models for prune and eval
    sub_dir = f"epoch0"
    if output_dir is not None:
        output_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_dir,exist_ok=True)

        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), output_model_file)

def multiTasksTraining(
        train_dataloaders,
        num_training_samples,
        optimizer,
        model,
        accelerator,
        num_step_per_epoch,
        args,
        eval_dataloaders=None,
        task_ids=None,
    ):

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(num_step_per_epoch / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name= args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps= args.max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(num_step_per_epoch / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_training_samples}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Sampling method = {args.training_sample_method}")
    
    if args.do_specific_head_train:
        start_specific_head_train_step = int(args.specific_train_start_proportion * args.max_train_steps)
        parameters_frozened = False

        n_task = len(train_dataloaders)
        n_layer = model.ptm.config.num_hidden_layers
        n_head = model.ptm.config.num_attention_heads
        model_hidden_size = model.ptm.config.hidden_size
        n_to_prune = int(n_layer * n_head * args.train_specific_head_proportion)
        
        head_masks = torch.ones((n_task, n_layer, n_head), dtype=torch.int8)
        shared_head = torch.ones((n_layer, n_head), dtype=torch.int8)
        
        head_weight_masks = torch.ones((n_task, n_layer, model_hidden_size, model_hidden_size), dtype=torch.int8)
        head_bias_masks = torch.ones((n_task, n_layer, model_hidden_size), dtype=torch.int8)
        
        logger.info(f"  Start specific head training at step = {start_specific_head_train_step}(proportion={args.specific_train_start_proportion})")

    # baseline setting
    if args.froze_ptm:
        # only finetune the classifiers
        grad_parameters = ["classifiers"]
        for n,p in model.named_parameters():
            if not any(n_gp in n for n_gp in grad_parameters):
                p.requires_grad_(False)

        logger.info(f"  Froze the parameters of pre-train model, only finetune the linear classifier (args.froze_ptm={args.froze_ptm})")

    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // num_training_samples
            resume_step -= starting_epoch * num_training_samples

    dataset_lens = [len(train_dataloader) for train_dataloader in train_dataloaders]

    best_result = renew_best_result(curr_state=None)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        batch_dataset_ids = None
        if args.training_sample_method == "square_root":
            batch_dataset_ids = square_root_sample(dataset_lens, current_epoch=epoch+1, total_epoch=args.num_train_epochs)
        elif args.training_sample_method == "annealed":
            batch_dataset_ids = annealed_sample(dataset_lens, current_epoch=epoch+1, total_epoch=args.num_train_epochs)
        else:
            # defaule proportional
            batch_dataset_ids = proportional_sample(dataset_lens, current_epoch=epoch+1, total_epoch=args.num_train_epochs)

        logger.info(f"Epoch {epoch+1} begins.") 
        logger.info(f"Current batch_dataset_ids[:20]={batch_dataset_ids[:20]}")
        
        for step, number in enumerate((tqdm(batch_dataset_ids, desc="Iteration",miniters=int(len(batch_dataset_ids)/100)))):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            
            batch = train_dataloaders[number].__iter__().__next__()
            # batch = tuple(t.to(device) for t in batch)

            outputs = model(**batch)

            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            # logger.info(f"Training Loss:{loss.item():.4f}")

            if step % args.gradient_accumulation_steps == 0 or step == len(batch_dataset_ids) - 1:
                
                #set the gradient of other attention head to 0 except the specific important head for this task
                if args.do_specific_head_train and parameters_frozened:
                    # we only update parameters for each batch of one task
                    assert(args.gradient_accumulation_steps == 1)
                    
                    for n, p in model.named_parameters():
                        if model.mha_name in n:
                            layer_idx = get_layer_idx(p_name=n)
                            if "weight" in n:
                                mask_grad(p, mask=head_weight_masks[number][layer_idx])
                            elif "bias" in n:
                                mask_grad(p, mask=head_bias_masks[number][layer_idx])
                            else:
                                raise ValueError(f"Invalid name:{n}")
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and step % args.gradient_accumulation_steps == 0:
                    if args.save_model:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                            os.makedirs(output_dir,exist_ok=True)
                        # accelerator.save_state(output_dir)
                        
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model.state_dict(), output_model_file)
                
                    if args.eval_checkpointing_step:
                        curr_result = multiTasksEval(model=model, 
                                            eval_dataloaders=eval_dataloaders, 
                                            task_ids=task_ids, 
                                            accelerator=accelerator, 
                                            save_result=False)

                        best_result = renew_best_result(curr_state=f"step-{completed_steps}", new_result=curr_result, old_result=best_result)

            # calculate head importance and frozen the parameters except the specific head and classifier
            if args.do_specific_head_train and not parameters_frozened and completed_steps >= start_specific_head_train_step and (step % args.gradient_accumulation_steps == 0):
                
                if args.save_model:
                    # save model before specific training
                    output_dir = f"normal_multitask_train"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                        os.makedirs(output_dir,exist_ok=True)
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model.state_dict(), output_model_file)
                    
                logger.info(f"*** Start specific head training at step {completed_steps} ***")
                
                # Determine the important heads for each task firstly
                for idx,train_dataloader in enumerate(train_dataloaders):

                    head_importance = calculate_head_importance(
                        model=model,
                        dataloader=train_dataloader,
                        batch_size=args.per_device_train_batch_size,
                        accelerator=accelerator,
                        normalize_scores_by_layer=args.normalize_pruning_by_layer,
                        disable_progress_bar=args.no_progress_bars,
                        same_div=args.same_div
                    )

                    # Determine which heads to prune
                    head_mask, to_prune = pruning.what_to_prune(
                        head_importance,
                        n_to_prune,
                        head_mask=head_masks[idx],
                        to_prune={},
                        at_least_x_heads_per_layer=args.at_least_x_heads_per_layer,
                        ascending=args.prune_acending
                    )

                    head_masks[idx] = 1 - head_mask

                    shared_head = shared_head & head_masks[idx]

                    logger.info(f"Top {args.train_specific_head_proportion} important heads for task {idx} are:")
                    logger.info(pruning.to_pruning_descriptor(to_prune))

                logger.info(f"Number of important heads shared are {torch.sum(shared_head).item()}.")
                
                # get the specific head idx for each task and their mask
                for task_id in range(len(train_dataloaders)):

                    if not args.train_shared_important_head:
                        head_masks[task_id] -= shared_head
                    
                    logger.info(f"Head masks for task{task_id} are:")
                    print_matrix(matrix=head_masks[task_id],format_str="{:>2d}")

                    for layer_i in range(n_layer):
                        head_weight_masks[task_id][layer_i], head_bias_masks[task_id][layer_i] = generate_attention_head_mask(
                            layer_mask=head_masks[task_id][layer_i],
                            hidden_size=model_hidden_size,
                            num_attention_heads=n_head
                            )
                
                if not args.train_other_parameters:
                    # only finetune the classifiers and top important head for specific tasks
                    grad_parameters = ["classifiers", model.mha_name]
                    for n,p in model.named_parameters():
                        if not any(n_gp in n for n_gp in grad_parameters):
                            p.requires_grad_(False)

                parameters_frozened = True

            if completed_steps >= args.max_train_steps:
                break
        

        # Test model after each epoch
        if args.eval_epoch:
            # print(f"Training: eval_dataloaders:{eval_dataloaders}")
            curr_result = multiTasksEval(model=model, 
                                            eval_dataloaders=eval_dataloaders, 
                                            task_ids=task_ids, 
                                            accelerator=accelerator, 
                                            save_result=False)
            
            best_result = renew_best_result(curr_state=f"epoch-{epoch+1}", new_result=curr_result, old_result=best_result)


        # Save for every epoch if checkpointing_steps is None
        if checkpointing_steps is None and args.save_model:
            output_dir = f"epoch{epoch+1}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                os.makedirs(output_dir,exist_ok=True)
            # accelerator.save_state(output_dir)

            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model.state_dict(), output_model_file)
        
        
    logger.info("***** Finished training *****")

    logger.info(f'***** Best Eval Result at {best_result["state"]} *****')

    for task_idx, task_id in enumerate(task_ids):
        current_task = data.taskid2taskname[task_id]
        eval_result = best_result["results"][task_idx]
        logger.info(f"***** Eval results for task:{current_task} *****")
        for key in sorted(eval_result.keys()):
            logger.info("%s = %s" % (key, str(eval_result[key])))

    if args.with_tracking:
        logger.info(f"  Training loss = {total_loss}")


def multiTasksDistill(
        train_dataloaders,
        num_training_samples,
        optimizer,
        student_model,
        teacher_model,
        accelerator,
        num_step_per_epoch,
        args,
        eval_dataloaders=None,
        task_ids=None,
    ):

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(num_step_per_epoch / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name= args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps= args.max_train_steps,
    ) if args.pred_distill else None

    lr_scheduler = accelerator.prepare(lr_scheduler) if lr_scheduler is not None else None

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(num_step_per_epoch / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    print("num_step_per_epoch",num_step_per_epoch,"args.gradient_accumulation_steps",args.gradient_accumulation_steps)
    print("args.max_train_steps",args.max_train_steps)
    
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_training_samples}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    if args.do_specific_head_train:
        start_specific_head_train_step = int(args.specific_train_start_proportion * args.max_train_steps)
        parameters_frozened = False

        n_task = len(train_dataloaders)
        n_layer = student_model.ptm.config.num_hidden_layers
        n_head = student_model.ptm.config.num_attention_heads
        model_hidden_size = student_model.ptm.config.hidden_size
        n_to_prune = int(n_layer * n_head * args.train_specific_head_proportion)
        
        head_masks = torch.ones((n_task, n_layer, n_head), dtype=torch.int8)
        shared_head = torch.ones((n_layer, n_head), dtype=torch.int8)
        
        head_weight_masks = torch.ones((n_task, n_layer, model_hidden_size, model_hidden_size), dtype=torch.int8)
        head_bias_masks = torch.ones((n_task, n_layer, model_hidden_size), dtype=torch.int8)
        
        logger.info(f"  Start specific head training at step = {start_specific_head_train_step}(proportion={args.specific_train_start_proportion})")


    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // num_training_samples
            resume_step -= starting_epoch * num_training_samples

    dataset_lens = [len(train_dataloader) for train_dataloader in train_dataloaders]

    # logger.info(f"len(batch_dataset_ids) = {len(batch_dataset_ids)}")

    # used for distillation loss function
    loss_mse = torch.nn.MSELoss()

    for epoch in range(starting_epoch, args.num_train_epochs):
        student_model.train()
        if args.with_tracking:
            total_loss = 0
        
        tr_loss = 0.
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.

        batch_dataset_ids = None
        if args.training_sample_method == "square_root":
            batch_dataset_ids = square_root_sample(dataset_lens, current_epoch=epoch+1, total_epoch=args.num_train_epochs)
        elif args.training_sample_method == "annealed":
            batch_dataset_ids = annealed_sample(dataset_lens, current_epoch=epoch+1, total_epoch=args.num_train_epochs)
        else:
            # defaule proportional
            batch_dataset_ids = proportional_sample(dataset_lens, current_epoch=epoch+1, total_epoch=args.num_train_epochs)

        logger.info(f"Epoch {epoch+1} begins.") 
        logger.info(f"Current batch_dataset_ids[:20]={batch_dataset_ids[:20]}")
        
        
        for step, number in enumerate((tqdm(batch_dataset_ids, desc="Iteration",miniters=int(len(batch_dataset_ids)/100)))):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            
            batch = train_dataloaders[number].__iter__().__next__()
            # batch = tuple(t.to(device) for t in batch)
            
            student_outputs = student_model(**batch, is_student=True)
            student_logits, student_atts, student_reps = student_outputs.logits, student_outputs.attentions, student_outputs.hidden_states

            with torch.no_grad():
                teacher_outputs = teacher_model(**batch, is_student=False)
                teacher_logits, teacher_atts, teacher_reps = teacher_outputs.logits, teacher_outputs.attentions, teacher_outputs.hidden_states
            
            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            device = student_logits.device

            if not args.pred_distill:
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)

                assert teacher_layer_num % student_layer_num == 0
                
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                teacher_att)

                    tmp_loss = loss_mse(student_att, teacher_att)
                    att_loss += tmp_loss
                
                new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                new_student_reps = student_reps
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss

                loss = rep_loss + att_loss
                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()
            else:
                output_mode = "classification" if data.taskid2labelnum[task_ids[number]] > 1 else "regression"

                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                    teacher_logits / args.temperature)
                elif output_mode == "regression":
                    loss_mse = torch.nn.MSELoss()
                    cls_loss = loss_mse(student_logits.view(-1), batch["labels"].view(-1))
                
                loss = cls_loss
                tr_cls_loss += cls_loss.item()

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            tr_loss += loss.item()
            
            # logger.info(f"Training Loss:{loss.item():.4f}")

            if step % args.gradient_accumulation_steps == 0 or step == len(batch_dataset_ids) - 1:
                
                #set the gradient of other attention head to 0 except the specific important head for this task
                if args.do_specific_head_train and parameters_frozened:
                    # we only update parameters for each batch of one task
                    assert(args.gradient_accumulation_steps == 1)
                    
                    for n, p in student_model.named_parameters():
                        if student_model.mha_name in n:
                            layer_idx = get_layer_idx(p_name=n)
                            if "weight" in n:
                                mask_grad(p, mask=head_weight_masks[number][layer_idx])
                            elif "bias" in n:
                                mask_grad(p, mask=head_bias_masks[number][layer_idx])
                            else:
                                raise ValueError(f"Invalid name:{n}")
                
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and step % args.gradient_accumulation_steps == 0:
                    
                    if args.save_model:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                            os.makedirs(output_dir,exist_ok=True)
                        # accelerator.save_state(output_dir)
                        
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(student_model.state_dict(), output_model_file)

                    if args.eval_checkpointing_step:
                        multiTasksEval(model=student_model, 
                            eval_dataloaders=eval_dataloaders, 
                            task_ids=task_ids, 
                            accelerator=accelerator, 
                            save_result=False)

            # calculate head importance and frozen the parameters except the specific head and classifier
            if args.do_specific_head_train and not parameters_frozened and completed_steps >= start_specific_head_train_step and (step % args.gradient_accumulation_steps == 0):
                
                if args.save_model:
                    # save model before specific training
                    output_dir = f"normal_multitask_train"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                        os.makedirs(output_dir,exist_ok=True)
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(student_model.state_dict(), output_model_file)
                    
                logger.info(f"*** Start specific head training at step {completed_steps} ***")
                
                # Determine the important heads for each task firstly
                for idx,train_dataloader in enumerate(train_dataloaders):

                    head_importance = calculate_head_importance(
                        model=student_model,
                        dataloader=train_dataloader,
                        batch_size=args.per_device_train_batch_size,
                        accelerator=accelerator,
                        normalize_scores_by_layer=args.normalize_pruning_by_layer,
                        disable_progress_bar=args.no_progress_bars,
                        same_div=args.same_div
                    )

                    # Determine which heads to prune
                    head_mask, to_prune = pruning.what_to_prune(
                        head_importance,
                        n_to_prune,
                        head_mask=head_masks[idx],
                        to_prune={},
                        at_least_x_heads_per_layer=args.at_least_x_heads_per_layer,
                        ascending=args.prune_acending
                    )

                    head_masks[idx] = 1 - head_mask

                    shared_head = shared_head & head_masks[idx]

                    logger.info(f"Top {args.train_specific_head_proportion} important heads for task {idx} are:")
                    logger.info(pruning.to_pruning_descriptor(to_prune))

                logger.info(f"Number of important heads shared are {torch.sum(shared_head).item()}.")
                
                # get the specific head idx for each task and their mask
                for task_id in range(len(train_dataloaders)):

                    if not args.train_shared_important_head:
                        head_masks[task_id] -= shared_head
                    
                    logger.info(f"Head masks for task{task_id} are:")
                    print_matrix(matrix=head_masks[task_id],format_str="{:>2d}")

                    for layer_i in range(n_layer):
                        head_weight_masks[task_id][layer_i], head_bias_masks[task_id][layer_i] = generate_attention_head_mask(
                            layer_mask=head_masks[task_id][layer_i],
                            hidden_size=model_hidden_size,
                            num_attention_heads=n_head
                            )
                
                if not args.train_other_parameters:
                    # only finetune the classifiers and top important head for specific tasks
                    grad_parameters = ["classifiers", student_model.mha_name]
                    for n,p in student_model.named_parameters():
                        if not any(n_gp in n for n_gp in grad_parameters):
                            p.requires_grad_(False)

                parameters_frozened = True

            if completed_steps >= args.max_train_steps:
                break
        

        # Test model after each epoch
        if args.eval_epoch:
            # print(f"Training: eval_dataloaders:{eval_dataloaders}")
            multiTasksEval(model=student_model, 
                           eval_dataloaders=eval_dataloaders, 
                           task_ids=task_ids, 
                           accelerator=accelerator, 
                           save_result=False)

        # Save for every epoch if checkpointing_steps is None
        if checkpointing_steps is None and args.save_model:
            output_dir = f"epoch{epoch+1}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                os.makedirs(output_dir,exist_ok=True)
            # accelerator.save_state(output_dir)

            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(student_model.state_dict(), output_model_file)
        
        if completed_steps >= args.max_train_steps:
            # save the last model, reminder: we must set checkpointing_steps if setting the max_train_steps
            output_dir = f"step_{completed_steps}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                os.makedirs(output_dir,exist_ok=True)
            # accelerator.save_state(output_dir)
            
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(student_model.state_dict(), output_model_file)
        
    logger.info("***** Finished training *****")
    if args.with_tracking:
        logger.info(f"  Training loss = {total_loss}")
    