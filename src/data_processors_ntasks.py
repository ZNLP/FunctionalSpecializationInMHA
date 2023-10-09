import csv
import os

from tqdm import tqdm, trange
import pandas as pd
from logger import logger

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import classifier_scoring as scoring
import six
import json
from datasets import load_dataset, load_metric

import random
from transformers import (
    DataCollatorWithPadding,
    default_data_collator,
)

_DATASET_SCRIPT="dataset_script.py"

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, dataset_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            dataset_label: see dataset_labels
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.dataset_label = dataset_label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, dataset_label=None):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, dataset_label=None):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_dummy_train_examples(self, data_dir, dataset_label=None):
        """Gets a collection of `InputExample`s for the train set."""
        return self.get_train_examples(data_dir, dataset_label)[:20]

    def get_dummy_dev_examples(self, data_dir, dataset_label=None):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_dev_examples(data_dir, dataset_label)[:10]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines

    @property
    def scorer(self):
        """Return a scoring class (accuracy, F1, etc...)"""
        return scoring.Accuracy()

num_labels_task = {
    "mrpc": 2,
    "mnli": 3,
    "cola": 2,
    "ag": 4,
    "imdb": 2,
    "yelp_p": 2,
    "yelp_f": 5,
    "dbpedia": 14,
    "yahoo": 10,
    "sst-2": 2,
    "qqp": 2,
    "qnli": 2,
    "sogou": 6,
    "cmnli": 3,
    "xnli-ar": 3,
    "xnli-bg": 3,
    "xnli-de": 3,
    "xnli-el": 3,
    "xnli-en": 3,
    "xnli-es": 3,
    "xnli-fr": 3,
    "xnli-hi": 3,
    "xnli-ru": 3,
    "xnli-sw": 3,
    "xnli-th": 3,
    "xnli-tr": 3,
    "xnli-ur": 3,
    "xnli-vi": 3,
    "xnli-zh": 3,
    "sts-b": 1,
    "wnli": 2,
    "rte": 2,
    "ag2pairs": 2,
    "scitail": 2,
    "snli": 3,
}

dataset_labels = {
    "mrpc": 1,
    "mnli": 2,
    "cola": 3,
    "ag": 4,
    "imdb": 5,
    "yelp_p": 6,
    "yelp_f": 7,
    "dbpedia": 8,
    "yahoo": 9,
    "sst-2": 10,
    "qqp": 11, 
    "qnli": 12,
    "sogou": 13,
    "cmnli": 14,
    "xnli-ar": 15,
    "xnli-bg": 16,
    "xnli-de": 17,
    "xnli-el": 18,
    "xnli-en": 19,
    "xnli-es": 20,
    "xnli-fr": 21,
    "xnli-hi": 22,
    "xnli-ru": 23,
    "xnli-sw": 24,
    "xnli-th": 25,
    "xnli-tr": 26,
    "xnli-ur": 27,
    "xnli-vi": 28,
    "xnli-zh": 29,
    "sts-b": 30,
    "wnli": 31,
    "rte": 32,
    "ag2pairs": 33,
    "scitail": 34,
    "snli": 35,
}

# used to define multitask BERT model
taskid2labelnum = {
    1: 2,
    2: 3,
    3: 2,
    4: 4,
    5: 2,
    6: 2,
    7: 5,
    8: 14,
    9: 10,
    10: 2,
    11: 2,
    12: 2,
    13: 6,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 3,
    22: 3,
    23: 3,
    24: 3,
    25: 3,
    26: 3,
    27: 3,
    28: 3,
    29: 3,
    30: 1,
    31: 2,
    32: 2,
    33: 2,
    34: 2,
    35: 3,
}

# used to define multitask BERT model
taskid2taskname = {
    1: "mrpc",
    2: "mnli",
    3: "cola",
    4: "ag",
    5: "imdb",
    6: "yelp_p",
    7: "yelp_f",
    8: "dbpedia",
    9: "yahoo",
    10: "sst-2",
    11: "qqp",
    12: "qnli",
    13: "sogou",
    14: "cmnli",
    15: "xnli-ar",
    16: "xnli-bg",
    17: "xnli-de",
    18: "xnli-el",
    19: "xnli-en",
    20: "xnli-es",
    21: "xnli-fr",
    22: "xnli-hi",
    23: "xnli-ru",
    24: "xnli-sw",
    25: "xnli-th",
    26: "xnli-tr",
    27: "xnli-ur",
    28: "xnli-vi",
    29: "xnli-zh",
    30: "sts-b",
    31: "wnli",
    32: "rte",
    33: "ag2pairs",
    34: "scitail",
    35: "snli",
}

class MultiProcessor(DataProcessor):
    """Processor for multi-tasks data set."""

    def __init__(self, tasks:str=None) -> None:
        super().__init__()
        if tasks is not None:
            # initialize the list of task id and labels
            self.task_ids = []
            # self.labels = []

            task_list = tasks.split(":")
            for i, task in enumerate(task_list):
                task = task.lower()
                if task not in dataset_labels:
                    return ValueError(f"Task not found: {task}")

                task_id = dataset_labels[task]
                self.task_ids.append(task_id)
                
                # self.labels.append(processors[task]().get_labels())

    def get_train_examples(self, tasks:str, data_dirs:str, data_num=None):
        """
        tasks: names of every task seperated by ":", e.g. "mnli:sst-2"
        data_dirs: diretorys of task seperated by ":", e.g. "../data/mnli_data:../data/sst-2_data"
        data_num: max num of data used for every task  
        """
        return self._fetch_examples(tasks,data_dirs,split="train",data_num=data_num)

    def get_dev_examples(self, tasks:str, data_dirs:str, data_num=None):
        """
        tasks: names of every task seperated by ":", e.g. "mnli:sst-2"
        data_dirs: diretorys of task seperated by ":", e.g. "../data/mnli_data:../data/sst-2_data"
        data_num: max num of data used for every task  
        """
        return self._fetch_examples(tasks,data_dirs,split="validation",data_num=data_num)

    def get_test_examples(self, tasks:str, data_dirs:str, data_num=None):
        """
        tasks: names of every task seperated by ":", e.g. "mnli:sst-2"
        data_dirs: diretorys of task seperated by ":", e.g. "../data/mnli_data:../data/sst-2_data"
        data_num: max num of data used for every task  
        """
        return self._fetch_examples(tasks,data_dirs,split="test",data_num=data_num)

    def get_all_examples(self, tasks:str, data_dirs:str, data_num=None):
        """
        tasks: names of every task seperated by ":", e.g. "mnli:sst-2"
        data_dirs: diretorys of task seperated by ":", e.g. "../data/mnli_data:../data/sst-2_data"
        data_num: max num of data used for every task  
        """
        return self._fetch_examples(tasks,data_dirs,split={"train":"train","validation":"validation"},data_num=data_num)

    def get_examples(self, tasks:str, data_dirs:str, split:str="train",data_num=None):
        """
        tasks: names of every task seperated by ":", e.g. "mnli:sst-2"
        data_dirs: diretorys of task seperated by ":", e.g. "../data/mnli_data:../data/sst-2_data"
        split: split of dataset, e.g., "train", "validation" or "test"
        data_num: max num of data used for every task  
        """
        return self._fetch_examples(tasks,data_dirs,split=split,data_num=data_num)

    def get_dummy_train_examples(self, tasks:str, data_dirs:str):
        """Gets a collection of `InputExample`s for the train set."""
        return self.get_train_examples(tasks, data_dirs, data_num="1000")

    def get_dummy_dev_examples(self, tasks:str, data_dirs:str):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_dev_examples(tasks, data_dirs, data_num="1000")


    def _fetch_examples(self, tasks:str, data_dirs:str, split, data_num=None):
        """fetch examples for the training and dev sets."""
        self.labels = []
        self.task_ids = []

        examples = []

        task_list = tasks.split(":")
        data_list = data_dirs.split(":")
        assert (len(task_list)==len(data_list))

        num_list = data_num.split(":") if data_num != None else None

        if num_list != None and len(num_list)==1:
            num_list = [num_list[0] for _ in range(len(task_list))]

        for i, task in enumerate(task_list):
            task = task.lower()
            if task not in dataset_labels:
                return ValueError(f"Task not found: {task}")

            task_id = dataset_labels[task]
            self.task_ids.append(task_id)

            script_path = os.path.join(data_list[i],_DATASET_SCRIPT)

            task_examples = load_dataset(script_path, split=split)
            
            if data_num != None:
                # assume restrict train data only
                if int(num_list[i]) < len(task_examples["train"]):
                    num_remain = int(num_list[i])
                    idx_remain = random.sample([s_idx for s_idx in range(len(task_examples["train"]))], k=num_remain)
                    task_examples["train"] = task_examples["train"].select(idx_remain)
                
                # if type(split)==str:
                #     task_examples[split] = task_examples[split][:int(num_list[i])]
                # else:
                #     for k,v in split.items():
                #         task_examples[k] = task_examples[k][:int(num_list[i])]

            examples.append(task_examples)
            # self.labels.append(processors[task]().get_labels())

            if type(split)==str:
                logger.info(f"Load {len(task_examples[split])} examples from {data_list[i]} for task {task}.")
            else:
                for k,v in split.items():
                    logger.info(f"Load {k} split {len(task_examples[k])} examples from {data_list[i]} for task {task}.")

        logger.info(f"Fetch examples for split:{split} done.")
        logger.info(f"Task list:{task_list}")
        logger.info(f"Task id list:{self.task_ids}")

        return examples

    def get_labels(self):
        """See base class."""
        if self.labels is None:
            return ValueError("Please define tasks first.")
        return self.labels


def label2id(label):
    label_map = {}
    for (i,label) in enumerate(label):
        label_map[label] = i
    return label_map

def convert2loader(raw_datasets, dataset_id:int, tokenizer, accelerator, args):
    if args.do_train or args.do_prune:
        label_list = raw_datasets["train"].features['label'].names if taskid2labelnum[dataset_id] > 1 else None
        column_names = raw_datasets["train"].column_names
    elif args.do_eval or args.do_predict:
        label_list = raw_datasets["validation"].features['label'].names if taskid2labelnum[dataset_id] > 1 else None
        column_names = raw_datasets["validation"].column_names
    else:
        raise ValueError("One of do_train, do_eval, do_predict or do_prune should be choosed!")

    # label_list.sort()  # Let's sort it for determinism

    non_label_column_names = [name for name in column_names if name != "label"]

    if len(non_label_column_names) >= 2:
        sentence1_key, sentence2_key = non_label_column_names[:2]
    else:
        sentence1_key, sentence2_key = non_label_column_names[0], None
        
    logger.info(f"Non_label_column_names: {non_label_column_names}")
    logger.info(f"sentence1_key: {sentence1_key}, sentence2_key: {sentence2_key}")

    
    # for classification tasks
    if taskid2labelnum[dataset_id] > 1:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        label2id = label_to_id
        id2label = {id: label for label, id in label2id.items()}
    else:
        label_to_id = None
        label2id = None
        id2label = None

    logger.info(f"label2id: {label2id}")
    logger.info(f"id2label: {id2label}")

    def preprocess_function(examples, **kwargs):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length" if args.pad_to_max_length else False, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None and examples["label"][0] == str:
                # Map labels to IDs (not necessary for GLUE tasks)
                # labels = [l for l in examples["label"]]
                # logger.info(f"example labels: {labels}")
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        result["dataset_labels"] = [kwargs["dataset_id"] for _ in result["labels"]]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False, #incase load same processed_datasets from different number of samples
            fn_kwargs={"dataset_id":dataset_id}
        )

    train_dataset = processed_datasets["train"] if args.do_train or args.do_prune else None
    eval_dataset = processed_datasets["validation"] if args.do_eval or args.do_predict else None

    if args.do_train or args.do_prune:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set id({dataset_id}): {train_dataset[index]}.")
    else:
        # Log a few random samples from the test set:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the test set id({dataset_id}): {eval_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    ) if args.do_train or args.do_prune else None

    eval_dataloader = DataLoader(
        eval_dataset, sampler=SequentialSampler(eval_dataset), collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    ) if args.do_eval or args.do_predict else None

    return train_dataloader, eval_dataloader, label_list, label2id, id2label
