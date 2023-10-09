# Interpreting and Exploiting Functional Specialization in Multi-Head Attention under Multi-task Learning

The Code for the EMNLP 2023 main conference paper "**Interpreting and Exploiting Functional Specialization in Multi-Head Attention under Multi-task Learning**".

Inspired by functional specialization in the human brain, which helps to efficiently handle multiple tasks, we propose an interpreting method to quantify the degree of functional specialization in multi-head attention (IAP), and a simple multi-task training method to increase functional specialization in multi-head attention (IAT). 

## How to run?

### Set up an virtual environment
```
conda create -n mha python=3.9
conda activate mha
pip install -r requirements.txt
```
### Important Attention-head Pruning (IAP)
1. (optional) Finetune a multitask learning model
```
cd script
bash multitasks_train.sh
```

2. Quantify the degree of functional specialization 
```
cd script
# Set up your private parameters, e.g., the path of multitask model and tasks
vim important_attention_prune.sh
bash important_attention_prune.sh
```

### Important Attention-head Training (IAT)
```
cd script
# Set up your private parameters, e.g., the path of source model and tasks
vim important_attention_train.sh
bash important_attention_train.sh
```
## AG-Pair Dataset
The AG-Pair dataset is built from the original dataset AG's News that contains 120k training samples from four topics.
Given a pair of news as input, the model has to predict whether they are belonging to the same topic (Same) or not (Different).

To generate this dataset, samples in AG are iterated in random order and have an equal chance to combine a sample in the same topic or the other three topics.
Thus the numbers of training samples in two classes are both 60k.
Moreover, each news in AG's News occurs exactly twice in the AG-Pair dataset to keep the same word frequency.

## Contact
lichong2021@ia.ac.cn

## How to cite our paper?
```
@inproceedings{li-etal-2023-FunctionalSpecialization,
  author    = {Chong Li and
               Shaonan Wang and
               Yunhao Zhang and
               Jiajun Zhang and
               Chengqing Zong},
  title = "Interpreting and Exploiting Functional Specialization in Multi-Head Attention under Multi-task Learning",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  year = "2023",
  address = "Singapore",
  publisher = "Association for Computational Linguistics",
}
```