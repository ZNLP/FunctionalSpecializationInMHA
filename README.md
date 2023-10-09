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