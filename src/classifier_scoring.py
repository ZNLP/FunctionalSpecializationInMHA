import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr

def accuracy(preds, labels, normalize=True, sample_weight=None):
    return float(accuracy_score(labels, preds, normalize=normalize, sample_weight=sample_weight))

def f1(preds, labels):
    f1 = f1_score(y_true=labels, y_pred=preds)
    return f1

def pearson_r(preds, labels):
    return pearsonr(preds, labels)[0]

def spearman_r(preds, labels):
    return spearmanr(preds, labels)[0]

def matthews_r(preds, labels):
    return matthews_corrcoef(labels, preds)

def d_score(prune_matrix:np.ndarray, base_results:list):
    assert prune_matrix.shape[0] == len(base_results)
    num_task = len(base_results)
    d_s = []
    for t_i in range(num_task):
        base_perf = base_results[t_i]["main"]
        self_prune = prune_matrix[t_i][t_i]
        other_prune = (np.sum(prune_matrix[t_i, :]) - self_prune) / (num_task-1)
        d_t = (other_prune - self_prune) / base_perf * 100.0
        d_s.append(d_t)    
    return d_s, sum(d_s)/float(num_task)

metrics_dict_implement = {
    "accuracy": accuracy,
    "f1": f1,
    "pearson_r": pearson_r,
    "spearman_r": spearman_r,
    "matthews_r": matthews_r
}

class Metrics(object):
    predictions = None
    references = None
    metric_dict = None
    main_eval_metric = None

    def __init__(self, metric_dict) -> None:
        self.metric_dict = metric_dict       
        self.predictions = []
        self.references = []

        self.main_eval_metric = metric_dict["main"]
        if self.main_eval_metric == "mean_metrics":
            self.main_eval_metric = "mean(" + "-".join([metric for metric in self.metric_dict["metrics"]])+")"

    def add_batch(self, predictions, references):
        if type(predictions) == list and type(references) == list:
            self.predictions.extend(predictions)
            self.references.extend(references)
        elif type(predictions) == np.ndarray and type(references) == np.ndarray:
            self.predictions.extend(predictions.tolist())
            self.references.extend(references.tolist())
        else :
            raise ValueError(f"Unsupport prediction type:{type(predictions)} and reference type:{type(references)}!")
    
    def compute(self):
        result = {}
        values = []
        metrics = self.metric_dict["metrics"]

        for metric in metrics:
            result[metric] = metrics_dict_implement[metric](preds=self.predictions, labels=self.references)
            values.append(result[metric])
        
        if self.metric_dict["main"] in metrics:
            result["main"] = result[self.metric_dict["main"]]
        elif self.metric_dict["main"] == "mean_metrics":
            result["main"] = np.mean(values)
        else:
            main_metric = self.metric_dict["main"]
            raise ValueError(f"metric of main :{main_metric} is not implemented")

        self.predictions = []
        self.references = []
        return result