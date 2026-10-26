import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score, f1_score

def evaluate_COR(COR_labels: np.ndarray, true_labels: np.ndarray, true_outliers: np.ndarray = None) -> dict:
    """
    Computes the four metrics used in the paper to evaluate COR: NMI and R_n for clustering,
    Jaccard and F-measure for outlier detection.

    Inputs:
    -   COR_labels: the vector of labels assigned by COR
    -   true_labels: the vector of true classes
    -   true_outliers: a boolean vector of the true outliers (if not provided, only NMI and R_n are computed using observations with positive labels)
    
    Output:
        A dictionary with the values of all four metrics
    """
    
    if true_outliers is None:
        inliers_mask = COR_labels != -1
        return {"NMI": normalized_mutual_info_score(true_labels[inliers_mask], COR_labels[inliers_mask], average_method="geometric"),
                "R_n": adjusted_rand_score(true_labels[inliers_mask], COR_labels[inliers_mask]),
                "Jaccard": None, "F-measure": None}
    
    true_partition = np.where(true_outliers, -1, true_labels) 
    outliers_mask = COR_labels == -1
    return {"NMI": normalized_mutual_info_score(true_partition, COR_labels, average_method="geometric"),
            "R_n": adjusted_rand_score(true_partition, COR_labels),
            "Jaccard": jaccard_score(true_outliers, outliers_mask, zero_division=1.0),
            "F-measure": f1_score(true_outliers, outliers_mask, zero_division=0.0)}

def runs_statistics(run_statistics: dict, true_labels: np.ndarray =None, true_outliers:np.ndarray =None) -> pd.DataFrame:
    """
    Summarises the statistics of the COR runs: mean and standard deviation of the objective value,
    of the number of iterations and of the evaluation metrics.

    Inputs:
    -   run_statistics: the dictionary of statistics returned by COR (objective values, labels and iterations)
    -   true_labels: the vector of true classes; if not provided, the metrics are not computed
    -   true_outliers: boolean vector of the true outliers

    Output:
        A dataframe with two rows (mean and std dev) and a column for each quantity
    """
    runs_stats = pd.DataFrame({"objective value": run_statistics["obj_values"],
                               "iterations": run_statistics["iterations"]})

    if true_labels is not None: #true_outliers can be passed as None: evaluate_COR checks that input itself
        metrics = pd.DataFrame([evaluate_COR(labels, true_labels, true_outliers) for labels in run_statistics["labels"]])
        runs_stats = pd.concat([runs_stats, metrics], axis=1)

    return runs_stats.agg(["mean", "std"])