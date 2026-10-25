import numpy as np
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