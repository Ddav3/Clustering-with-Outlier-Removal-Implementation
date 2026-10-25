import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def plot_correlation(df: pd.DataFrame):    
    """
    Creates a Heatmap using the correlation between an input dataframe's columns.
    Input:
        The dataframe 
    """     
    mask = np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool)) 

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), mask=mask, annot=True,cmap="RdBu_r", vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.title("Correlation")
    plt.tight_layout()
    plt.show()

def make_categoric_partitions(df:pd.DataFrame, ignore: list = None) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """
    Separates the categorical columns of a dataframe and turns them into partitions.
    
    Inputs:
    -   df: the dataframe
    -   ignore: the list with the names of the columns we wish to ignore; these will be removed 
        and won't be turned into partitions
    
    Outputs:
    -   the dataframe with only the remaining numerical columns
    -   the list of additional partitions
    """

    ignore = ignore or []
    categ_cols =[col for col in df.select_dtypes(exclude="number").columns if col not in ignore]
    partitions = [pd.factorize(df[col])[0] for col in categ_cols] 
    return df.drop(columns=categ_cols+ignore), partitions
    
def normalise(df: pd.DataFrame)-> pd.DataFrame:  #unused
    """
    Normalises each column of a given dataframe.
    """
    for col in df.columns:
        if df[col].dtype.kind not in "iuf":
            print("Dataset is not fully numeric. Force quitted.")
            return

    min = df.min(axis = 0)
    max = df.max(axis = 0)
    range = (max - min).replace(0,1)
    return (df - min)/range

def standardise(df: pd.DataFrame)-> pd.DataFrame:
    """
    Standardises each column of a given dataframe.
    """
    for col in df.columns:
        if df[col].dtype.kind not in "iuf":
            print("Dataset is not fully numeric. Force quitted.")
            return

    mean = df.mean(axis=0)
    std = df.std(axis=0).replace(0, 1)
    return (df - mean) / std

def PCA_scatter_plot(df: pd.DataFrame, labels: np.ndarray):
    p = PCA(2).fit_transform(df)
    inliers_mask = labels>=0
    plt.figure(figsize=(10,10))
    plt.scatter(p[inliers_mask,0], p[inliers_mask,1], c = labels[inliers_mask], cmap = "tab10", s = 33, alpha = 0.6)
    plt.scatter(p[labels ==-1, 0], p[labels ==-1, 1], marker = "x", c = "black", s = 20, label = "outliers found by COR")
    # plt.scatter(p[true_outliers, 0], p[true_outliers, 1], facecolors="none", edgecolors="red",
    #               s=70, label="real outliers")
    plt.legend()
    plt.show()