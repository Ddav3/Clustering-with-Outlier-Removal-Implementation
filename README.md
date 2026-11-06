# Clustering with Outlier Removal (COR) - Implementation

This project is the implementation of the **COR** algorithm proposed in the paper:

> H. Liu, J. Li, Y. Wu, Y. Fu, *Clustering With Outlier Removal*, IEEE Transactions on Knowledge and Data Engineering, 33(6), 2021.

COR performs clustering and outlier detection at the same time. 
After several basic partitions are generated with K-means, they are mapped into a concatenated binary matrix on which K-means-- is run iteratively. The centroids are assigned using the KL-divergence as distance function to determine the closest centroid for each observation. Finally, Holoentropy is adopted as objective function to be minimized.

## Requirements

Python 3.10 or later. It is recommended to install the libraries in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Run the code

Open `cor_test.ipynb` and run the cells in order. The first cells define COR and the global settings;
the following sections run the test on different datasets.

Each test cell runs a single configuration of $K$ and $o$: the comments in the cells indicate which parameters to change
to obtain the other results shown in the report. Note that the sensitivity analysis on *Forest Fires* runs COR on a
15 x 15 grid of values, so it may take a while.

## Datasets

| Dataset | Source |
|---|---|
| Statlog (Shuttle) | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/148/statlog+shuttle) |
| Wine Quality | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| Credit Card Fraud Detection | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Forest Fires | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/162/forest+fires) |