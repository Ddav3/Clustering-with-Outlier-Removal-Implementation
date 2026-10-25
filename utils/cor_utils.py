import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
def generate_basic_partitions(X:pd.DataFrame, r:int, k_min:int = 2, k_max: int= None,seed:int = None)-> list[np.ndarray]:
    """
    Runs K-means on a data matrix to generate r basic partitions with variable number of clusters each. 
    Inputs:
    -   X: the data matrix;
    -   r: the number of basic partitions; 
    -   k_min: the minimum number of clusters in each partition. 2 by default;
    -   k_max: the maximum number of clusters that may compone each partition. No default value is provided;
    Outputs:
        The r partitions, in the form of a list of length r
    """
    if k_max is None:
        k_max = k_min**2
    elif k_max <k_min:
        print("Autofix - basic_partitioning: k_min greater than k_max; set to square.")
        k_max = k_min**2

    X = np.asarray(X,dtype = float)    
    basic_partitions = []
    np.random.seed(seed)
    for i in range(r):
        Kmeans_model = KMeans(n_clusters=np.random.randint(k_min,k_max+1), random_state=i, n_init=10)
        partition = Kmeans_model.fit_predict(X)
        basic_partitions.append(partition)
    return basic_partitions

def build_binary_matrices(partitions: list[np.ndarray])-> tuple[np.ndarray, np.ndarray]:
    """
    Constructs the binary matrices B and its complement, mapping observations from the original matrix and assigning 1 
    to the cell [l,ij] if, w.r.t. partition i, observation l belongs to cluster j.
    Input:
        The list of partitions.
    Output:
        Both matrix B and its complement.
    """
    cols = []
    for partition in partitions:
        for label in np.unique(partition):
            cols.append((partition ==label).astype(float))
    B = np.array(cols).T
    return B, 1-B

def generate_centroids(BB: list[np.ndarray], K: int)-> list[np.ndarray]:
    """
    Produces K centroids for the binary matrix BB, randomly variating the dimensions.
    Inputs:
    -   BB: the concatenated binary matrix
    -   K: the number of centroids (and clusters)
    Output:
        A list of K centroids randomly taken from BB with slightly varied values. 
    """
    uniques = np.unique(BB, axis = 0)
    if(K > BB.shape[0]):
        raise ValueError("Numerical error: too much more clusters than observations. Cannot initialize centroids")
    elif (K > len(uniques)):
        raise ValueError("Numerical error: not enough unique rows for K clusters. Cannot initialize centroids")

    return uniques[np.random.choice(len(uniques), K, replace =False)].copy()

def compute_kl_distances(BB: np.ndarray, centroids: np.ndarray, return_matrix:bool= False) -> tuple[np.ndarray,np.ndarray]| tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Computes the KL distances between all points of the matrix and the centroids. 
    Note that the we are working on the original formula is: D_KL(P(x)||Q(x)) = sum_P(x) log (P(x)/Q(x))
    and since b can be either 0 or 1, we have that:
    -   the limit for x->0 of xlog(x/m) is 0,
    -   1log(1/m) = - log(m),
    we directly compute the result as: - BB * log (m).
    Inputs:
    -   BB: the concatenated binary matrix 
    -   centroids: the vector of K centroids 
    -   return_matrix: whether the whole matrix should be returned or not
    Outputs: 
    -   the n x K matrix of distances, if 'return_matrix' is set to True
    -   the label of the cluster of the closest centroid
    -   the distance from the closest centroid
    """
    m = np.clip(centroids, 1e-12, None)
    distances = -(BB @ np.log2(m).T)
    if return_matrix:
        return distances, distances.argmin(axis=1), distances.min(axis = 1)
    return distances.argmin(axis=1), distances.min(axis = 1)

def extract_outliers(distances_vector: np.ndarray, how_many_o:int) ->np.ndarray:
    """
    Looks for the top 'how_many_o' values having the highest distance in the distance vector and provides
    their indexes.
    Inputs:
    -   the distance vector
    -   how_many_o: the number of outliers we are looking for
    Output:
        The vector of indexes of the outlier candidates.
    """
    if (how_many_o<= 0):
        return np.array([], dtype=int)
    return np.argsort(distances_vector)[-how_many_o:]

def update_centroids(BB:np.ndarray, labels: np.ndarray, old_centroids: np.ndarray) -> np.ndarray:
    """
    Updates the coordinates of the given centroids by arithmetic mean of the clusters components. The cluster belonging is defined 
    through a vector of labels.
    Inputs:
    -   BB: the concatenated binary matrix 
    -   labels: the vector of labels indicating the cluster of each data point
    -   old_centroids: the old vector of centroids
    Output:
        The new vector of centroids
    """
    centroids = old_centroids.copy()
    for k in range(len(old_centroids)):
        cluster = BB[labels == k]
        if(len(cluster) > 0):
            centroids[k] = cluster.mean(axis = 0)
    return centroids

def compute_holoentropy(BB: np.ndarray,labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Provides the result of the objective function on the K-means-- configuration dictated by the labels.
    The value is computed as the sum of the entropies for each cluster.
    Inputs:
    -   BB: the concatenated binary matrix
    -   labels: the vector of labels
    Output: 
        The objective value 
    """
    result = 0
    for k in range(len(centroids)):
        C_k = BB[labels == k] 
        if len(C_k) == 0:
            continue
        p = C_k.mean(axis= 0) 
        p = p[p > 0]   
        result += - len(C_k)* (p * np.log2(p)).sum() 
    return result