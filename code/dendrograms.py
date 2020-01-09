import pickle
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

if __name__ == '__main__':
    for fname in ['all_words', 'stopwords_excluded']:
        for dim in range(3):
            froot = f"../cache/{fname}/distance_matrices/dimension{dim}/"
            D = np.load(f"{froot}distance_matrix.npy")
            indices = pickle.load(open(f"{froot}filename_to_distance_matrix_row_and_column_index.P", 'rb'))
            indices = [k for k in indices.keys()]
            np.fill_diagonal(D, 0)
            L = linkage(squareform(D), method='average')
            plt.figure(figsize=(20, 10))
            dendrogram(L, orientation='left', labels=indices)
            plt.savefig(f"{froot}dendrogram.png")
