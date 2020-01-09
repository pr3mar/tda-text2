import numpy as np
import os

all_words = False

all_vs_stop_word_path = 'all_words' if all_words else 'stopwords_excluded'
DATA_PATH = f'../cache/{all_vs_stop_word_path}/'

r = 5

for file_name in os.listdir(DATA_PATH + 'full_dimension/'):
    if file_name.endswith('.npy'):
        matrix = np.load(DATA_PATH + 'full_dimension/' + file_name)
        u, s, vh = np.linalg.svd(matrix)
        reduced = vh[:r, :]  # there is a difference in notation between instruction PDF and Numpy - the matrix V is not
        # transposed in numpy -> see first sentence of https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
        np.save(DATA_PATH + 'reduced_dimension/' + file_name, reduced)
