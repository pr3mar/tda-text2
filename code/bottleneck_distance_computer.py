import pickle
import gudhi
from pathlib import Path
import numpy as np

all_words = False

all_vs_stop_word_path = 'all_words' if all_words else 'stopwords_excluded'
DATA_PATH = f'../cache/{all_vs_stop_word_path}/reduced_dimension/'


def get_persistence_of_dimension(full_persistence, dimension):
    return [[birth, death] for dim, (birth, death) in full_persistence if dim == dimension]


with open(DATA_PATH + 'file_name_to_persistence.P', 'rb') as pickle_file:
    filename_to_persistence = pickle.load(pickle_file)
print(filename_to_persistence)

filename_to_index = {}
for filename, persistence in filename_to_persistence.items():
    if filename not in filename_to_index:
        filename_to_index[filename] = len(filename_to_index)

print(filename_to_index)

for dimension in range(0, 3):
    Path(f'../cache/{all_vs_stop_word_path}/distance_matrices/dimension{dimension}/').mkdir(parents=True, exist_ok=True)
    with open(
            f'../cache/{all_vs_stop_word_path}/distance_matrices/dimension{dimension}/filename_to_distance_matrix_row_and_column_index.P',
            'wb') as filename_to_index_file:
        pickle.dump(filename_to_index, filename_to_index_file)

    distance_matrix = np.zeros((len(filename_to_index), len(filename_to_index)))
    for filename1, persistence1 in filename_to_persistence.items():
        for filename2, persistence2 in filename_to_persistence.items():
            distance_matrix[filename_to_index[filename1], filename_to_index[filename2]] = gudhi.bottleneck_distance(
                get_persistence_of_dimension(persistence1, dimension),
                get_persistence_of_dimension(persistence2, dimension))
            if filename_to_index[filename1] == filename_to_index[filename2]:
                print(distance_matrix[filename_to_index[filename1], filename_to_index[filename2]])

    np.save(f"../cache/{all_vs_stop_word_path}/distance_matrices/dimension{dimension}/distance_matrix", distance_matrix)
