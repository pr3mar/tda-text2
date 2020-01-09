import numpy as np
import os
import gudhi
from bisect import bisect_left, bisect_right
import pickle
import matplotlib.pyplot as plt

DATA_PATH = '../cache/all_words/reduced_dimension/'

file_name_to_persistence = {}


def euclidian_distance(a, b):
    return np.linalg.norm(a - b)


def largest_number_in_list_smaller_than_k(l, x):
    return bisect_left(l, x) - 1
    # i = bisect_left(l, x)
    # if i:
    #     return (i - 1)
    # else:
    #     return -1


def smallest_number_in_list_larger_or_equal_than_k(l, k):
    for elem in l:
        if elem >= k:
            return elem
    return k
    # if k > l[-1]:
    #     print('SHOULD BE INF', k)
    #     return k
    # return l[bisect_right(l, k)]


def get_max_R(persistence, R, max_dimension=2):
    real_R = -1
    for dim, (birth, death) in persistence:
        if dim > max_dimension:
            continue
        if R >= death > real_R:
            real_R = death
    return real_R


def build_partition_points(R):
    return [0 + R / 10 * i for i in range(11)]


def fix_persistence(persistence, maximal_distance_R, max_dimension=2):
    R = get_max_R(persistence, maximal_distance_R, max_dimension=max_dimension)
    print("REAL R", R)
    partition_points = build_partition_points(R)

    fixed_persistence = []
    for dim, (birth, death) in persistence:
        if dim > max_dimension:
            continue
        new_birth = smallest_number_in_list_larger_or_equal_than_k(partition_points, birth)
        new_death = smallest_number_in_list_larger_or_equal_than_k(partition_points, death)
        fixed_persistence.append((dim, (new_birth, new_death)))
    return fixed_persistence


for file_name in os.listdir(DATA_PATH):
    if file_name.endswith('.npy'):
        # if file_name != 'plays.npy' and file_name != 'mysticism.npy':
        #     continue
        print('Working on', file_name)
        matrix = np.load(DATA_PATH + file_name).transpose()
        maximal_distance_R = -1
        for index1, document1 in enumerate(matrix):
            for index2, document2 in enumerate(matrix):
                if index2 <= index1:
                    continue

                maximal_distance_R = max(maximal_distance_R, euclidian_distance(document1, document2))

        # file_name_to_R[file_name] = maximal_distance_R

        print(maximal_distance_R)

        alpha_complex = gudhi.AlphaComplex(points=matrix)
        persistence = alpha_complex.create_simplex_tree().persistence()
        fixed_persistence = fix_persistence(persistence, maximal_distance_R)
        file_name_to_persistence[file_name] = fixed_persistence
        # break

print(file_name_to_persistence)
with open(DATA_PATH + 'file_name_to_persistence.P', 'wb') as pickle_file:
    pickle.dump(file_name_to_persistence, pickle_file)

for filename, persistence in file_name_to_persistence.items():
    plt.figure()
    fig = gudhi.plot_persistence_diagram(persistence, legend=True)
    fig.savefig(f"../Persistence diagram plots/{filename.split('.')[0]}.png")
