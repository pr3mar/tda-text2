import numpy as np
import json
import re
import pathlib
import logging as logger
from os import listdir
from os.path import isfile, join
from sklearn.utils.extmath import randomized_svd
from laserembeddings import Laser
from time import time
import nltk
# nltk.download('punkt')  # run this line the first time only
logger.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logger.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
laser = Laser()
DATASET_BASE = "../dataset/"  # assumes work dir in $PROJECT_ROOT/code/


def get_genre_data(data, components=10, iters=5):
    result = {}
    for genre in data:
        begin = time()
        logger.info(f"[{genre}] Processing {genre}")
        M = get_genre_matrix(genre, data[genre])
        begin_svd = time()
        U, S, VT = randomized_svd(M, n_components=components, n_iter=iters, random_state=None)
        logger.debug(f"[{genre}] Computing SVD decomposition with k = {components} finished in {time() - begin_svd:.3f}")
        result[genre] = {
            "M": M,
            "U": U,
            "S": S,
            "VT": VT,
        }
        logger.info(f"[{genre}] Time for processing: {time() - begin}")
    return result


def get_genre_matrix(genre, authors):
    onlyfiles = [f for f in listdir(DATASET_BASE) if isfile(join(DATASET_BASE, f))]
    current = [x for x in onlyfiles if any(x.startswith(a) for a in authors)]
    M = np.zeros((1024, len(current)))  # each column is a document
    for i, book in enumerate(current):
        M[:, i] = get_document_matrix(book, genre)
    logger.debug(f"[{genre}] Saving matrix to cache.")
    np.save(f"../cache/{genre}", M)
    return M


def get_document_matrix(fname, genre):
    with open(f"{DATASET_BASE}{fname}") as file:
        logger.debug(f"[{genre}][{fname}] Processing `{fname}`")
        begin_t = time()
        sentences = tokenizer.tokenize(re.sub(r"(\s|\n)+", " ", file.read()))
        logger.debug(f"[{genre}][{fname}] Time used for tokenizing: {time() - begin_t:.3f}, #sentences: {len(sentences)}")
        begin_e = time()
        M = laser.embed_sentences(sentences, lang="en")
        logger.debug(f"[{genre}][{fname}] Time used for embeddings: {time() - begin_e:.3f}")
        logger.debug(f"[{genre}][{fname}] Time used for all: {time() - begin_t:.3f}")
        return np.mean(M, axis=0)


if __name__ == '__main__':
    pathlib.Path('../cache').mkdir(parents=True, exist_ok=True)        # stores distance matrices
    data = json.load(open("../data_categories.json"))
    genre_data = get_genre_data(data)

