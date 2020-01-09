import logging as logger
import numpy as np
import json
import re
import pathlib
import nltk
from os import listdir
from os.path import isfile, join
from sklearn.utils.extmath import randomized_svd
from laserembeddings import Laser
from time import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('punkt')        # run this line the first time only
# nltk.download('stopwords')    # run this line the first time only
logger.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logger.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words = set(stopwords.words('english'))
laser = Laser()
DATASET_BASE = "../dataset/"  # assumes work dir in $PROJECT_ROOT/code/


def get_genre_data(data, components=10, iters=5, remove_stop_words=False):
    result = {}
    for genre in data:
        begin = time()
        logger.info(f"[{genre}] Processing {genre}")
        M = get_genre_matrix(genre, data[genre], remove_stop_words)
        begin_svd = time()
        U, S, VT = randomized_svd(M, n_components=components, n_iter=iters, random_state=None)
        logger.debug(
            f"[{genre}] Computing SVD decomposition with k = {components} finished in {time() - begin_svd:.3f}")
        result[genre] = {
            "M": M,
            "U": U,
            "S": S,
            "VT": VT,
        }
        logger.info(f"[{genre}] Time for processing: {time() - begin}")
    return result


def get_genre_matrix(genre, authors, remove_stop_words):
    onlyfiles = [f for f in listdir(DATASET_BASE) if isfile(join(DATASET_BASE, f))]
    current = [x for x in onlyfiles if any(x.startswith(a) for a in authors)]
    M = np.zeros((1024, len(current)))  # each column is a document
    for i, book in enumerate(current):
        M[:, i] = get_document_matrix(book, genre, remove_stop_words)
    logger.debug(f"[{genre}] Saving matrix to cache.")
    if remove_stop_words:
        np.save(f"../cache/stopwords_excluded/full_dimension/{genre}", M)
    else:
        np.save(f"../cache/full_dimension/full_dimension/{genre}", M)
    return M


def get_document_matrix(fname, genre, remove_stop_words):
    with open(f"{DATASET_BASE}{fname}") as file:
        logger.debug(f"[{genre}][{fname}] Processing `{fname}`")
        begin_t = time()
        sentences = tokenizer.tokenize(re.sub(r"(\s|\n)+", " ", file.read()))
        if remove_stop_words:
            logger.debug(f"[{genre}][{fname}] Removing stop words.")
            sentences = [" ".join([w for w in word_tokenize(s) if w not in stop_words]) for s in sentences]
        logger.debug(f"[{genre}][{fname}] Time used for tokenizing: {time() - begin_t:.3f}, #sentences: {len(sentences)}")
        begin_e = time()
        M = laser.embed_sentences(sentences, lang="en")
        logger.debug(f"[{genre}][{fname}] Time used for embeddings: {time() - begin_e:.3f}")
        logger.debug(f"[{genre}][{fname}] Time used for all: {time() - begin_t:.3f}")
        return np.mean(M, axis=0)


if __name__ == '__main__':
    pathlib.Path('../cache/').mkdir(parents=True, exist_ok=True)
    begin = time()
    data = json.load(open("../data_categories.json"))
    genre_data = get_genre_data(data, remove_stop_words=True)
    logger.info(f"Total time: {time() - begin}")
