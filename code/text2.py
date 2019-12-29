import numpy as np
import json
import re
from os import listdir
from os.path import isfile, join
from laserembeddings import Laser
from time import time
import nltk
# nltk.download('punkt')  # run this line the first time only

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
laser = Laser()
DATASET_BASE = "../dataset/"



def get_genre_matrix(genre, authors):
    onlyfiles = [f for f in listdir(DATASET_BASE) if isfile(join(DATASET_BASE, f))]
    current = [x for x in onlyfiles if any(x.startswith(a) for a in authors)]
    M = np.zeros((len(current), 1024))
    for i, book in enumerate(current):
        M[i, :] = get_file_matrix(book)
    np.save(f"../cache/{genre}", M)
    return M


def get_file_matrix(fname):
    with open(f"{DATASET_BASE}{fname}") as file:
        sentences = tokenizer.tokenize(re.sub(r"(\s|\n)+", " ", file.read()))
        return np.mean(laser.embed_sentences(sentences, lang="en"), axis=0)


if __name__ == '__main__':
    # embeddings = get_file_matrix('../dataset/Thornton Waldo Burgess___Whitefoot the Wood Mouse.txt')
    # print(embeddings.shape)
    data = json.load(open("../data_categories.json"))
    matrices = []
    for genre in data:
        begin = time()
        matrices.append(get_genre_matrix(genre, data[genre]))
        print(f"Time for processing: {time() - begin}")
    # text = "Hey, how are you? I'm OK and you?! this is \"another sentence\". Woo hooo"
    # sentences = tokenizer.tokenize(text)
    # print(sentences)
    # embeddings = laser.embed_sentences(sentences, lang='en')
    # print(embeddings.shape)
    # print(embeddings)
