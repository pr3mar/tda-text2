import numpy as np
import requests
import nltk
from laserembeddings import Laser
# nltk.download('punkt')  # run this line the first time only

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
laser = Laser()

if __name__ == '__main__':
    text = "Hey, how are you? I'm OK and you?! this is \"another sentence\". Woo hooo"
    sentences = tokenizer.tokenize(text)
    print(sentences)
    embeddings = laser.embed_sentences(sentences, lang='en')
    print(embeddings.shape)
    print(embeddings)
