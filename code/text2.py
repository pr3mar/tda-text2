import numpy as np
import requests
import nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

if __name__ == '__main__':
    url = "http://127.0.0.1:8080/vectorize"
    text = "Hey, how are you? I'm OK and you? This is another sentence."
    sentences = tokenizer.tokenize(text)
    print(sentences)
    for s in sentences:
        params = {"q": s, "lang": "en"}
        resp = np.array(requests.get(url=url, params=params).json()["embedding"])
        print(resp.shape)
        print(resp)

