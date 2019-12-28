import numpy as np
import requests
import nltk
# nltk.download('punkt')  # run this line the first time only
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
port = 8080  # the same as from the docker command

if __name__ == '__main__':
    url = f"http://127.0.0.1:{port}/vectorize"
    text = "Hey, how are you? I'm OK and you?! this is \"another sentence\""
    sentences = tokenizer.tokenize(text)
    print(sentences)
    for s in sentences:
        params = {"q": s, "lang": "en"}
        resp = np.array(requests.get(url=url, params=params).json()["embedding"])
        print(resp.shape)
        print(resp)

