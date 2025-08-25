import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)


def load_docs(path="docs.jsonl"):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(d)
    return docs

#checking for overlapping tokens to rank the response

def tokenize(s):
    
    stop_words = set(stopwords.words('english'))
    # Tokenize and normalize to lowercase
    tokens = word_tokenize(s.lower())
    # Filter out stopwords
    return [w for w in tokens if w not in stop_words and w.isalnum()]

def query_score(q,d):
    qset = set(tokenize(q))
    dset = set(tokenize(d))

    return len(qset.intersection(dset))
#sort the array in reverse according to the score and give us top-5 answer
def response_query(query,docs):

    scored = []
    for d in docs:
        score = query_score(query,d["text"])
        scored.append((score,d))


    scored.sort(key=lambda x:x[0],reverse=True)
    return [d for score,d in scored[:5]]
    
