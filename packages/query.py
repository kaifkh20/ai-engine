import math
import nltk
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)

DOCS_PATH = "docs.jsonl"
INDEX_PATH = "index.json"
STATS_PATH = "doc_stats.json"

def load_docs(path=DOCS_PATH):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(d)
    return docs

def load_index(path=INDEX_PATH):
    with open(path,"r") as f:
        index = json.load(f)
    return index

def load_stats(path=STATS_PATH):
    with open(path,"r") as f:
        stats = json.load(f)
    return stats
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

def ranking(scores,top_k=5):
    docs = load_docs(DOCS_PATH)
    doc_map = {str(doc["id"]) : doc["url"] for doc in docs}
    
    return [(doc_id,doc_map.get(doc_id,""),score) for doc_id,score in scores.most_common(top_k)]


def tf_idf(query_tokens,index_path=INDEX_PATH,stats_path=STATS_PATH):
    index = load_index(index_path)
    stats = load_stats(stats_path)

    N = stats["N"]
    doc_len = stats["doc_len"]

    scores = Counter()

    for word in query_tokens:
        if word not in index:
            continue
        df = index[word]["df"] 
        idf = math.log((N+1)/(df+1))+1  #smoothed idf or also laplace smoothing

        for doc_id,count in index[word]["docs"].items():
            tf = count/doc_len[doc_id]
            scores[doc_id] = tf*idf
    
    return scores

def bm25(query_tokens,index_path=INDEX_PATH,stats_path=STATS_PATH,k1=1.2,b=0.75):
    
    index = load_index(index_path)
    stats = load_stats(stats_path)

    N = stats["N"]
    doc_len = stats["doc_len"]
    avg_len = stats["avg_len"]

    scores = Counter()

    for word in query_tokens:
        if word not in index:
            continue
        df = index[word]["df"] 
        idf = math.log((N+1)/(df+1))+1  #smoothed idf or also laplace smoothing
        
        for doc_id,count in index[word]["docs"].items():
            
            ft_d = index[word]["docs"][doc_id]
            l_doc = doc_len[doc_id] 
            scores[doc_id] = idf*((ft_d*(k1+1))/(ft_d+(k1*((1-b)+b*(l_doc/avg_len)))))

    return scores

def query_index(query, index_path=INDEX_PATH):
    index = load_index(index_path)
    
    query_tokens = tokenize(query)
    
    tf_idf_scores = tf_idf(query_tokens)
    bm25_scores = bm25(query_tokens)

    return ranking(bm25_scores)

#sort the array in reverse according to the score and give us top-5 answer
def response_query(query):

    """
    scored = []
    for d in docs:
        score = query_score(query,d["text"])
        scored.append((score,d))


    scored.sort(key=lambda x:x[0],reverse=True)
    return [d for score,d in scored[:5]]
    """

    result_of_query = query_index(query,index_path=INDEX_PATH)
    
    print(f"Length of res_query : {len(result_of_query)}")

    for s_no,(doc_id,doc_url,score) in enumerate(result_of_query,1):
        print(f"{s_no} : Documnet Id :{doc_id} : {doc_url} with a score of {score}\n")

    print(f"Query Succesfull")

