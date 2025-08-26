import math
import json
from collections import Counter
from . import text_preprocess as tp
from thefuzz import process, fuzz

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

def query_index(query_tokens, index_path=INDEX_PATH):
    
    index = load_index(index_path)
    
    
    tf_idf_scores = tf_idf(query_tokens)
    bm25_scores = bm25(query_tokens)

    return ranking(bm25_scores)


def fuzzy_fill(query_tokens,cutoff=80,index_path=INDEX_PATH):
    
    index = load_index(index_path)
    
    modified_tokens = []

    for token in query_tokens:
        if token not in index:
            # find closest match in index keys using fuzzy matching
            token_to_replace = process.extractOne(
                token,
                index.keys(),
                score_cutoff=cutoff
            )
        
            if token_to_replace:
                matched_token = token_to_replace[0]
                modified_tokens.append(matched_token)
        else:
            modified_tokens.append(token)

    return modified_tokens
            

def response_query(query):

    """
    scored = []
    for d in docs:
        score = query_score(query,d["text"])
        scored.append((score,d))


    scored.sort(key=lambda x:x[0],reverse=True)
    return [d for score,d in scored[:5]]
    """
    query_tokens = tp.text_preprocess(query)
    
    query_tokens = fuzzy_fill(query_tokens)

    result_of_query = query_index(query_tokens,index_path=INDEX_PATH)
    
    print(f"Length of res_query : {len(result_of_query)}")

    for s_no,(doc_id,doc_url,score) in enumerate(result_of_query,1):
        print(f"{s_no} : Documnet Id :{doc_id} : {doc_url} with a score of {score}\n")

    print(f"Query Succesfull")

