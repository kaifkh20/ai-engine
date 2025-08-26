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

def ranking(scores,top_k=10):
    docs = load_docs(DOCS_PATH)
    doc_map = {str(doc["id"]) : doc["url"] for doc in docs}
    
    return [(doc_id,doc_map.get(doc_id,""),score) for doc_id,score in scores.most_common(top_k)]


def tf_idf(query_tokens, index_path=INDEX_PATH, stats_path=STATS_PATH):
    index = load_index(index_path)
    stats = load_stats(stats_path)
    N = stats["N"]
    doc_len = stats["doc_len"]
    scores = Counter()
    
    for word in query_tokens:
        if word not in index:
            continue
        df = index[word]["df"] 
        idf = math.log((N+1)/(df+1))+1  # smoothed idf or also laplace smoothing
        
        for doc_id, doc_info in index[word]["docs"].items():
            tf_raw = doc_info["tf"]  # Get tf from the nested dictionary
            tf = tf_raw / doc_len[doc_id]  # Normalize by document length
            scores[doc_id] += tf * idf  # Use += to accumulate scores for multiple query terms
    
    return scores


def bm25(query_tokens, index_path=INDEX_PATH, stats_path=STATS_PATH, k1=1.2, b=0.75):
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
        idf = math.log((N+1)/(df+1))+1  # smoothed idf or also laplace smoothing
        
        for doc_id, doc_info in index[word]["docs"].items():
            ft_d = doc_info["tf"]  # Get tf from the nested dictionary
            l_doc = doc_len[doc_id] 
            bm25_score = idf * ((ft_d * (k1 + 1)) / (ft_d + (k1 * ((1 - b) + b * (l_doc / avg_len)))))
            scores[doc_id] += bm25_score  # Use += to accumulate scores for multiple query terms
    
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
            
def check_phrase_in_doc(positions_list):
    """
    positions_list: [[pos1, pos2...], [pos1, pos2...], ...]
    Returns count of phrase occurrences in doc
    """
    count = 0
    # For each position of the first term
    for pos in positions_list[0]:
        match = True
        # Check if following terms appear in sequence
        for i in range(1, len(positions_list)):
            if (pos + i) not in positions_list[i]:
                match = False
                break
        if match:
            count += 1
    return count

def phrase_search(phrase_terms, index_file=INDEX_PATH):
    index = load_index()
    docs = load_docs()
    doc_map = {str(doc["id"]) : doc["url"] for doc in docs}

    postings = []
    for term in phrase_terms:
        if term not in index:
            return []
        postings.append(index[term]["docs"])

    common_docs = set(postings[0].keys())
    for p in postings[1:]:
        common_docs &= set(p.keys())
    
    scores = Counter()
    for doc_id in common_docs:
        positions_list = [postings[i][doc_id]["pos"] for i in range(len(phrase_terms))]
        phrase_count = check_phrase_in_doc(positions_list)
        if phrase_count > 0:
            scores[doc_id] = phrase_count  # simple scoring
    
    return ranking(scores)

def normal_search(tokens, index_path):
    return query_index(tokens, index_path=index_path)  

def preprocess_query(query):
    tokens = tp.text_preprocess(query)
    tokens = fuzzy_fill(tokens)
    return tokens

def merge_results(normal_results, phrase_results, phrase_boost=2.0):
    combined_scores = {}
    
    # Add normal results
    for doc_id,doc_url, score in normal_results:
        combined_scores[doc_id] = {"url":doc_url, "score": score}

    # Add phrase results with boost
    for doc_id,doc_url,phrase_score in phrase_results:
        if doc_id in combined_scores:
            combined_scores[doc_id]["score"] += phrase_score * phrase_boost
        else:
            combined_scores[doc_id] = {"url":doc_url, "score": phrase_score * phrase_boost}

    # Sort

    return sorted(
        [(doc_id, vals["url"], vals["score"]) for doc_id, vals in combined_scores.items()],
        key=lambda x: x[2],
        reverse=True
    )
    

def response_query(query):
    
    tokens = preprocess_query(query)

    normal_results = normal_search(tokens, INDEX_PATH)
    phrase_results = phrase_search(tokens, INDEX_PATH)

    final_results = merge_results(normal_results, phrase_results)

    print(f"Length of res_query : {len(final_results)}")
    for s_no, (doc_id, doc_url, score) in enumerate(final_results, 1):
        print(f"{s_no}: Document Id: {doc_id} | {doc_url} | Score: {score}")
    print("Query Successful")

