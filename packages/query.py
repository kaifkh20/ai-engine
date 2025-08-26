import nltk
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)

DOCS_PATH = "docs.jsonl"
INDEX_PATH = "index.json"

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

def query_index(query, index_path=INDEX_PATH):
    index = load_index(index_path)
    
    #print(f"Loaded index: {len(index)} words")
    #print(f"Sample index keys: {list(index.keys())[:10]}")  # Show first 10 keys
    
    query_tokens = tokenize(query)
    #print(f"Query tokens: {query_tokens}")
    
    scores = Counter()
    for word in query_tokens:
        #print(f"Looking for word: '{word}'")
        if word in index:
           #print(f"  Found '{word}' in index with {len(index[word])} documents")
            for doc_id, count in index[word].items():
                scores[doc_id] += count
                #print(f"    Doc {doc_id}: +{count} (total: {scores[doc_id]})")
        #else:
            #print(f"  '{word}' NOT found in index")
    
    #print(f"Final scores: {scores}")
    
    docs = load_docs(path=DOCS_PATH)
    resulted_docs = {}
    for id,doc in enumerate(docs,1):
        resulted_docs[id] = doc["url"]
    

    return [(doc_id, resulted_docs[int(doc_id)]) for doc_id, _ in scores.most_common(5)]


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

    for s_no,(doc_id,doc_url) in enumerate(result_of_query,1):
        print(f"{s_no} : Documnet Id :{doc_id} : {doc_url} \n")

    print(f"Query Succesfull")

