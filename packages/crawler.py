import requests
from bs4 import BeautifulSoup
import json 
import os
from datetime import datetime
from collections import defaultdict, Counter

from . import text_preprocess as tp

# List of seed URLs
SEEDS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Information_retrieval",
    "https://en.wikipedia.org/wiki/Natural_language_processing"

]
DOCS_FILE = "docs.jsonl"
INDEX_FILE = "index.json"
STATS_FILE = "doc_stats.json"

def load_existing_docs():
    """Load existing documents from file and return a set of URLs and the documents list."""
    existing_urls = set()
    existing_docs = []
    
    if os.path.exists(DOCS_FILE):
        try:
            with open(DOCS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line.strip())
                        existing_urls.add(doc.get("url"))
                        existing_docs.append(doc)
            print(f"Loaded {len(existing_docs)} existing documents")
        except Exception as e:
            print(f"Error loading existing docs: {e}")
    
    return existing_urls, existing_docs

def fetch_page(url):
    try:
        headers = {"User-Agent": "AI/SearchBot/0.1"}
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"Failed to fetch page {url}: {e}")
        return None

def extract_content(html, url):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else "Untitled"
    text = soup.get_text(separator=" ", strip=True)
    text_tokens = tp.text_preprocess(text)
    
    return {
        "url": url,
        "title": title,
        "text": ' '.join(text_tokens),
        "fetched_at": datetime.utcnow().isoformat()
    }

def build_inverted_index(docs_file=DOCS_FILE, index_file=INDEX_FILE,stats_file=STATS_FILE):
    """Build complete inverted index from all documents."""
    index = defaultdict(lambda:{"docs":{},"df":0})
    
    doc_len = {}
    total_docs = 0

    with open(docs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line.strip())
                doc_id, text = str(doc["id"]), doc["text"].split()  # Ensure doc_id is string
                
                total_docs += 1
                doc_len[doc_id] = len(text)

                freqs = Counter(text)
                
                for word, count in freqs.items():
                    index[word]["docs"][doc_id] = count
                    index[word]["df"] +=1
                
    # Save index
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    
    # Save doc stats
    avg_len = sum(doc_len.values()) / total_docs if total_docs > 0 else 0
    stats = {"doc_len": doc_len, "avg_len": avg_len, "N": total_docs}
    
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)
    
    print(f"Inverted index built with {len(index)} unique words across {total_docs} docs")


def update_inverted_index(new_docs, index_file=INDEX_FILE, stats_file=STATS_FILE):
    """Update existing index with new documents only."""
    # Load existing index
    existing_index = defaultdict(lambda: {"docs": {}, "df": 0})
    if os.path.exists(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                loaded_index = json.load(f)
                for word, doc_dict in loaded_index.items():
                    existing_index[word] = doc_dict
            print(f"Loaded existing index with {len(existing_index)} words")
        except Exception as e:
            print(f"Error loading existing index: {e}")

    # Load stats
    if os.path.exists(stats_file):
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        doc_len = stats.get("doc_len", {})
        total_docs = stats.get("N", 0)
    else:
        doc_len = {}
        total_docs = 0

    # Add new documents
    for doc in new_docs:
        doc_id, tokens = str(doc["id"]), doc["text"].split()

        if doc_id in doc_len:  # skip if already indexed
            continue

        total_docs += 1
        doc_len[doc_id] = len(tokens)

        freqs = Counter(tokens)
        for word, count in freqs.items():
            existing_index[word]["docs"][doc_id] = count
            existing_index[word]["df"] += 1  # increase doc frequency by 1 for this doc

    # Save updated index
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(existing_index, f, ensure_ascii=False)

    # Save stats
    avg_len = sum(doc_len.values()) / total_docs if total_docs > 0 else 0
    stats = {"doc_len": doc_len, "avg_len": avg_len, "N": total_docs}

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

    print(f"Updated index with {len(new_docs)} new documents. Total docs: {total_docs}, Total words: {len(existing_index)}")

def save_to_file():
    # Load existing documents
    existing_urls, existing_docs = load_existing_docs()
    
    new_docs = []
    skipped_count = 0
    
    for url in SEEDS:
        if url in existing_urls:
            print(f"Skipping already fetched URL: {url}")
            skipped_count += 1
            continue
            
        print(f"Fetching new URL: {url}")
        html = fetch_page(url)
        if not html:
            continue
            
        doc = extract_content(html, url)
        # Assign ID based on total number of docs (existing + new)
        doc["id"] = len(existing_docs) + len(new_docs) + 1
        new_docs.append(doc)
    
    if new_docs:
        # Append new documents to the file
        with open(DOCS_FILE, "a", encoding="utf-8") as f:
            for doc in new_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"Added {len(new_docs)} new documents to {DOCS_FILE}")
        
        # Only update index if we have new documents
        if os.path.exists(INDEX_FILE):
            update_inverted_index(new_docs, index_file=INDEX_FILE)
        else:
            # If index doesn't exist, build it from scratch
            print("Index file doesn't exist. Building complete index...")
            build_inverted_index(docs_file=DOCS_FILE, index_file=INDEX_FILE)
    else:
        print("No new documents to add - index remains unchanged")
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} already existing documents")
    
    total_docs = len(existing_docs) + len(new_docs)
    print(f"Total documents in collection: {total_docs}")

def crawl():
    save_to_file()
