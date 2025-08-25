import requests
from bs4 import BeautifulSoup
import json 
import os
from datetime import datetime

# List of seed URLs
SEEDS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Information_retrieval",
    "https://en.wikipedia.org/wiki/Natural_language_processing"
]

DOCS_FILE = "docs.jsonl"

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
    return {
        "url": url,
        "title": title,
        "text": text,
        "fetched_at": datetime.utcnow().isoformat()
    }

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
    else:
        print("No new documents to add")
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} already existing documents")
    
    total_docs = len(existing_docs) + len(new_docs)
    print(f"Total documents in collection: {total_docs}")

def crawl():
    save_to_file()

