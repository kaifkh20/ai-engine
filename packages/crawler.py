import requests
from bs4 import BeautifulSoup
import json 
import os
from datetime import datetime
from collections import defaultdict, Counter
from urllib.parse import urljoin, urlparse
import time
import random
from typing import Set, List, Dict, Optional

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from packages import text_preprocess as tp


DOCS_FILE = "docs.jsonl"
INDEX_FILE = "index.json"
STATS_FILE = "doc_stats.json"
FAISS_FILE = "index.faiss"
VECTOR_FILE = "vector.json"
CONFIG_FILE = "crawler_config.json"

class URLDiscoverer:
    """Handles URL discovery from various sources"""
    
    def __init__(self, config_file="crawler_config.json"):
        self.load_config(config_file)
        self.discovered_urls: Set[str] = set()
        
    def load_config(self, config_file=CONFIG_FILE):
        """Load URL discovery configuration"""
        
        default_config = {}

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        self.config = default_config
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be included"""
        parsed = urlparse(url)
        
        # Check allowed domains
        if self.config["allowed_domains"]:
            if parsed.netloc not in self.config["allowed_domains"]:
                return False
        
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            return False
            
        return True
   
    def discover_from_manual_seeds(self):
        """Discover URLs from manually specified seeds in the config"""
        urls = []
        seeds = self.config.get("manual_seeds", [])
        max_urls = self.config.get("max_urls_per_source", 50)

        for url in seeds[:max_urls]:
            if self.is_valid_url(url):
                urls.append(url)
            else:
                print(f"Invalid manual seed URL skipped: {url}")

        print(f"Discovered {len(urls)} URLs from manual seeds")
        return urls 

    def discover_from_sitemaps(self):
        """Discover URLs from XML sitemaps"""
        urls = []
        max_urls = self.config["max_urls_per_source"]
        
        for sitemap_url in self.config["sitemap_urls"]:
            print(f"Processing sitemap: {sitemap_url}")
            
            try:
                headers = {
                    "User-Agent": "AITestingBot/1.0 (mailto:kaifkhan.saif@gmail.com) PythonRequestsForToyProject/2.0"
                } 
                response = requests.get(sitemap_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'xml')
                
                # Extract URLs from sitemap
                url_elements = soup.find_all('url')
                for url_elem in url_elements[:max_urls]:
                    loc = url_elem.find('loc')
                    if loc and self.is_valid_url(loc.text):
                        urls.append(loc.text)
                        
                time.sleep(self.config["delay_between_requests"])
                        
            except Exception as e:
                print(f"Error processing sitemap {sitemap_url}: {e}")
        
        print(f"Discovered {len(urls)} URLs from sitemaps")
        return urls
    
    def discover_from_wikipedia_api(self) -> List[str]:
        """Discover URLs using Wikipedia search API"""
        urls = []
        wiki_config = self.config["search_apis"]["wikipedia"]

        
        if not wiki_config.get("enabled"):
            return urls
        
        base_url = wiki_config["base_url"]
        topics = wiki_config["topics"]
        max_per_topic = self.config["max_urls_per_source"] // len(topics) if topics else 10
        
        for topic in topics:
            print(f"Searching Wikipedia for: {topic['topic']}")
            
            if topic["fetched"]==True:
                print("Already fetched from wiki_api")
                continue

            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': topic["topic"],
                'srlimit': max_per_topic
            }
            
            try:
                headers = {
                    "User-Agent": "AITestingBot/1.0 (mailto:kaifkhan.saif@gmail.com) PythonRequestsForToyProject/2.0"
                } 
                response = requests.get(base_url, params=params,headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'query' in data and 'search' in data['query']:
                    for result in data['query']['search']:
                        page_title = result['title']
                        url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                        
                        if self.is_valid_url(url):
                            urls.append(url)
                            topic["fetched"] = True               
                time.sleep(self.config["delay_between_requests"])
                            
            except Exception as e:
                print(f"Error searching Wikipedia for '{topic}': {e}")
        
        print(f"Discovered {len(urls)} URLs from Wikipedia API")
        return urls
    
    def update_crawler_config(self, config_file="crawler_config.json"):

        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    

    def discover_all_urls(self):
        """Main method to discover URLs from all sources"""
        all_urls = []
        print("=== Starting URL Discovery ===")

        # Method 1: Sitemaps
        sitemap_urls = self.discover_from_sitemaps()
        all_urls.extend(sitemap_urls)

        # Method 2: Wikipedia API
        wiki_urls = self.discover_from_wikipedia_api()
        all_urls.extend(wiki_urls)

        # Method 3: Manual seeds
        manual_urls = self.discover_from_manual_seeds()
        all_urls.extend(manual_urls)

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(all_urls))
        
        self.update_crawler_config()
        print(f"Total unique URLs discovered: {len(unique_urls)}")
        return unique_urls


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
        "fetched_at": datetime.utcnow().isoformat(),
        "raw_text" : text
    }

def add_to_faiss(doc_id, doc_text, update=False, faiss_file=FAISS_FILE,vector_file=VECTOR_FILE):
    """
    Add document to FAISS index.
    
    """
    
    # Input validation
    if not doc_id or not doc_text:
        raise ValueError("doc_id and doc_text cannot be empty")
    
    # Initialize model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Generate embeddings
    embeddings = model.encode([doc_text])
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]  # Usually 384 for all-MiniLM-L6-v2
    
    if os.path.exists(faiss_file):
        # UPDATE MODE: Load existing index
        try:
            index = faiss.read_index(faiss_file)
            print(f"Existing index loaded successfully with {index.ntotal} vectors")
            
            # Load existing mapping
            try:
                with open(vector_file, "r") as f:
                    existing_mapping = json.load(f)
            except FileNotFoundError:
                existing_mapping = {}
                print("Warning: Vector mapping file not found, creating new mapping")
                
        except Exception as e:
            print(f"Error loading existing index: {e}")
            print("Creating new index instead")
            index = faiss.IndexFlatL2(dimension)
            existing_mapping = {}
    else:
        print("Building new index from scratch")
        index = faiss.IndexFlatL2(dimension)
        existing_mapping = {}
    
    # Check if doc_id already exists
    if doc_id in existing_mapping:
        print(f"Warning: doc_id '{doc_id}' already exists. It will be overwritten.")
    
    # Add new embedding to index
    index.add(embeddings)
    vector_index_id = index.ntotal - 1
    
    # Update mapping
    existing_mapping[doc_id] = vector_index_id
    
    # Save updated index
    try:
        faiss.write_index(index, faiss_file)
        print(f"Index saved successfully to {faiss_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS index: {e}")
    
    # Save updated mapping
    try:
        
        with open(vector_file, "w") as f:
            json.dump(existing_mapping, f)
        print(f"Vector mapping saved successfully to {faiss_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save vector mapping: {e}")
    
    print(f"Document '{doc_id}' added successfully at index position {vector_index_id}")


def build_inverted_index(docs_file=DOCS_FILE, index_file=INDEX_FILE, stats_file=STATS_FILE):
    """Build complete inverted index from all documents with positional information."""
    index = defaultdict(lambda: {"docs": {}, "df": 0})
    
    doc_len = {}
    total_docs = 0
    with open(docs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line.strip())
                doc_id, text = str(doc["id"]), doc["text"].split()  # Ensure doc_id is string
                
                total_docs += 1
                doc_len[doc_id] = len(text)
                
                # Track word positions
                word_positions = defaultdict(list)
                for pos, word in enumerate(text):
                    word_positions[word].append(pos)
                
                # Build index with tf and positions
                for word, positions in word_positions.items():
                    index[word]["docs"][doc_id] = {
                        "tf": len(positions),
                        "pos": positions
                    }
                    index[word]["df"] += 1
                add_to_faiss(doc_id=doc_id,doc_text=doc["text"])
                
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
        
        # Track word positions
        word_positions = defaultdict(list)
        for pos, word in enumerate(tokens):
            word_positions[word].append(pos)
        
        # Update index with tf and positions
        for word, positions in word_positions.items():
            existing_index[word]["docs"][doc_id] = {
                "tf": len(positions),
                "pos": positions
            }
            existing_index[word]["df"] += 1  # increase doc frequency by 1 for this doc
        
        add_to_faiss(doc_id=doc_id,doc_text=doc["text"])
    
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
    
    discoverer = URLDiscoverer()
    seeds = discoverer.discover_all_urls()
    print(f"Found {len(seeds)} URLs to crawl") 

    new_docs = []
    skipped_count = 0
    
    for url in seeds:
        if url in existing_urls:
            print(f"Skipping already fetched URL: {url}")
            skipped_count += 1
            continue
            
        print(f"Fetching new URL: {url}")
        html = fetch_page(url)
        if not html or not url:
            continue
            
        doc = extract_content(html, url)
        # Assign ID based on total number of docs (existing + new)
        doc["id"] = str(len(existing_docs) + len(new_docs) + 1)
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
        
        #only update fssai when we have new docs
        
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


def crawl(query=None):
    '''
    save_to_file_flag = False
    if query is not None:
        discoverer = URLDiscoverer()
        discoverer.load_config()
        config = discoverer.config
        wiki_config = config["search_apis"]["wikipedia"]["topics"]
        
        topic_present = False
        for topic in wiki_config:
            if topic["topic"]==query:
                topic_present = True
                break

        if topic_present==False:
            append_topic = {"topic":query,"fetched":False}
            wiki_config.append(append_topic)
            
            discoverer.update_crawler_config()
            
            save_to_file_flag = True 
    '''
    if query is not None and save_to_file_flag is True:
        save_to_file()
    elif query is None:
        save_to_file()
    
if __name__=='__main__':
    crawl(query=None)
