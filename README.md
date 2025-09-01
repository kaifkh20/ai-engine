# AI Search Engine with Client–Server Architecture 
<a href="https://youtu.be/I2MTm8PGtPQ"> Video Example</a>

A modular information retrieval system built from scratch with traditional IR, neural reranking, vector search, and RAG-style generation.  
The system exposes a CLI-based client–server interface.

## IMPORTANT
This tool uses GEMINI-api so please create an .env file with 
GEMINI_API=YOUR_API_KEY

for it to work without failure 
---

## Features

1. **Crawler**
   - Starts from manual seeds or expands dynamically based on queries.
   - Extracts and stores page text into `docs.jsonl`.

2. **Indexing**
   - Builds inverted index (`index.json`).
   - Tracks term frequency, document frequency, and lengths.

3. **Ranking**
   - Implements **TF-IDF** and **BM25**.
   - **Phrase queries** and **fuzzy matching** supported.

4. **Vector Search**
   - Stores dense embeddings using **FAISS**.
   - Query embeddings compared to document embeddings for semantic search.

5. **Hybrid Scoring**
   - Combines BM25 and FAISS scores with weighted merging.
   - Produces top-k candidates.

6. **Reranking**
   - Uses a **cross-encoder** for fine-grained scoring of top candidates.

7. **RAG-style Generation**
   - Top retrieved documents passed with query to a generative model (e.g., Gemini).
   - Produces coherent, context-aware answers.
     
8. **Text Preprocessing**
   - Pre-process query and document text
   - Tokenization + Stemmerization
    
10. **Client–Server Architecture**
   - **Server**: Hosts index, embeddings, and search logic.
   - **Client**: CLI that sends queries, receives results, and displays answers.

---

## AI Components
# Vector Search with Embeddings (FAISS)

   - Documents and queries are converted into dense embeddings using a language model.

   - This allows semantic search (matching meaning, not just words).

# Cross-Encoder Reranker

   - A transformer-based model scores the relevance of query–document pairs.

   - Unlike BM25 (bag-of-words), this uses deep contextual understanding of language.

# RAG-style Answer Generation

  - A generative model (e.g., Gemini) is given the query + retrieved context.

  - It produces a natural-language answer, simulating an intelligent assistant.

## Installation

```bash
git clone https://github.com/kaifkh20/ai-engine
cd ai-engine
pip install -r req.txt

```
# Usage
# Start the server

Make sure Python 3.11 or 3.10 is installed.

```
python -m packages.crawler #to run the crawler run this only if its first time
                           # or you update the crawler_config.json
python server.py

```

# Client 

```
python client.py "your query here"

```

# Dependencies

As stated in req.text

# Next Steps

 - Scale crawling for larger datasets.

 - Web Client

 - Extend client with more commands.

 - Benchmark against standard IR datasets.
