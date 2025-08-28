import argparse
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    args = parser.parse_args()
    
    resp = requests.get(
        "http://127.0.0.1:8080/query",
        params={"q": args.query}        
    )
    
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} - {resp.text}")
        return
    
    data = resp.json()
    
    print("Search results:")
    
    rag_answer = data["rag_answer"].split(":")[1]
    list_of_docs = data["list_of_docs"]
    print(f"\n{rag_answer}\n")
    
    for idx, (doc_id, doc_url) in enumerate(list_of_docs, 1):
        print(f"S.no : {idx} ID:{doc_id} - {doc_url}")

if __name__ == "__main__":
    main()
