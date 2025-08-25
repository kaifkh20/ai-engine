from packages import query,crawler

def main():
    
    crawler.crawl()

    docs = query.load_docs("docs.jsonl")

    query_input = input("query: ")
    
    results = query.response_query(query_input,docs)
    

    for i, d in enumerate(results, 1):
        print(f"{i}. {d['title']} ({d['url']})")
        print("   ", d["text"][:200], "...\n")  # snippet

if __name__=="__main__":
    main()

