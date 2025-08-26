from packages import query,crawler

def main():
    
    crawler.crawl()

    docs = query.load_docs("docs.jsonl")

    query_input = input("query: ")
    
    query.response_query(query_input)

if __name__=="__main__":
    main()

