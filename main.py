from packages import query,crawler

def main():
    
    crawler.crawl()

    query_input = input("query: ")
    
    query.response_query(query_input)

if __name__=="__main__":
    main()

