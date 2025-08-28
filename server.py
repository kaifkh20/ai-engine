from packages import query,crawler
from flask import Flask,request,abort,jsonify
app = Flask(__name__)

@app.route("/query")
def query_search():
    q = request.args.get('q')  # Changed from 'name' to 'q'
    if not q: 
        abort(400, description="Invalid Query - 'q' parameter is missing or empty.")
    

    rag_answer, list_of_docs = query.response_query(q)
    
    ans_docs = []

    for doc in list_of_docs:
        ans_docs.append((float(doc[0]),doc[1]))


    return jsonify({
        "rag_answer": rag_answer,
        "list_of_docs": ans_docs
    })

if __name__ == "__main__":  # Fixed the syntax here
    app.run(debug=True, host='127.0.0.1', port=8080)
