# app.py

from flask import Flask, request, jsonify, render_template
from rag_engine import retrieve_answer
import csv
import os
app = Flask(__name__)

LOG_FILE = "query_answer_log.csv"
chat_history = []

def log_query_answer(query, answer):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["query", "answer"])  # header
        writer.writerow([ query, answer])


@app.route("/")
def index():
    return render_template("index.html")
@app.route("/ask", methods=["POST"])
def ask():
    global chat_history
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # Append query to history
    chat_history.append({"role": "user", "content": query})

    llm_answer, top_chunks = retrieve_answer(query)

    # Append answer to history
    chat_history.append({"role": "assistant", "content": llm_answer})

    # Construct short-term context string
    short_term_context = ""
    for turn in chat_history[-4:]:  # last 2 user/assistant turns
        role = "User" if turn["role"] == "user" else "Bot"
        short_term_context += f"{role}: {turn['content']}\n"

    return jsonify({
        "query": query,
        "answer": llm_answer,
        "top_chunks": top_chunks
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
