import requests
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


def build_prompt(context, question, chat_context=""):
    prompt = f"""
    তুমি একজন সহায়ক বাংলা সহকারী।

    তোমাকে একটি প্রশ্ন এবং কিছু প্রাসঙ্গিক তথ্য (context) দেওয়া হবে।

    ✅ কেবল context-এ যেটা আছে, সেটাই বলবে। অনুমান বা বাহিরের তথ্য ব্যবহার করা যাবে না।

    ❌ যদি context-এ প্রশ্নের উত্তর না থাকে, তাহলে বলবে: "আমি দুঃখিত, আমি এই প্রশ্নের উত্তর খুঁজে পাইনি।"

    📝 উত্তরটি সংক্ষিপ্ত এবং স্পষ্ট হওয়া উচিত। দয়া করে মাত্র প্রয়োজনীয় তথ্যই দাও।

    ---

    পূর্বের কথোপকথন:
    {chat_context}

    📚 প্রাসঙ্গিক তথ্য:
    {context}

    ❓ প্রশ্ন:
    {question}

    ✅ উত্তর:
    """
    return prompt



def call_ollama_llm(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
        }
    )
    if response.ok:
        return response.json().get("response", "").strip()
    else:
        print(f"LLM HTTP Error {response.status_code}: {response.text}")
        return "Error in LLM CALL"

def load_index_and_chunks(index_path="faiss.index", chunk_path="chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Embed the user query
def embed_query(query, model_name="distiluse-base-multilingual-cased-v2"):
    model = SentenceTransformer(model_name, device="cuda")
    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)  # normalize along axis=1  # Normalize for cosine similarity
    return q_vec

def shorten_answer(answer, max_sentences=2):
    sentences = answer.split('।') 
    short_answer = '।'.join(sentences[:max_sentences]).strip()
    if not short_answer.endswith('।'):
        short_answer += '।'
    return short_answer

def retrieve_answer(query, top_k=20):
    index, chunks = load_index_and_chunks()
    q_vec = embed_query(query)
    D, I = index.search(q_vec, top_k)
    threshold = 0.7
    top_chunks = [chunks[i] for i, score in zip(I[0], D[0]) if score > threshold]

    print("Similarity Scores:", D[0])
    context = "\n\n".join(top_chunks)
    print("🔍 Retrieved Chunks:\n")
    for i, chunk in enumerate(top_chunks):
        print(f"[Chunk {i+1}]\n{chunk}\n")
    prompt = build_prompt(context, query)
    llm_answer = call_ollama_llm(prompt)
    llm_answer_short = shorten_answer(llm_answer, max_sentences=2)

    return llm_answer_short, top_chunks
