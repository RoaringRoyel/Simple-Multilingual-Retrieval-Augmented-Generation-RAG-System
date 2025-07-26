# 📚 Multilingual RAG System (Bangla + English) — AI Engineer Assessment

This is a simple Retrieval-Augmented Generation (RAG) system that supports both Bangla and English queries over a textbook-based knowledge base. It can answer fact-based questions by retrieving relevant context from a Bangla PDF and generating grounded responses using a local LLM via Ollama.

 ## 📁 Project Structure project-root/ 
 ├── templates/
 
 │ └── index.html # Simple UI form to send queries 
 
 ├── static/ 
 
 ├── app.py # Flask API + route for UI 
 
 ├── build_db.py # PDF processing & FAISS index creation 
 
 ├── faiss_index/
 
 ├── index.faiss # FAISS vector index 
 
 │── chunks.pkl # Stored chunks 
 
 │ └── style.css # Styling for the frontend 
 
 ├── HSC26-Bangla1st-Paper.pdf # Bangla textbook (dataset) 
 
 ├── requirements.txt # Python dependencies 
 
 └── README.md # Project guide and reflections

## 🔧 Features

- ✅ Accepts user queries in both **Bangla** and **English**
- 🔍 Retrieves relevant document chunks from **HSC26 Bangla 1st Paper**
- 🤖 Generates answers using **Ollama** (Mistral model)
- 🧠 Uses **FAISS** for vector similarity search
- 🌐 Lightweight **Flask API** for interaction

---
🧪 Technologies Used
Python

Flask

FAISS

Sentence-Transformers

PyMuPDF

Ollama (Mistral model)

Bangla Text (Unicode) Processing
---

## 📡 API Documentation

This project includes a simple RESTful API built with Flask that exposes one main endpoint:

---

### 🔹 POST /query

Retrieves relevant context from the Bangla textbook and generates a grounded answer using a local LLM (via Ollama).

#### 📥 Request Body (JSON)

```json
{
  "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
}
Response Body (JSON)
{
  "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
  "answer": "মামাকে",
  "context": "অনুপমের জীবন-সংগ্রামে মামার অবদান ছিল গুরুত্বপূর্ণ। ... (retrieved chunk)"
}
```
###
📄 Questions & Reflections
1. 📌 What method/library did you use to extract text and why?
I used PyMuPDF (fitz) as it accurately extracts Bangla Unicode text and handles page layouts well. PDFPlumber had layout issues in this case.


2. ✂️ What chunking strategy did you choose?
I used Bangla sentence-based chunking using punctuation (।) and grouped 2–3 sentences (up to ~400 chars). This balances context without losing precision. The pdf have issues with containing context as the MCQ question , option and results are distant from each other. It is very hard to bring them under a same context chunk. Various window_size and sliding windows had been tried in order to try and error. Hence window_size = 5 and Strid = 2 works better than other tunings.
A custom data_clean function had been used to handle the bangla punctulation marks along with the bltk updated library. [ Try with ntlk and more customized function has enhance the pre processing performance. 

3. 🔠 What embedding model did you use?
I used paraphrase-multilingual-MiniLM-L12-v2 from sentence-transformers because it supports Bangla + English and performs well on semantic similarity tasks with low latency. For this task I have used CUDA (RTX 4070 ti super ) and the code is for cuda as well. 

4. 📐 How are you comparing the query with stored chunks?
Embeddings are normalized and compared using cosine similarity via FAISS.IndexFlatIP. This ensures fast top-k retrieval. 

5. 🤝 How do you ensure meaningful comparison of queries and chunks?
Both queries and chunks are embedded in the same semantic space, and I avoid over-filtering short answers during chunking. Queries are taken from the UI and send to rag_engine and the function retrieve_answer is triggered. Maintaining a threshhold of 0.7 for cosine similarity to take only the chunks are related to the context. Also in the terminal, the similirity scores are printed to ease the hypertuning for future work. Also, to check relavnace of the selected chunks in terminal they are printed. 

6. 📈 Do the results seem relevant? What would improve them?
Mostly No. Cheching the logs the similarity are around 60-90 for most of the time. But sometimes it go toward 40% as well. Hypertuning threshhold and sliding window, context_size with top_k [ not too much, as too much will create extra context ]. Need more tuning for the task. Also, often the RAG shows hallucination.

Trying llama2:7b-chat increases the hallucination
Trying mistral:latest makes it more relavent. 
more try on ollama or GPT model needs to be tried here for fine-tuning.

The data pre-processing issue is already mentioned avobe.
Also when embedding extra tag can help.


Phrase-level chunking in dense parts
## 📦 Setup Instructions

### 1. 📁 Clone & Install Dependencies
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt

### 2. 🧱 Build Vector Index
Change to CPU if you are not using a GPU
```bash
python build_db.py

### 3. Run Ollama Model
Make sure you have downloaded mistral:latest    6577803aa9a0    4.4 GB
```bash
ollama run mistral

### 4. 🚀 Launch Flask App
```bash
python app.py


