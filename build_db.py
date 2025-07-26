import fitz 
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from bltk.langtools import Tokenizer
import unicodedata
import re
import pdfplumber
#### INIT BLTK COMPONENTS ####
tokenizer = Tokenizer()

#Extracting text from PDF using pdfplumber

def clean_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = text.replace('✓', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_pdfplumber(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                cleaned_text = clean_text(text)
                full_text += cleaned_text + "\n\n"
            else:
                print(f"Error here: {page_num}")
    return full_text

def split_text_to_chunks(text, window_size=2, stride=2):
    sentences = tokenizer.sentence_tokenizer(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    chunks = []
    for i in range(0, len(sentences) - window_size + 1, stride):
        chunk = " ".join(sentences[i:i + window_size]).strip()
        if len(chunk) > 20:
            chunks.append(chunk)
    return chunks


#Embeding to chunks using SentenceTransformer with GPU/CUDA ===
def embed_chunks(chunks, model_name="distiluse-base-multilingual-cased-v2"):
    model = SentenceTransformer(model_name, device="cuda")  # Use GPU
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return normalized_embeddings

#Building FAISS index with cosine similarity
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index
def save_index(index, index_path):
    faiss.write_index(index, index_path)

def save_chunks(chunks, chunk_path):
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)


if __name__ == "__main__":
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    text = extract_text_pdfplumber(pdf_path)
    cleaned_text = clean_text(text)
    chunks = split_text_to_chunks(cleaned_text)

    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)

    save_index(index, "faiss.index")
    save_chunks(chunks, "chunks.pkl")

    print("Done")