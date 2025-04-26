###1)
### Extract text from PDF
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
import torch
import numpy as np
import PyPDF2
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2025-01-01-preview"
)

from pathlib import Path

def find_pdf_on_desktop(filename="EJ333.pdf"):
    """Searches for the PDF file in common desktop locations (EN/中文 systems)."""
    candidate_paths = [
        Path.home() / "Desktop" / filename,
        Path.home() / "OneDrive" / "Desktop" / filename,
        Path.home() / "OneDrive" / "桌面" / filename,
        Path.home() / "桌面" / filename
    ]
    for path in candidate_paths:
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Could not locate {filename} on known desktop paths.")
###

class PDFParser:
    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
def main_1():
    parser = PDFParser()
    pdf_path = find_pdf_on_desktop("EJ333.pdf")
    text = parser.extract_text_from_pdf(pdf_path)
    print(text)
###

###2)
### Embed sample sentence
def main_2():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    print("Embedding shape:", embedding.shape)
###

###3)
### Create FAISS index and test search logic
def encode_text_to_vector(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def create_faiss_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype=np.float32))
    return index

def main_3():
    samples = ["This is the first item.", "This is the second one.", "Another unrelated sentence."]
    vectors = np.vstack([encode_text_to_vector(p) for p in samples])
    index = create_faiss_index(vectors)

    query = "first item"
    query_vec = encode_text_to_vector(query)
    _, top_indices = index.search(query_vec, k=2)
    print("Top 2 neighbors:", top_indices)

###

###4）
### Full pipeline - PDF parse + vector search + LLM answer
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder="./models",
            use_auth_token=False
        )

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

class VectorDatabase:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.paragraphs = []

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query_vector, top_k=3):
        query_vector = np.array(query_vector).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices


import requests

def generate_answer_with_azure(client, deployment, context, question):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an academic writing assistant."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        }
    ]
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=800,
        temperature=0.7
    )
    return completion.choices[0].message.content.strip()


import re
def split_into_paragraphs(text):
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    paragraphs = []
    temp = []
    for i, sentence in enumerate(sentences):
        temp.append(sentence)
        if len(temp) >= 3:
            paragraphs.append(" ".join(temp))
            temp = []
    if temp:
        paragraphs.append(" ".join(temp))
    return paragraphs


import os

api_key = os.getenv("OPENROUTER_API_KEY")

def main_4():
    parser = PDFParser()
    pdf_path = find_pdf_on_desktop("EJ333.pdf")
    text = parser.extract_text_from_pdf(pdf_path)
    paragraphs = split_into_paragraphs(text)

    embedder = Embedder()
    para_vecs = np.vstack([embedder.get_embedding(p) for p in paragraphs])

    db = VectorDatabase(para_vecs.shape[1])
    db.paragraphs = paragraphs
    db.add(para_vecs)

    MAX_CALLS = 100
    MAX_TOKENS = 80000
    call_count = 0
    total_tokens_used = 0

    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower().strip() == 'exit':
            print("Exiting interactive mode.")
            break

        call_count += 1
        if call_count > MAX_CALLS:
            print("Reached maximum question limit (100). Exiting.")
            break

        query_vec = embedder.get_embedding(query)
        _, indices = db.search(query_vec, top_k=5)
        top_k_paragraphs = [paragraphs[idx] for idx in indices[0] if idx != -1]
        context = '\n\n'.join(top_k_paragraphs)
        context = context[:1600]

        print(f"\nTop 3 matched paragraphs:\n{context}\n")

        answer = generate_answer_with_azure(client, deployment, context, query)



        if total_tokens_used > MAX_TOKENS:
            print("Reached maximum token usage (80,000 tokens). Exiting.")
            break

        print(f"\nLLM Answer:\n{answer}\n")

        print("\nSuggested test questions:")
        print(f"1. How does this relate to independent learning using mobile devices?")
        print(f"2. What are the practical benefits of smartphones in vocabulary learning?")


###

### Run all parts
if __name__ == "__main__":
    main_1()
    main_2()
    main_3()
    main_4()
###
