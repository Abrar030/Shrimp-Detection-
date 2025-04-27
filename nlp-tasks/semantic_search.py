import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

df = pd.read_csv("data/faq.csv")  
questions = df["question"].tolist()
answers = df["answer"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(questions)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def search(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        results.append({"question": questions[idx], "answer": answers[idx]})
    return results


if __name__ == "__main__":
    query = "How to reset my password?"
    results = search(query)
    print(f"Top results for '{query}':")
    for res in results:
        print(f"Q: {res['question']}\nA: {res['answer']}\n")