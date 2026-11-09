import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# --- Configuration ---
DATA_DIR = "data"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3
RELEVANCE_THRESHOLD = 0.3


# --- Document Loading and Chunking ---
def load_documents(data_dir):
    """Load all .txt files from the data directory."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"text": text, "source": filename})
    return documents


def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split documents into overlapping chunks."""
    chunks = []
    for doc in documents:
        text = doc["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "source": doc["source"]})
            start += chunk_size - overlap
    return chunks


# --- Embedding and Indexing ---
def build_index(chunks, model):
    """Generate embeddings and build a FAISS index."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"Indexed {index.ntotal} vectors of dimension {dimension}")
    return index


# --- Retrieval ---
def retrieve(query, model, index, chunks, top_k=TOP_K):
    """Find the most relevant chunks for a query."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "score": float(scores[0][i]),
        })
    return results


# --- Generation ---
def rag_answer(question, model, index, chunks, generator):
    """Retrieve context and generate a grounded answer."""
    results = retrieve(question, model, index, chunks, top_k=TOP_K)

    # Check relevance threshold
    if results[0]["score"] < RELEVANCE_THRESHOLD:
        return (
            "I don't know - my knowledge base doesn't seem to contain "
            "relevant information for this question."
        ), []

    # Build context from retrieved chunks
    context = "\n".join(f"- {r['text']}" for r in results)

    prompt = (
        f"Answer the question based only on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    output = generator(prompt, max_new_tokens=150, do_sample=False, return_full_text=False)
    answer = output[0]["generated_text"].strip()

    return answer, results


# --- Main ---
def main():
    print("Loading documents...")
    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Building FAISS index...")
    index = build_index(chunks, embed_model)

    print("Loading generation model...")
    generator = pipeline("text-generation", model="distilgpt2")

    print("\nRAG Chatbot ready! Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        answer, sources = rag_answer(question, embed_model, index, chunks, generator)

        print(f"\nBot: {answer}")
        if sources:
            print("\nSources:")
            for s in sources:
                print(f"  [{s['source']}] (score: {s['score']:.3f}) {s['text'][:80]}...")
        print()


if __name__ == "__main__":
    main()