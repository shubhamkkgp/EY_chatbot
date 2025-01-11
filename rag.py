import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.neighbors import NearestNeighbors  # Use scikit-learn instead of FAISS

# Load the knowledge base JSON file
def load_knowledge_base(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Extract text and metadata
def prepare_data(knowledge_base):
    texts = []
    metadata = []
    for entry in knowledge_base:
        for chunk in entry["content"]:
            texts.append(chunk["text"])
            metadata.append({
                "title": entry["metadata"]["title"],
                "url": entry["metadata"]["url"],
                "type": chunk["type"]
            })
    return texts, metadata

# Generate embeddings and build scikit-learn index
def create_sklearn_index(texts, embedding_model):
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    index = NearestNeighbors(n_neighbors=3, metric="euclidean")
    index.fit(embeddings)  # Fit embeddings
    return index, embeddings

# Retrieve relevant chunks
def retrieve(query, embedding_model, index, embeddings, texts, metadata, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.kneighbors(query_embedding, n_neighbors=top_k)
    results = [
        {
            "text": texts[i],
            "metadata": metadata[i],
            "distance": distances[0][idx]
        }
        for idx, i in enumerate(indices[0])
    ]
    return results

# Generate a response
def generate_response(query, retrieved_results, generator_model, tokenizer):
    context = "\n".join([result["text"] for result in retrieved_results])
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"
    
    # Tokenize and generate response
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator_model.generate(inputs, max_length=200, num_beams=3, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit App
st.title("Knowledge Base Query System")
st.write("Enter your query below to retrieve relevant contexts and generate a response.")

# Load knowledge base and initialize components
file_path = "zerodha_varsity_knowledge_base.json"  
knowledge_base = load_knowledge_base(file_path)
texts, metadata = prepare_data(knowledge_base)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
sklearn_index, embeddings = create_sklearn_index(texts, embedding_model)

generator_model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

query = st.text_input("Query", "")

if query:
    with st.spinner("Retrieving and generating response..."):
        retrieved_results = retrieve(query, embedding_model, sklearn_index, embeddings, texts, metadata, top_k=3)

        st.subheader("Retrieved Results")
        for result in retrieved_results:
            st.write(f"**Text:** {result['text']}")
            st.write(f"**Metadata:** {result['metadata']}")
            st.write("---")

        response = generate_response(query, retrieved_results, generator_model, tokenizer)

        st.subheader("Generated Response")
        st.write(response)
