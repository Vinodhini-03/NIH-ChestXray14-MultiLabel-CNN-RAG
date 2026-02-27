# ============================================================
# RAG Pipeline for NIH ChestX-ray14 Project
# ============================================================
# RAG = Retrieval Augmented Generation
# Like you're 5: Instead of LLM guessing, we give it a 
# mini-textbook to look up facts before answering.
# ============================================================

# Step 1: Install required libraries
# Run this in terminal first:
# pip install sentence-transformers faiss-cpu openai

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json

# ============================================================
# STEP 1: Load and Chunk the Knowledge Base
# ============================================================
# Chunk = a small piece of text
# We split the knowledge base into chunks so we can
# retrieve only the relevant pieces for each query

def load_and_chunk_knowledge_base(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by double newline — each disease block is one chunk
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# ============================================================
# STEP 2: Create Embeddings and Build Vector Index
# ============================================================
# Embedding = converting text into a list of numbers
# that capture the meaning of the text
# Vector Index = a searchable database of these numbers
# FAISS = Facebook AI Similarity Search (fast vector search library)

def build_vector_index(chunks):
    print("Loading embedding model...")
    
    # SentenceTransformer = model that converts text to embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Creating embeddings for all chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Convert to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')
    
    # Build FAISS index
    # d = dimension of each embedding vector
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)  # L2 = Euclidean distance
    index.add(embeddings)
    
    print(f"Vector index built with {index.ntotal} chunks")
    return index, embeddings, model

# ============================================================
# STEP 3: Retrieve Relevant Chunks
# ============================================================
# Query = the question we want to answer
# We convert the query to an embedding and find the
# most similar chunks in our knowledge base

def retrieve_relevant_chunks(query, index, chunks, model, top_k=3):
    # Convert query to embedding
    query_embedding = model.encode([query]).astype('float32')
    
    # Search for top_k most similar chunks
    # D = distances, I = indices of retrieved chunks
    D, I = index.search(query_embedding, top_k)
    
    retrieved = []
    for idx in I[0]:
        if idx < len(chunks):
            retrieved.append(chunks[idx])
    
    return retrieved

# ============================================================
# STEP 4: Generate LLM Summary with Citations
# ============================================================
# This function takes model predictions and retrieved chunks
# and generates a structured, cited interpretation

def generate_rag_summary(predictions, label_names, retrieved_chunks, threshold=0.5):
    # Find predicted diseases above threshold
    predicted_diseases = []
    for i, (label, prob) in enumerate(zip(label_names, predictions)):
        if prob >= threshold:
            predicted_diseases.append((label, prob))
    
    # Sort by probability descending
    predicted_diseases.sort(key=lambda x: x[1], reverse=True)
    
    # Build the summary
    summary = []
    summary.append("=" * 60)
    summary.append("AI-ASSISTED CHEST X-RAY INTERPRETATION")
    summary.append("=" * 60)
    summary.append("")
    
    # DISCLAIMER — MANDATORY per project requirements
    summary.append("⚠️  DISCLAIMER: This is an AI-assisted analysis tool only.")
    summary.append("This system is NOT a diagnostic tool and must NOT replace")
    summary.append("professional medical judgment. All findings must be reviewed")
    summary.append("by a qualified radiologist before any clinical decision.")
    summary.append("")
    
    if not predicted_diseases:
        summary.append("No findings detected above threshold.")
    else:
        summary.append("DETECTED FINDINGS:")
        summary.append("-" * 40)
        
        for label, prob in predicted_diseases:
            # Uncertainty-aware phrasing based on probability
            if prob >= 0.8:
                confidence = "High confidence"
            elif prob >= 0.65:
                confidence = "Moderate confidence"
            else:
                confidence = "Low confidence"
            
            summary.append(f"\n• {label}: {prob:.1%} ({confidence})")
    
    summary.append("")
    summary.append("KNOWLEDGE BASE CONTEXT:")
    summary.append("-" * 40)
    
    # Add retrieved chunks as citations
    for i, chunk in enumerate(retrieved_chunks, 1):
        summary.append(f"\n[Citation {i}]")
        # Show first 3 lines of each chunk
        chunk_lines = chunk.split('\n')[:3]
        for line in chunk_lines:
            summary.append(f"  {line}")
    
    summary.append("")
    summary.append("=" * 60)
    summary.append("END OF AI-ASSISTED REPORT")
    summary.append("=" * 60)
    
    return '\n'.join(summary)

# ============================================================
# MAIN: Run the RAG Pipeline
# ============================================================
if __name__ == "__main__":
    
    LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    # Load knowledge base
    KB_PATH = os.path.join(os.path.dirname(__file__), 'knowledge_base.txt')
    chunks = load_and_chunk_knowledge_base(KB_PATH)
    
    # Build vector index
    index, embeddings, embed_model = build_vector_index(chunks)
    
    # Example prediction (simulated — replace with real model output)
    example_predictions = [
        0.82,  # Atelectasis
        0.15,  # Cardiomegaly
        0.71,  # Effusion
        0.45,  # Infiltration
        0.08,  # Mass
        0.12,  # Nodule
        0.05,  # Pneumonia
        0.03,  # Pneumothorax
        0.23,  # Consolidation
        0.11,  # Edema
        0.06,  # Emphysema
        0.04,  # Fibrosis
        0.09,  # Pleural_Thickening
        0.02   # Hernia
    ]
    
    # Build query from predicted diseases
    predicted_labels = [LABELS[i] for i, p in enumerate(example_predictions) if p >= 0.5]
    query = f"Information about {', '.join(predicted_labels)} in chest X-ray"
    print(f"\nQuery: {query}")
    
    # Retrieve relevant chunks
    retrieved = retrieve_relevant_chunks(query, index, chunks, embed_model, top_k=3)
    
    # Generate summary
    summary = generate_rag_summary(example_predictions, LABELS, retrieved)
    print(summary)
    
    # Save summary to file
    output_path = os.path.join(os.path.dirname(__file__), 'sample_rag_output.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\nSummary saved to: {output_path}")