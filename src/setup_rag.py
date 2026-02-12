"""Initialize ChromaDB and load customer histories for RAG retrieval."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "config"))

import pandas as pd
import chromadb

from settings import (
    CUSTOMER_HISTORIES_FILE,
    CHROMA_DIR,
    RAG_COLLECTION_NAME,
    RAG_BATCH_SIZE,
)


def setup_chromadb():
    """Create ChromaDB collection and ingest customer histories."""

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Recreate collection
    try:
        client.delete_collection(RAG_COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=RAG_COLLECTION_NAME,
        metadata={"description": "Customer interaction histories for RAG"},
    )

    # Load histories
    print("Loading customer histories...")
    df = pd.read_csv(CUSTOMER_HISTORIES_FILE)
    print(f"Loaded {len(df)} records.")

    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        documents.append(row["history"])
        metadatas.append({
            "customerID": row["customerID"],
            "tenure": int(row["tenure"]),
            "MonthlyCharges": float(row["MonthlyCharges"]),
            "TotalCharges": float(row["TotalCharges"]),
            "CLV": float(row["CLV"]),
            "Contract": str(row["Contract"]),
        })
        ids.append(row["customerID"])

    # Batch insert
    for i in range(0, len(documents), RAG_BATCH_SIZE):
        end = min(i + RAG_BATCH_SIZE, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"  Batch {i // RAG_BATCH_SIZE + 1}/{(len(documents) - 1) // RAG_BATCH_SIZE + 1}")

    print(f"\nAdded {len(documents)} histories to ChromaDB.")
    print(f"Database location: {CHROMA_DIR}")

    # Quick test
    print("\nTest query: 'billing issues'")
    results = collection.query(query_texts=["billing issues"], n_results=3)
    print(f"Found {len(results['documents'][0])} relevant customers.")

    return collection


if __name__ == "__main__":
    setup_chromadb()
