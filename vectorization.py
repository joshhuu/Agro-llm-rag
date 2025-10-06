# vectorization_with_progress.py

import os
import pandas as pd
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from urllib.parse import quote_plus
from tqdm import tqdm  # progress bar

# --------------------------
# 1. Load DB config
# --------------------------
load_dotenv()

DB_USERNAME = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DB_PASSWORD_ENCODED = quote_plus(DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# --------------------------
# 2. Setup Chroma
# --------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_store")

# Use SentenceTransformer for CPU embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --------------------------
# 3. Utility: process one table
# --------------------------
def process_table(table_name, id_prefix):
    print(f"\n Processing table: {table_name}")
    with engine.connect() as conn:
        df = pd.read_sql_table(table_name, conn)

    if df.empty:
        print(" Table is empty, skipping.")
        return

    # Convert entire row into a string for embedding
    df["text"] = df.apply(lambda row: " | ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
    df["doc_id"] = df.index.astype(str).map(lambda i: f"{id_prefix}_{i}")

    # Create/reuse collection
    collection = chroma_client.get_or_create_collection(name=table_name)

    # Find already stored IDs
    existing = set(collection.get()["ids"])
    to_add = df[~df["doc_id"].isin(existing)]

    if to_add.empty:
        print(f" All {len(df)} rows already embedded.")
        return

    print(f"➡️ Embedding {len(to_add)} new rows out of {len(df)} total")

    # Batch process with progress bar
    batch_size = 100
    total_batches = (len(to_add) // batch_size) + (1 if len(to_add) % batch_size else 0)

    for i in range(total_batches):
        batch = to_add.iloc[i * batch_size : (i + 1) * batch_size]
        if batch.empty:
            continue

        # Show progress
        print(f"   Batch {i+1}/{total_batches} | Rows {i*batch_size}-{i*batch_size+len(batch)-1}")

        embeddings = embedder.encode(batch["text"].tolist(), show_progress_bar=False)
        collection.add(
            ids=batch["doc_id"].tolist(),
            documents=batch["text"].tolist(),
            embeddings=embeddings
        )

    print(f"🎉 Completed: {table_name} (now {collection.count()} docs stored)")

# --------------------------
# 4. Main
# --------------------------
def main():
    tables = ["argo_meta_meta", "argo_meta_variables",
              "argo_prof_meta", "argo_prof_variables",
              "argo_rtraj_meta", "argo_rtraj_variables",
              "argo_tech_meta", "argo_tech_variables"]

    for t in tables:
        process_table(t, id_prefix=t)

if __name__ == "__main__":
    main()
