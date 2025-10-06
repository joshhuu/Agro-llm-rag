import streamlit as st
import chromadb

# connect to the same folder where embeddings were stored
chroma_client = chromadb.PersistentClient(path="./chroma_store")

st.title("🔍 ChromaDB Query Explorer")

# get list of collections
collections = chroma_client.list_collections()
if not collections:
    st.error("❌ No collections found in ChromaDB. Run your embedding script first.")
else:
    collection_names = [c.name for c in collections]
    collection_name = st.selectbox("Choose a collection", collection_names)
    collection = chroma_client.get_collection(collection_name)

    query_text = st.text_input("Enter your query")
    if query_text:
        results = collection.query(
            query_texts=[query_text],
            n_results=5
        )
        st.write("### Results:")
        for i, doc in enumerate(results["documents"][0]):
            st.write(f"**{i+1}.** {doc}")
