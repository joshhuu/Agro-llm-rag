import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load .env file
load_dotenv()

# Grab credentials
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

# Encode password safely if it has @ or .
password_encoded = quote_plus(password)

# Create engine
engine = create_engine(f"postgresql+psycopg2://{user}:{password_encoded}@{host}:{port}/{database}")

# Load Chroma DB
embeddings = OpenAIEmbeddings()  # or your preferred embeddings model
chroma_db = Chroma(persist_directory="chroma_store", embedding_function=embeddings)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("ARGO Ocean Data Conversational Explorer 🌊")

query = st.text_input("Ask a question about ARGO data:")

if query:
    st.info("Processing your query...")

    # -----------------------------
    # 1. Semantic search in Chroma
    # -----------------------------
    results = chroma_db.similarity_search(query, k=1)
    if results:
        best_match = results[0].page_content
    else:
        st.warning("No relevant match found in metadata.")
        best_match = None

    st.write(f"**Best metadata match:** {best_match}")

    # -----------------------------
    # 2. Extract parameters from query or metadata
    # -----------------------------
    year_match = re.search(r"(19|20)\d{2}", query) or re.search(r"Year: (\d{4})", best_match or "")
    year = year_match.group(0) if year_match else None

    params = []
    for p in ["temperature", "salinity"]:
        if p in query.lower() or (best_match and p in best_match.lower()):
            params.append(p)

    if not params:
        params = ["temperature", "salinity"]  # default fallback

    st.write(f"**Year:** {year}, **Parameters:** {params}")

    # -----------------------------
    # 3. Build SQL query dynamically
    # -----------------------------
    if year:
        sql = f"""
            SELECT profile_number, depth_index, {', '.join(params)}, latitude, longitude
            FROM profiles
            WHERE time LIKE '{year}%'
        """
        df = pd.read_sql(sql, engine)
        st.success(f"Fetched {len(df)} rows for visualization.")
    else:
        st.warning("Could not determine year from query.")
        df = pd.DataFrame()

    # -----------------------------
    # 4. Interactive visualization
    # -----------------------------
    if not df.empty:
        fig = go.Figure()
        for param in params:
            for profile in df["profile_number"].unique()[:5]:  # first 5 profiles
                profile_data = df[df["profile_number"] == profile]
                fig.add_trace(
                    go.Scatter(
                        x=profile_data[param],
                        y=profile_data["depth_index"],
                        mode="lines+markers",
                        name=f"{param.capitalize()} - Profile {profile}"
                    )
                )

        fig.update_yaxes(autorange="reversed", title="Depth Index")
        fig.update_xaxes(title="Value")
        fig.update_layout(title=f"ARGO Profiles for {year}", legend_title="Parameter & Profile")
        st.plotly_chart(fig)

    # -----------------------------
    # 5. Optional: nearest floats
    # -----------------------------
    loc_match = re.search(r"(\d+\.?\d*)[NnSs],?\s*(\d+\.?\d*)[EeWw]", query)
    if loc_match:
        lat, lon = float(loc_match.group(1)), float(loc_match.group(2))
        st.write(f"Finding nearest floats to ({lat}, {lon})...")
        df["distance"] = ((df["latitude"] - lat)**2 + (df["longitude"] - lon)**2)**0.5
        nearest = df.sort_values("distance").head(5)
        st.dataframe(nearest[["profile_number", "latitude", "longitude", "distance"]])
