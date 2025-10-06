import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus

# --------------------------
# 1. Load DB config from .env
# --------------------------
load_dotenv()

DB_USERNAME = os.getenv("DB_USER")      # Use the correct env var key
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# URL encode password for safety
DB_PASSWORD_ENCODED = quote_plus(DB_PASSWORD)

DATA_DIR = "data/inventories"

# --------------------------
# 2. Utility: Create database if not exists
# --------------------------
def create_database_if_not_exists():
    # Create engine connected to 'postgres' DB without any connection pool
    default_engine = create_engine(
        f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/postgres",
        isolation_level="AUTOCOMMIT"  # Set AUTOCOMMIT at engine level to avoid transactions
    )
    
    with default_engine.connect() as conn:
        # Check if the target DB exists
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
            {"dbname": DB_NAME}
        )
        exists = result.scalar() is not None

        if not exists:
            conn.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
            print(f"Created database '{DB_NAME}'")
        else:
            print(f"Database '{DB_NAME}' already exists.")

# --------------------------
# 3. Connect to target DB
# --------------------------
def get_db_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

# --------------------------
# 4. Inventory parser
# --------------------------
def load_inventory(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_metadata_row(inv):
    row = {}
    row.update(inv.get("dimensions", {}))
    row.update(inv.get("global_attributes", {}))
    return pd.DataFrame([row])

def prepare_variables_df(inv):
    variables = inv.get("variables", {})
    rows = []
    for name, meta in variables.items():
        rows.append({
            "name": name,
            "dtype": meta.get("dtype"),
            "shape": str(meta.get("shape")),
            "attributes": json.dumps(meta.get("attributes")),
            "sample": json.dumps(meta.get("sample"))
        })
    return pd.DataFrame(rows)

# --------------------------
# 5. Ingest a single file
# --------------------------
def ingest_inventory(file_basename, table_prefix, engine):
    print(f"\n📥 Processing: {file_basename}")
    path = os.path.join(DATA_DIR, file_basename)
    inv = load_inventory(path)

    df_meta = prepare_metadata_row(inv)
    df_vars = prepare_variables_df(inv)

    # Insert to tables
    with engine.begin() as conn:
        df_meta.to_sql(f"{table_prefix}_meta", conn, index=False, if_exists="replace")
        df_vars.to_sql(f"{table_prefix}_variables", conn, index=False, if_exists="replace")
        print(f"✅ Inserted into: {table_prefix}_meta and {table_prefix}_variables")

# --------------------------
# 6. Main execution
# --------------------------
def main():
    create_database_if_not_exists()
    engine = get_db_engine()

    # Define files and table prefixes
    datasets = {
        "1900121_meta_inventory.json": "argo_meta",
        "1900121_prof_inventory.json": "argo_prof",
        "1900121_Rtraj_inventory.json": "argo_rtraj",
        "1900121_tech_inventory.json": "argo_tech",
    }

    for file, prefix in datasets.items():
        ingest_inventory(file, prefix, engine)

    print("\n🎉 All datasets inserted successfully.")

if __name__ == "__main__":
    main()
