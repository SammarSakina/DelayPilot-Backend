import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# Read weather from local PostgreSQL
local_url = (
    f"postgresql+psycopg2://"
    f"{os.getenv('PG_USER','postgres')}:{os.getenv('PG_PASSWORD','delaypilot2026')}"
    f"@{os.getenv('PG_HOST','localhost')}:{os.getenv('PG_PORT','5432')}"
    f"/{os.getenv('PG_DB','delaypilot_db')}"
)
local_engine = create_engine(local_url)
df = pd.read_sql("SELECT * FROM weather_hourly", local_engine)
print(f"Read {len(df)} rows from local weather_hourly")
print(f"Columns: {df.columns.tolist()}")

# Write to Supabase
supa_url = os.getenv("DATABASE_URL")
if supa_url.startswith("postgres://"):
    supa_url = supa_url.replace("postgres://", "postgresql://", 1)
if "sslmode" not in supa_url:
    supa_url += "?sslmode=require"
supa_url_sa = supa_url.replace("postgresql://", "postgresql+psycopg2://", 1)
supa_engine = create_engine(supa_url_sa)
df.to_sql("weather_hourly", supa_engine, if_exists="replace", index=False)
print(f"Written {len(df)} rows to Supabase weather_hourly")
print("Done.")