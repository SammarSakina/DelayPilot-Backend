import pandas as pd, os
from dotenv import load_dotenv
from sqlalchemy import create_engine
load_dotenv()
url = os.getenv('DATABASE_URL').replace('postgres://', 'postgresql://', 1)
if 'sslmode' not in url:
    url += '?sslmode=require'
engine = create_engine(url.replace('postgresql://', 'postgresql+psycopg2://', 1))

base = pd.read_sql("SELECT COUNT(*) as cnt, MIN(dep_sched_utc) as min_dt, MAX(dep_sched_utc) as max_dt FROM training_flights_base", engine)
bf   = pd.read_sql("SELECT COUNT(*) as cnt, MIN(flight_date) as min_dt, MAX(flight_date) as max_dt FROM training_flights_backfill", engine)
print("training_flights_base:")
print(base.to_string())
print("\ntraining_flights_backfill:")
print(bf.to_string())
print("\nTotal combined rows:", int(base.iloc[0]['cnt']) + int(bf.iloc[0]['cnt']))