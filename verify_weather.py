import pandas as pd, os
from dotenv import load_dotenv
from sqlalchemy import create_engine
load_dotenv()
url = os.getenv('DATABASE_URL').replace('postgres://', 'postgresql://', 1)
if 'sslmode' not in url:
    url += '?sslmode=require'
engine = create_engine(url.replace('postgresql://', 'postgresql+psycopg2://', 1))
r = pd.read_sql("""
    SELECT airport_icao,
           COUNT(*) as rows,
           MIN(hour_utc) as min_dt,
           MAX(hour_utc) as max_dt
    FROM weather_hourly
    GROUP BY airport_icao
    ORDER BY airport_icao
""", engine)
print(r.to_string())
print("Total rows:", r['rows'].sum())