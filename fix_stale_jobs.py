import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
url = os.getenv('DATABASE_URL')
if 'sslmode' not in url:
    url += '?sslmode=require'
conn = psycopg2.connect(url)
conn.autocommit = True
with conn.cursor() as cur:
    # Mark all stale running/queued jobs as cancelled
    # EXCEPT jobs that already have outcome = rejected
    # and EXCEPT the highest job id (most recent)
    cur.execute("""
        UPDATE retrain_jobs
        SET status        = 'failed',
            outcome       = 'cancelled',
            error_message = 'Job interrupted — server was stopped before completion.',
            finished_at   = NOW()
        WHERE status IN ('running', 'queued')
          AND outcome IS DISTINCT FROM 'rejected'
          AND id < (SELECT MAX(id) FROM retrain_jobs)
    """)
    print('Fixed rows:', cur.rowcount)

    # Show current state
    cur.execute("""
        SELECT id, status, outcome
        FROM retrain_jobs
        ORDER BY id
    """)
    for row in cur.fetchall():
        print(f"  Job #{row[0]}: status={row[1]} outcome={row[2]}")
conn.close()
print('Done.')