import psycopg2

# PostgreSQL Connection Details
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "rutgers1234",
    "host": "resume-screener-db.cd2ysagkm5vw.ap-southeast-2.rds.amazonaws.com",
    "port": "5432",
}

def create_table():
    """Connect to PostgreSQL and create the resumes table."""
    query = """
    CREATE TABLE IF NOT EXISTS resumes (
        id SERIAL PRIMARY KEY,
        filename TEXT,
        text TEXT
    );
    """
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        print(conn)
        cur = conn.cursor()

        # Execute the CREATE TABLE query
        cur.execute(query)
        conn.commit()  # Commit the changes

        # Close the connection
        cur.close()
        conn.close()

        print("Table 'resumes' created successfully (if not exists).")

    except Exception as e:
        print(f"Error: {e}")

# Run the function
if __name__ == "__main__":
    create_table()
