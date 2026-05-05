import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'knbs'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

    def get_connection(self):
        try:
            return psycopg2.connect(**self.connection_params)
        except Exception as e:
            print(f"Connection error: {e}")
            return None

    def execute_query(self, query, params=None):
        conn = self.get_connection()
        if not conn:
            return None, "Database connection failed"
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                if query.strip().upper().startswith("SELECT"):
                    return [dict(row) for row in cursor.fetchall()], None
                else:
                    conn.commit()
                    return {"message": "Success"}, None
        except Exception as e:
            return None, str(e)
        finally:
            conn.close()

    def get_table_schema(self):
        query = """
        SELECT t.table_name, c.column_name, c.data_type, c.is_nullable, c.column_default
        FROM information_schema.tables t
        JOIN information_schema.columns c ON t.table_name = c.table_name
        WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name, c.ordinal_position;
        """
        results, error = self.execute_query(query)
        if error:
            return None, error

        schema = {}
        for row in results:
            t = row['table_name']
            if t not in schema:
                schema[t] = []
            schema[t].append({
                'column': row['column_name'],
                'type': row['data_type'],
                'nullable': row['is_nullable'],
                'default': row['column_default']
            })
        return schema, None