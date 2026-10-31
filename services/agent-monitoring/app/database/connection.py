import os

import psycopg


def get_database_connection():
    """Create a PostgreSQL connection for the IntelliCore event store."""

    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "intellicore"),
        user=os.getenv("POSTGRES_USER", "intellicore"),
        password=os.getenv("POSTGRES_PASSWORD", "intellicore_dev"),
    )