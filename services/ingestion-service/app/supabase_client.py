import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPOSITORY_ROOT / ".env"

load_dotenv(ENV_FILE)


def get_supabase_client() -> Client:
    """Create a Supabase client using local environment configuration."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not key:
        raise RuntimeError(
            "Supabase configuration is missing. "
            "Set SUPABASE_URL and SUPABASE_KEY in the repository .env file."
        )

    return create_client(url, key)
