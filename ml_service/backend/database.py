"""Database connection setup."""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use environment variable for DB URL or default to a sensible default (for local testing/fallback)
# In docker-compose, this should be set to postgresql://user:password@db/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/llm_service")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
