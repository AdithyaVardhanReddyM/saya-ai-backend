import os
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import LONGTEXT
from tidb_vector.sqlalchemy import VectorType
from dotenv import load_dotenv

load_dotenv()

# TiDB connection configuration
TIDB_HOST = os.getenv("TIDB_HOST")
TIDB_PORT = os.getenv("TIDB_PORT")
TIDB_USER = os.getenv("TIDB_USER")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
TIDB_DB_NAME = os.getenv("TIDB_DB_NAME")
CA_PATH = os.getenv("CA_PATH")

# Create database URL
DATABASE_URL = f"mysql+pymysql://{TIDB_USER}:{TIDB_PASSWORD}@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DB_NAME}?ssl_ca={CA_PATH}&ssl_verify_cert=true&ssl_verify_identity=true"

# Create engine
engine = create_engine(DATABASE_URL, echo=False)  # Set to True for SQL debugging

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class Embeddings(Base):
    __tablename__ = "embeddings"
    
    # Map to existing table schema - don't create, just map
    id = Column(String(36), primary_key=True)
    agentId = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    vectorMetadata = Column(JSON, nullable=True)
    vector = Column(VectorType(1024), nullable=False)  # TiDB vector column with 1024 dimensions
    createdAt = Column(DateTime, nullable=False)
    
    # Don't create table, just map to existing one
    __table_args__ = {'extend_existing': True}

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()