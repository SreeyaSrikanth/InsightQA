"""
SQLite DB for InsightQA metadata:
- KnowledgeBase: one row per ingestion
- Document: one row per uploaded file
"""

from __future__ import annotations

import os
import datetime as dt
from typing import Generator

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Boolean,
    DateTime,
    Integer,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# Use a simple SQLite file in the project root by default
DATABASE_URL = os.getenv("INSIGHTQA_DB_URL", "sqlite:///./insightqa.db")

# For SQLite we need this connect_args tweak
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String, primary_key=True, index=True)  # UUID string
    name = Column(String)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(String, index=True)          # FK to KnowledgeBase.id (logical)
    filename = Column(String, index=True)
    path = Column(String)                       # full path on disk (assets/...)
    mime_type = Column(String, nullable=True)
    role = Column(String, default="support")    # "main" or "support"
    is_html = Column(Boolean, default=False)
    is_primary_html = Column(Boolean, default=False)
    uploaded_at = Column(DateTime, default=dt.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
