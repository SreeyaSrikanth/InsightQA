from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import uuid
import mimetypes
import shutil

from . import parsers
from .vectordb import add_chunks, _get_collection
from .rag_agent import generate_testcases_rag
from .selenium_generator import generate_selenium_script
from .db import get_db, KnowledgeBase, Document

from sqlalchemy.orm import Session

app = FastAPI(title="InsightQA Backend")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSETS_DIR = Path("assets")

class TestCaseRequest(BaseModel):
    kb_id: str           # which knowledge base to use
    query: str
    top_k: int = 5


class ScriptRequest(BaseModel):
    kb_id: str           # which knowledge base
    testcase: Dict[str, Any]
    html_filename: str   # which HTML/UI file within that KB


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Ingestion ----------
@app.post("/ingest")
@app.post("/ingest")
async def ingest(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Generic ingestion:
      - Create a new KnowledgeBase (one per 'Build Knowledge Base' action)
      - Save ALL uploaded files to assets/<kb_id>/<filename>
      - Mark the first HTML as 'main'
      - All files are chunked and stored in the vector DB with kb_id + role
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    kb_id = str(uuid.uuid4())
    kb = KnowledgeBase(id=kb_id, name=name)
    db.add(kb)
    db.commit()

    docs_chunks: List[str] = []
    docs_meta: List[Dict[str, Any]] = []
    ids: List[str] = []
    ingested_docs: List[Dict[str, Any]] = []

    kb_dir = ASSETS_DIR / kb_id
    kb_dir.mkdir(parents=True, exist_ok=True)

    primary_html_set = False

    for f in files:
        raw = await f.read()
        filename = f.filename
        suffix = Path(filename).suffix.lower()
        mime_type, _ = mimetypes.guess_type(filename)

        is_html = suffix in [".html", ".htm"]
        role = "main" if is_html and not primary_html_set else "support"
        is_primary_html = is_html and not primary_html_set

        # 1) Save to assets/<kb_id>/<filename>
        save_path = kb_dir / filename
        save_path.write_bytes(raw)

        # 2) Create Document row
        doc_row = Document(
            kb_id=kb_id,
            filename=filename,
            path=str(save_path),
            mime_type=mime_type or "application/octet-stream",
            role=role,
            is_html=is_html,
            is_primary_html=is_primary_html,
        )
        db.add(doc_row)

        if is_primary_html:
            primary_html_set = True

        # 3) Parse for knowledge base
        text = parsers.parse_any(filename, raw)

        # naive chunking
        CHUNK_SIZE = 800
        OVERLAP = 150
        start_idx = 0
        while start_idx < len(text):
            chunk = text[start_idx : start_idx + CHUNK_SIZE]
            start_idx += CHUNK_SIZE - OVERLAP
            chunk_id = str(uuid.uuid4())
            docs_chunks.append(chunk)
            docs_meta.append(
                {
                    "source_document": filename,
                    "chunk_index": len(docs_chunks) - 1,
                    "kb_id": kb_id,
                    "doc_role": role,
                }
            )
            ids.append(chunk_id)

        ingested_docs.append(
            {
                "filename": filename,
                "role": role,
                "is_html": is_html,
                "is_primary_html": is_primary_html,
                "path": str(save_path),
            }
        )

    db.commit()

    # 4) Add to vector DB
    add_chunks(docs_chunks, docs_meta, ids)

    return {
        "status": "ok",
        "kb_id": kb_id,
        "name": name,
        "chunks_indexed": len(docs_chunks),
        "documents": ingested_docs,
    }


# ---------- RAG test case generation ----------
@app.post("/agent/testcases")
def generate_testcases(req: TestCaseRequest):
    """
    Generate test cases using only chunks from the selected KB.
    Here we filter to doc_role='support' so specs/docs drive the tests.
    If you also want HTML text in context, use doc_roles=["support","main"].
    """
    out = generate_testcases_rag(
        user_query=req.query,
        top_k=req.top_k,
        kb_id=req.kb_id,
        doc_roles=["main","support"]
    )
    return out


# ---------- Selenium script generation ----------
@app.post("/agent/generate_script")
def generate_script(req: ScriptRequest, db: Session = Depends(get_db)):
    # Look up the document to ensure it belongs to this kb_id
    doc = (
        db.query(Document)
        .filter(
            Document.kb_id == req.kb_id,
            Document.filename == req.html_filename,
            Document.is_html == True,
        )
        .first()
    )

    if not doc:
        raise HTTPException(
            status_code=400,
            detail=f"HTML file '{req.html_filename}' not found for kb_id='{req.kb_id}'. "
                   "Make sure you uploaded it in this knowledge base.",
        )

    script = generate_selenium_script(req.testcase, doc.path)
    return {"script": script}

@app.get("/kb/list")
def list_kbs(db: Session = Depends(get_db)):
    kbs = (
        db.query(KnowledgeBase)
        .order_by(KnowledgeBase.created_at.desc())
        .all()
    )
    out = []
    for kb in kbs:
        docs = (
            db.query(Document)
            .filter(Document.kb_id == kb.id)
            .order_by(Document.id.asc())
            .all()
        )
        out.append(
            {
                "kb_id": kb.id,
                "kb_name": kb.name,              # <-- ADD THIS
                "created_at": kb.created_at.isoformat(),
                "documents": [
                    {
                        "filename": d.filename,
                        "role": d.role,
                        "is_html": d.is_html,
                        "is_primary_html": d.is_primary_html,
                        "path": d.path,
                    }
                    for d in docs
                ],
            }
        )
    return out

@app.post("/kb/rename")
def rename_kb(kb_id: str = Form(...), new_name: str = Form(...), db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="KB not found")
    kb.name = new_name
    db.commit()
    return {"status": "ok", "kb_id": kb_id, "new_name": new_name}

@app.post("/kb/delete")
def delete_kb(kb_id: str = Form(...), db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="KB not found")

    # Delete documents
    docs = db.query(Document).filter(Document.kb_id == kb_id).all()
    for doc in docs:
        db.delete(doc)

    # Delete KB row
    db.delete(kb)
    db.commit()

    # Delete vector DB entries
    coll = _get_collection()
    coll.delete(where={"kb_id": kb_id})

    # Delete file directory
    kb_dir = Path(f"assets/{kb_id}")
    shutil.rmtree(kb_dir, ignore_errors=True)

    return {"status": "ok", "deleted_kb_id": kb_id}

@app.get("/kb/view/{kb_id}")
def view_kb(kb_id: str, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="KB not found")

    docs = db.query(Document).filter(Document.kb_id == kb_id).all()

    return {
        "kb_name": kb.name,
        "documents": [
            {
                "filename": d.filename,
                "role": d.role,
            }
            for d in docs
        ]
    }
