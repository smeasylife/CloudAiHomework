from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR
from app.graph import QuestionGenerationGraph
from app.ingestion import ingest_pdf
from app.llm import GeminiClient
from app.rag_store import RagStore
from app.schemas import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    IngestResponse,
    RecordStatusResponse,
)


llm = GeminiClient()
rag_store: RagStore | None = None
question_graph: QuestionGenerationGraph | None = None
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global rag_store, question_graph

    rag_store = RagStore()
    question_graph = QuestionGenerationGraph(llm=llm, rag_store=rag_store)
    try:
        yield
    finally:
        if question_graph:
            await question_graph.close()
        if rag_store:
            rag_store.close()


app = FastAPI(title="LangGraph RAG Question Generator Toy", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/records/status", response_model=RecordStatusResponse)
async def record_status():
    return RecordStatusResponse(has_record=get_rag_store().has_record())


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    pdf: UploadFile = File(...),
):
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    try:
        store = get_rag_store()
        question_seed_count = await store.seed_interview_questions(llm)
        pdf_bytes = await pdf.read()
        chunk_count = await ingest_pdf(
            pdf_bytes=pdf_bytes,
            llm=llm,
            rag_store=store,
        )
        return IngestResponse(
            chunk_count=chunk_count,
            question_seed_count=question_seed_count,
        )
    except Exception as exc:
        message = str(exc) or repr(exc)
        logger.error("PDF ingest failed: %s: %s", type(exc).__name__, message)
        raise HTTPException(status_code=500, detail=message) from exc


@app.post("/api/questions/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    graph = get_question_graph()
    try:
        result = await graph.run(
            target_school=request.target_school,
            target_major=request.target_major,
            interview_type=request.interview_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateQuestionsResponse(
        questions=result.get("approved_questions", []),
    )


def get_rag_store() -> RagStore:
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG 저장소가 아직 준비되지 않았습니다.")
    return rag_store


def get_question_graph() -> QuestionGenerationGraph:
    if question_graph is None:
        raise HTTPException(status_code=503, detail="LangGraph가 아직 준비되지 않았습니다.")
    return question_graph
