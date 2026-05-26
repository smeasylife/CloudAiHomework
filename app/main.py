"""FastAPI entrypoint for the local LangGraph RAG interview toy app."""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR
from app.graph import InterviewGraph
from app.ingestion import ingest_pdf
from app.llm import GeminiClient
from app.rag_store import RagStore
from app.schemas import (
    AnswerRequest,
    AnswerResponse,
    IngestResponse,
    ReportResponse,
    StartInterviewRequest,
    StartInterviewResponse,
)
from app.session_store import FIRST_QUESTION, SessionStore


app = FastAPI(title="LangGraph RAG Interview Toy", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

llm = GeminiClient()
rag_store = RagStore()
session_store = SessionStore()
interview_graph = InterviewGraph(llm=llm, rag_store=rag_store)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    pdf: UploadFile = File(...),
    target_department: str = Form(...),
):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    try:
        question_seed_count = await rag_store.seed_interview_questions(llm)
        pdf_bytes = await pdf.read()
        record_id, chunk_count = await ingest_pdf(
            pdf_bytes=pdf_bytes,
            target_department=target_department,
            llm=llm,
            rag_store=rag_store,
        )
        return IngestResponse(
            record_id=record_id,
            chunk_count=chunk_count,
            question_seed_count=question_seed_count,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/interview/start", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest):
    session = session_store.create_session(
        record_id=request.record_id,
        target_university=request.target_university,
        target_department=request.target_department,
        difficulty=request.difficulty,
    )
    return StartInterviewResponse(session_id=session["session_id"], first_question=FIRST_QUESTION)


@app.post("/api/interview/answer", response_model=AnswerResponse)
async def answer_interview(request: AnswerRequest):
    try:
        session = session_store.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if session.get("status") == "COMPLETED":
        return AnswerResponse(
            next_question=None,
            current_sub_topic=session.get("current_sub_topic", ""),
            action="wrap_up",
            remaining_time=session.get("remaining_time", 0),
            is_finished=True,
        )

    result = await interview_graph.run(
        session=session,
        answer=request.answer,
        response_time=request.response_time,
    )
    update_session_from_graph(session, result, request.answer, request.response_time)
    session_store.update(session)

    return AnswerResponse(
        next_question=result.get("next_question"),
        current_sub_topic=session.get("current_sub_topic", ""),
        action=result.get("action", "new_topic"),
        remaining_time=session.get("remaining_time", 0),
        is_finished=session.get("status") == "COMPLETED",
    )


@app.get("/api/interview/{session_id}/report", response_model=ReportResponse)
async def report(session_id: str):
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    report_data = session.get("final_report") or {
        "message": "아직 면접이 종료되지 않았습니다.",
        "interview_logs": session.get("interview_logs", []),
    }
    return ReportResponse(
        session_id=session_id,
        status=session.get("status", "IN_PROGRESS"),
        report=report_data,
    )


def update_session_from_graph(
    session: dict,
    result: dict,
    answer: str,
    response_time: int,
) -> None:
    logs = session.get("interview_logs", [])
    if logs:
        logs[-1]["answer"] = answer
        logs[-1]["response_time"] = response_time
        logs[-1]["timestamp"] = datetime.now().isoformat(timespec="seconds")

    session["remaining_time"] = result.get("remaining_time", session.get("remaining_time", 600))
    session["current_sub_topic"] = result.get("current_sub_topic", session.get("current_sub_topic", ""))
    session["asked_sub_topics"] = result.get("asked_sub_topics", session.get("asked_sub_topics", []))
    session["follow_up_count"] = result.get("follow_up_count", session.get("follow_up_count", 0))
    session["question_count"] = result.get("question_count", session.get("question_count", len(logs)))

    if result.get("action") == "wrap_up":
        session["status"] = "COMPLETED"
        session["final_report"] = result.get("final_report", {})
        return

    next_question = result.get("next_question")
    if next_question:
        logs.append(
            {
                "question": next_question,
                "answer": "",
                "response_time": 0,
                "sub_topic": session.get("current_sub_topic", ""),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
    session["interview_logs"] = logs

