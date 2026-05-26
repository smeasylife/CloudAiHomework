"""Request and response schemas for the toy API."""

from typing import Any
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    record_id: str
    chunk_count: int
    question_seed_count: int


class StartInterviewRequest(BaseModel):
    record_id: str
    target_university: str = Field(..., examples=["가천대학교"])
    target_department: str = Field(..., examples=["컴퓨터공학과"])
    difficulty: str = Field("Normal", pattern="^(Easy|Normal|Hard)$")


class StartInterviewResponse(BaseModel):
    session_id: str
    first_question: str


class AnswerRequest(BaseModel):
    session_id: str
    answer: str
    response_time: int = Field(30, ge=0, le=600)


class AnswerResponse(BaseModel):
    next_question: str | None
    current_sub_topic: str
    action: str
    remaining_time: int
    is_finished: bool


class ReportResponse(BaseModel):
    session_id: str
    status: str
    report: dict[str, Any]

