from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    chunk_count: int
    question_seed_count: int


class RecordStatusResponse(BaseModel):
    has_record: bool


class GenerateQuestionsRequest(BaseModel):
    target_school: str = Field(..., examples=["한양대학교"])
    target_major: str = Field(..., examples=["컴퓨터공학과"])
    interview_type: str = Field("학생부종합")


class GeneratedQuestionResponse(BaseModel):
    category: str
    content: str
    difficulty: str
    evaluation: dict[str, Any] | None = None


class GenerateQuestionsResponse(BaseModel):
    questions: list[GeneratedQuestionResponse]
