"""Gemini PDF OCR/chunking and Chroma ingestion.

원 프로젝트의 핵심 방식처럼 PDF 내용을 로컬 텍스트 추출기로 읽지 않고,
Gemini에게 PDF 자체를 전달해 OCR, 개인정보 마스킹, 카테고리 청킹을 맡긴다.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from google.genai import types

from app.config import GEMINI_CHAT_MODEL
from app.llm import GeminiClient
from app.rag_store import RagStore


CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["category", "content"],
            },
        }
    },
    "required": ["records"],
}


async def ingest_pdf(
    *,
    pdf_bytes: bytes,
    target_department: str,
    llm: GeminiClient,
    rag_store: RagStore,
) -> tuple[str, int]:
    record_id = f"record-{uuid.uuid4().hex[:10]}"
    chunks = await chunk_student_record_pdf(pdf_bytes, llm)
    chunk_count = await rag_store.add_record_chunks(
        record_id=record_id,
        chunks=chunks,
        target_department=target_department,
        llm=llm,
    )
    return record_id, chunk_count


async def chunk_student_record_pdf(pdf_bytes: bytes, llm: GeminiClient) -> list[dict[str, str]]:
    """Send the original PDF bytes to Gemini for OCR + semantic chunking."""
    prompt = """당신은 학교 생활기록부 전문 분석가입니다.

첨부된 PDF는 학생의 생활기록부입니다. PDF의 텍스트를 직접 읽고, 면접 RAG에 사용할 청크를 JSON으로 만드세요.

청킹 규칙:
1. 개인정보 완전 삭제: 이름 → [이름], 번호 → [번호], 주소 → [주소]
2. 카테고리는 출결, 성적, 동아리, 리더십, 인성/태도, 진로/자율, 독서, 봉사, 세특, 기타 중 하나
3. 보이는 원문을 근거로 작성하고, 없는 내용을 추가/추측/요약하지 말 것
4. 불분명하거나 읽기 어려운 텍스트는 [일부 텍스트 누락]으로 표시하거나 제외
5. 표 데이터의 숫자, 날짜, 점수는 절대 임의 변경하지 말 것
6. 같은 카테고리의 관련 활동은 하나의 content에 묶되, 너무 길면 여러 청크로 분리
7. 각 content는 대략 300~800자
8. JSON 외 텍스트는 출력하지 말 것

출력 형식:
{
  "records": [
    {
      "category": "세특",
      "content": "PDF에 명확히 존재하는 내용"
    }
  ]
}
"""
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    response = await asyncio.wait_for(
        llm.client.aio.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[prompt, pdf_part],
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
                "response_json_schema": CHUNK_SCHEMA,
            },
        ),
        timeout=180,
    )

    try:
        result = json.loads(response.text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini PDF OCR/청킹 응답을 JSON으로 파싱하지 못했습니다.") from exc

    chunks = [
        {
            "category": str(item.get("category") or "기타"),
            "content": str(item.get("content") or "").strip(),
        }
        for item in result.get("records", [])
        if str(item.get("content") or "").strip()
    ]
    if not chunks:
        raise ValueError("Gemini가 PDF에서 생활기록부 청크를 생성하지 못했습니다.")
    return chunks

