from __future__ import annotations

import asyncio

import fitz
from google.genai import types

from app.llm import GeminiClient
from app.rag_store import RagStore


PDF_PAGES_PER_REQUEST = 3

CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
        }
    },
    "required": ["records"],
}


async def ingest_pdf(
    *,
    pdf_bytes: bytes,
    llm: GeminiClient,
    rag_store: RagStore,
) -> int:
    chunks = await chunk_student_record_pdf(pdf_bytes, llm)
    return await rag_store.add_record_chunks(chunks=chunks, llm=llm)


async def chunk_student_record_pdf(pdf_bytes: bytes, llm: GeminiClient) -> list[dict[str, str]]:
    pdf_parts = split_pdf(pdf_bytes, pages_per_part=PDF_PAGES_PER_REQUEST)
    part_chunks = await asyncio.gather(
        *(_chunk_student_record_pdf_part(part, llm) for part in pdf_parts)
    )
    return [chunk for chunks in part_chunks for chunk in chunks]


async def _chunk_student_record_pdf_part(pdf_bytes: bytes, llm: GeminiClient) -> list[dict[str, str]]:
    prompt = """당신은 학교 생활기록부 전문 분석가입니다.

첨부된 PDF는 학생의 생활기록부입니다. PDF의 텍스트를 직접 읽고, 면접 RAG에 사용할 청크를 JSON으로 만드세요.

청킹 규칙:
1. 개인정보 완전 삭제: 이름 → [이름], 번호 → [번호], 주소 → [주소]
2. 보이는 원문을 그대로 작성하고, 없는 내용을 추가/추측/요약하지 말 것
3. 불분명하거나 읽기 어려운 텍스트는 [일부 텍스트 누락]으로 표시하거나 제외
4. 표 데이터의 숫자, 날짜, 점수는 절대 임의 변경하지 말 것
5. 관련 활동은 하나의 content에 묶되, 너무 길면 여러 청크로 분리
6. 각 content는 대략 300~800자
7. JSON 외 텍스트는 출력하지 말 것

출력 형식:
{
  "records": [
    {
      "content": "PDF에 명확히 존재하는 내용"
    }
  ]
}
"""
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    result = await llm.generate_json(
        [prompt, pdf_part],
        schema=CHUNK_SCHEMA,
        temperature=0.1,
        timeout=180,
    )

    chunks = [
        {
            "content": str(item.get("content") or "").strip(),
        }
        for item in result.get("records", [])
        if str(item.get("content") or "").strip()
    ]
    return chunks


def split_pdf(pdf_bytes: bytes, *, pages_per_part: int) -> list[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: list[bytes] = []

    try:
        total_pages = len(doc)
        for start in range(0, total_pages, pages_per_part):
            part = fitz.open()
            try:
                part.insert_pdf(
                    doc,
                    from_page=start,
                    to_page=min(start + pages_per_part, total_pages) - 1,
                )
                parts.append(part.tobytes())
            finally:
                part.close()
    finally:
        doc.close()

    return parts
