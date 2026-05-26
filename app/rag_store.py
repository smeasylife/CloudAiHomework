"""Chroma-backed RAG store for student-record chunks and real interview questions."""

from __future__ import annotations

import json
import uuid
from typing import Any

import chromadb

from app.config import CHROMA_DIR, INTERVIEW_QUESTIONS_FILE
from app.llm import GeminiClient


class RagStore:
    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.record_chunks = self.client.get_or_create_collection("student_record_chunks")
        self.interview_questions = self.client.get_or_create_collection("interview_questions")

    async def seed_interview_questions(self, llm: GeminiClient) -> int:
        if self.interview_questions.count() > 0:
            return self.interview_questions.count()

        rows = json.loads(INTERVIEW_QUESTIONS_FILE.read_text(encoding="utf-8"))
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            search_context = row.get("search_context") or row.get("question") or ""
            question = row.get("question") or ""
            ids.append(f"interview-question-{idx}")
            documents.append(search_context)
            metadatas.append(
                {
                    "university": row.get("university") or "미상",
                    "admission_type": row.get("admission_type") or "미상",
                    "department": row.get("department") or "미상",
                    "category": row.get("category") or "기타",
                    "question": question,
                }
            )

        embeddings = await _embed_in_batches(llm, documents)
        self.interview_questions.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    async def add_record_chunks(
        self,
        *,
        record_id: str,
        chunks: list[dict[str, str]],
        target_department: str,
        llm: GeminiClient,
    ) -> int:
        if not chunks:
            return 0

        try:
            self.record_chunks.delete(where={"record_id": record_id})
        except Exception:
            pass

        documents = [chunk["content"] for chunk in chunks]
        embeddings = await _embed_in_batches(llm, documents)
        ids = [f"{record_id}-{idx}-{uuid.uuid4().hex[:8]}" for idx in range(len(chunks))]
        metadatas = [
            {
                "record_id": record_id,
                "chunk_index": idx,
                "category": chunk.get("category", "기타"),
                "target_department": target_department,
            }
            for idx, chunk in enumerate(chunks)
        ]

        self.record_chunks.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    async def search_record_chunks(
        self,
        *,
        record_id: str,
        query: str,
        llm: GeminiClient,
        limit: int = 4,
    ) -> list[str]:
        query_embedding = await llm.embed_text(query)
        result = self.record_chunks.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"record_id": record_id},
        )
        return _first_documents(result)

    async def search_interview_questions(
        self,
        *,
        query: str,
        llm: GeminiClient,
        limit: int = 8,
    ) -> list[str]:
        query_embedding = await llm.embed_text(query)
        result = self.interview_questions.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )
        metadatas = result.get("metadatas", [[]])[0] or []
        questions = [metadata.get("question", "") for metadata in metadatas]
        return [question for question in questions if question]


async def _embed_in_batches(llm: GeminiClient, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        embeddings.extend(await llm.embed_texts(texts[start : start + batch_size]))
    return embeddings


def _first_documents(result: dict[str, Any]) -> list[str]:
    docs = result.get("documents", [[]])[0] or []
    return [doc for doc in docs if doc]

