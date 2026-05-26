"""Small Gemini wrapper used by ingestion, RAG and LangGraph nodes."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from google import genai
from google.genai import types

from app.config import GEMINI_CHAT_MODEL, GEMINI_EMBEDDING_MODEL, GOOGLE_API_KEY


class GeminiClient:
    def __init__(self) -> None:
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY가 없습니다. langgraph_rag_interview_toy/.env.example를 참고해 "
                "환경 변수를 설정한 뒤 다시 실행하세요."
            )
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    async def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.4,
        timeout: int = 60,
    ) -> str:
        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=prompt,
                config={"temperature": temperature},
            ),
            timeout=timeout,
        )
        return (response.text or "").strip()

    async def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": 0.2,
            "response_mime_type": "application/json",
        }
        if schema:
            config["response_json_schema"] = schema

        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=prompt,
                config=config,
            ),
            timeout=timeout,
        )
        text = (response.text or "").strip()
        return parse_json_object(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.aio.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return [embedding.values for embedding in response.embeddings]

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse JSON even when a model accidentally wraps it in fences."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)

