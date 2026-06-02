from __future__ import annotations

import asyncio
import json
from typing import Any

from google import genai
from google.genai import errors, types

from app.config import settings


class GeminiClient:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.google_api_key)
        self.chat_model = settings.gemini_chat_model
        self.fallback_chat_model = "gemini-2.5-flash-lite"

    async def generate_content(
        self,
        contents: Any,
        *,
        config: dict[str, Any],
        timeout: int,
    ) -> Any:
        try:
            return await self._generate_content(
                model=self.chat_model,
                contents=contents,
                config=config,
                timeout=timeout,
            )
        except errors.ServerError:
            return await self._generate_content(
                model=self.fallback_chat_model,
                contents=contents,
                config=config,
                timeout=timeout,
            )

    async def _generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: dict[str, Any],
        timeout: int,
    ) -> Any:
        return await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            ),
            timeout=timeout,
        )

    async def generate_json(
        self,
        contents: Any,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        timeout: int = 60,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": temperature,
            "response_mime_type": "application/json",
        }
        if schema:
            config["response_json_schema"] = schema

        response = await self.generate_content(
            contents,
            config=config,
            timeout=timeout,
        )
        text = (response.text or "").strip()
        return json.loads(text or "{}")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.aio.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return [embedding.values for embedding in response.embeddings]
