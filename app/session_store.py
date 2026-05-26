"""Tiny JSON-backed session store for local demos."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any

from app.config import SESSION_FILE


FIRST_QUESTION = "자기소개 부탁드립니다."


class SessionStore:
    def __init__(self) -> None:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, dict[str, Any]] = self._load()

    def create_session(
        self,
        *,
        record_id: str,
        target_university: str,
        target_department: str,
        difficulty: str,
    ) -> dict[str, Any]:
        session_id = f"session-{uuid.uuid4().hex[:10]}"
        session = {
            "session_id": session_id,
            "record_id": record_id,
            "target_university": target_university,
            "target_department": target_department,
            "difficulty": difficulty,
            "status": "IN_PROGRESS",
            "current_sub_topic": "",
            "asked_sub_topics": [],
            "follow_up_count": 0,
            "question_count": 1,
            "remaining_time": 600,
            "interview_logs": [
                {
                    "question": FIRST_QUESTION,
                    "answer": "",
                    "response_time": 0,
                    "sub_topic": "",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            ],
            "final_report": {},
        }
        self.sessions[session_id] = session
        self._save()
        return deepcopy(session)

    def get(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            raise KeyError(f"세션을 찾을 수 없습니다: {session_id}")
        return deepcopy(self.sessions[session_id])

    def update(self, session: dict[str, Any]) -> dict[str, Any]:
        self.sessions[session["session_id"]] = deepcopy(session)
        self._save()
        return deepcopy(session)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not SESSION_FILE.exists():
            return {}
        try:
            return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self) -> None:
        SESSION_FILE.write_text(
            json.dumps(self.sessions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

