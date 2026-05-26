"""LangGraph interview flow with RAG-backed question generation."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import SUB_TOPICS
from app.llm import GeminiClient, parse_json_object
from app.rag_store import RagStore


class InterviewState(TypedDict, total=False):
    session_id: str
    record_id: str
    target_university: str
    target_department: str
    difficulty: str
    current_sub_topic: str
    asked_sub_topics: list[str]
    follow_up_count: int
    question_count: int
    remaining_time: int
    interview_logs: list[dict[str, Any]]
    last_question: str
    answer: str
    response_time: int
    action: str
    analysis: str
    student_context: list[str]
    few_shot_questions: list[str]
    next_question: str
    final_report: dict[str, Any]


class InterviewGraph:
    def __init__(self, *, llm: GeminiClient, rag_store: RagStore) -> None:
        self.llm = llm
        self.rag_store = rag_store
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(InterviewState)
        builder.add_node("analyze_answer", self.analyze_answer)
        builder.add_node("decide_next_action", self.decide_next_action)
        builder.add_node("retrieve_context", self.retrieve_context)
        builder.add_node("generate_follow_up", self.generate_follow_up)
        builder.add_node("generate_new_topic", self.generate_new_topic)
        builder.add_node("generate_report", self.generate_report)

        builder.add_edge(START, "analyze_answer")
        builder.add_edge("analyze_answer", "decide_next_action")
        builder.add_conditional_edges(
            "decide_next_action",
            route_after_decision,
            {
                "retrieve_context": "retrieve_context",
                "generate_report": "generate_report",
            },
        )
        builder.add_conditional_edges(
            "retrieve_context",
            route_after_retrieval,
            {
                "generate_follow_up": "generate_follow_up",
                "generate_new_topic": "generate_new_topic",
            },
        )
        builder.add_edge("generate_follow_up", END)
        builder.add_edge("generate_new_topic", END)
        builder.add_edge("generate_report", END)
        return builder.compile()

    async def run(self, *, session: dict[str, Any], answer: str, response_time: int) -> InterviewState:
        logs = session.get("interview_logs", [])
        state: InterviewState = {
            "session_id": session["session_id"],
            "record_id": session["record_id"],
            "target_university": session["target_university"],
            "target_department": session["target_department"],
            "difficulty": session["difficulty"],
            "current_sub_topic": session.get("current_sub_topic", ""),
            "asked_sub_topics": session.get("asked_sub_topics", []),
            "follow_up_count": session.get("follow_up_count", 0),
            "question_count": session.get("question_count", len(logs)),
            "remaining_time": max(0, session.get("remaining_time", 600) - response_time),
            "interview_logs": logs,
            "last_question": logs[-1]["question"] if logs else "",
            "answer": answer,
            "response_time": response_time,
        }
        return await self.graph.ainvoke(state)

    async def analyze_answer(self, state: InterviewState) -> dict[str, Any]:
        if not state.get("current_sub_topic"):
            return {
                "analysis": "자기소개 답변 이후 첫 학생부 주제로 전환합니다.",
                "action": "new_topic",
            }

        prompt = f"""당신은 대학 입시 면접관입니다. 학생 답변을 분석해 다음 액션을 하나만 고르세요.

가능한 액션:
- follow_up: 답변이 추상적이거나 근거, 판단 기준, 배운 점을 더 물어봐야 함
- new_topic: 답변이 충분하고 다음 학생부 주제로 넘어가도 됨
- wrap_up: 시간이 부족하거나 면접을 종료해도 됨

면접 난이도: {state.get("difficulty", "Normal")}
현재 주제: {state.get("current_sub_topic", "")}
꼬리 질문 횟수: {state.get("follow_up_count", 0)}
남은 시간: {state.get("remaining_time", 0)}초

질문:
{state.get("last_question", "")}

학생 답변:
{state.get("answer", "")}

출력은 follow_up, new_topic, wrap_up 중 하나만 반환하세요."""
        try:
            action = (await self.llm.generate_text(prompt, temperature=0.1, timeout=45)).strip().lower()
            if action not in {"follow_up", "new_topic", "wrap_up"}:
                action = heuristic_action(state)
        except Exception:
            action = heuristic_action(state)
        return {"analysis": f"LLM suggested action: {action}", "action": action}

    async def decide_next_action(self, state: InterviewState) -> dict[str, Any]:
        remaining_time = state.get("remaining_time", 0)
        asked_sub_topics = state.get("asked_sub_topics", [])
        follow_up_count = state.get("follow_up_count", 0)
        action = state.get("action", "new_topic")

        if remaining_time < 30 or len(asked_sub_topics) >= 4:
            action = "wrap_up"
        elif follow_up_count >= 2:
            action = "new_topic"
        elif action not in {"follow_up", "new_topic", "wrap_up"}:
            action = "new_topic"

        return {"action": action}

    async def retrieve_context(self, state: InterviewState) -> dict[str, Any]:
        action = state["action"]
        asked_sub_topics = list(state.get("asked_sub_topics", []))
        current_sub_topic = state.get("current_sub_topic", "")

        if action == "new_topic":
            if current_sub_topic and current_sub_topic not in asked_sub_topics:
                asked_sub_topics.append(current_sub_topic)
            current_sub_topic = choose_next_topic(asked_sub_topics, state.get("answer", ""))
            follow_up_count = 0
        else:
            if not current_sub_topic:
                current_sub_topic = choose_next_topic(asked_sub_topics, state.get("answer", ""))
            follow_up_count = state.get("follow_up_count", 0)

        query = f"{state.get('target_department', '')} | {current_sub_topic}"
        student_context = await self.rag_store.search_record_chunks(
            record_id=state["record_id"],
            query=query,
            llm=self.llm,
            limit=4,
        )
        few_shot_questions = await self.rag_store.search_interview_questions(
            query=query,
            llm=self.llm,
            limit=8,
        )
        return {
            "current_sub_topic": current_sub_topic,
            "asked_sub_topics": asked_sub_topics,
            "follow_up_count": follow_up_count,
            "student_context": student_context,
            "few_shot_questions": few_shot_questions,
        }

    async def generate_follow_up(self, state: InterviewState) -> dict[str, Any]:
        prompt = question_prompt(
            state,
            instruction=(
                "학생 답변에서 언급한 구체적 사례, 판단 근거, 배운 점을 집요하게 확인하는 "
                "꼬리 질문 1개를 생성하세요."
            ),
        )
        question = await self._safe_question(prompt)
        return {
            "next_question": question,
            "follow_up_count": state.get("follow_up_count", 0) + 1,
            "question_count": state.get("question_count", 0) + 1,
        }

    async def generate_new_topic(self, state: InterviewState) -> dict[str, Any]:
        prompt = question_prompt(
            state,
            instruction=(
                "새로운 학생부 주제에 대한 첫 질문 1개를 생성하세요. 개방형으로 묻되, "
                "학생부 근거와 실제 면접 질문 예시의 톤을 반영하세요."
            ),
        )
        question = await self._safe_question(prompt)
        return {
            "next_question": question,
            "question_count": state.get("question_count", 0) + 1,
        }

    async def generate_report(self, state: InterviewState) -> dict[str, Any]:
        logs = completed_logs(state)
        prompt = f"""당신은 대입 면접 피드백 코치입니다. 아래 면접 로그를 JSON 리포트로 평가하세요.

지원 대학: {state.get("target_university", "")}
지원 학과: {state.get("target_department", "")}
난이도: {state.get("difficulty", "Normal")}

면접 로그:
{json.dumps(logs, ensure_ascii=False, indent=2)}

반드시 아래 JSON 구조로만 반환하세요:
{{
  "scores": {{
    "전공적합성": 0,
    "인성": 0,
    "발전가능성": 0,
    "의사소통능력": 0,
    "총점": 0
  }},
  "strengths": ["강점"],
  "weaknesses": ["약점"],
  "per_question_feedback": [
    {{
      "question": "질문",
      "sub_topic": "주제",
      "evaluation": "좋음/보통/부족",
      "improvement": "개선 포인트"
    }}
  ]
}}"""
        try:
            text = await self.llm.generate_text(prompt, temperature=0.2, timeout=60)
            report = parse_json_object(text)
        except Exception:
            report = fallback_report(logs)
        return {"final_report": report, "next_question": None}

    async def _safe_question(self, prompt: str) -> str:
        try:
            question = await self.llm.generate_text(prompt, temperature=0.5, timeout=60)
        except Exception:
            question = ""
        question = question.strip()
        return question or "답변에서 가장 중요하게 판단했던 기준과 그 이유를 구체적으로 설명해 주세요."


def route_after_decision(state: InterviewState) -> str:
    if state.get("action") == "wrap_up":
        return "generate_report"
    return "retrieve_context"


def route_after_retrieval(state: InterviewState) -> str:
    if state.get("action") == "follow_up":
        return "generate_follow_up"
    return "generate_new_topic"


def heuristic_action(state: InterviewState) -> str:
    answer = state.get("answer", "")
    if state.get("remaining_time", 0) < 30:
        return "wrap_up"
    if state.get("follow_up_count", 0) >= 1:
        return "new_topic"
    if len(answer) < 80:
        return "follow_up"
    return "new_topic"


def choose_next_topic(asked_sub_topics: list[str], answer: str) -> str:
    priority = ["진로/자율", "동아리", "세특", "독서", "리더십", "인성/태도", "성적", "봉사", "출결"]
    answer_text = answer.lower()
    keyword_topics = [
        ("동아리", ["동아리", "프로젝트", "팀"]),
        ("독서", ["책", "독서", "읽"]),
        ("리더십", ["리더", "회장", "갈등", "조장"]),
        ("진로/자율", ["진로", "전공", "컴퓨터", "ai", "개발"]),
        ("봉사", ["봉사"]),
        ("성적", ["성적", "과목", "수학", "과학"]),
    ]
    for topic, keywords in keyword_topics:
        if topic not in asked_sub_topics and any(keyword in answer_text for keyword in keywords):
            return topic
    for topic in priority:
        if topic not in asked_sub_topics:
            return topic
    return SUB_TOPICS[0]


def question_prompt(state: InterviewState, *, instruction: str) -> str:
    student_context = "\n\n".join(f"- {item}" for item in state.get("student_context", [])) or "- 관련 청크 없음"
    few_shot = "\n".join(f"- {item}" for item in state.get("few_shot_questions", [])) or "- 예시 없음"
    return f"""당신은 학생부종합전형 대입 면접관입니다.

면접 난이도: {state.get("difficulty", "Normal")}
지원 대학: {state.get("target_university", "")}
지원 학과: {state.get("target_department", "")}
현재 주제: {state.get("current_sub_topic", "")}

이전 질문:
{state.get("last_question", "")}

학생 답변:
{state.get("answer", "")}

검색된 생활기록부 청크:
{student_context}

실제 면접 질문 예시:
{few_shot}

지침:
- {instruction}
- 질문은 한국어 1문장으로 작성
- 학생부에 없는 내용을 단정하지 말 것
- 고등학생 면접 수준을 유지하되 근거와 판단 기준을 물을 것
- 질문만 출력하고 설명은 쓰지 말 것
"""


def completed_logs(state: InterviewState) -> list[dict[str, Any]]:
    logs = list(state.get("interview_logs", []))
    if logs:
        logs[-1] = {
            **logs[-1],
            "answer": state.get("answer", ""),
            "response_time": state.get("response_time", 0),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    return logs


def fallback_report(logs: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [log for log in logs if log.get("answer")]
    base = min(80, 45 + len(answered) * 8)
    return {
        "scores": {
            "전공적합성": base,
            "인성": min(90, base + 5),
            "발전가능성": base,
            "의사소통능력": min(85, base + 3),
            "총점": min(100, base + 5),
        },
        "strengths": ["면접 흐름 완주", "경험 기반 답변 시도"],
        "weaknesses": ["구체적 수치와 결과 보강 필요"],
        "per_question_feedback": [
            {
                "question": log.get("question", ""),
                "sub_topic": log.get("sub_topic", ""),
                "evaluation": "보통",
                "improvement": "결론을 먼저 말하고, 활동의 결과와 배운 점을 구체적으로 덧붙이세요.",
            }
            for log in answered
        ],
    }

