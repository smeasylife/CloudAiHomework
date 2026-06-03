from __future__ import annotations

import asyncio
import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm import GeminiClient
from app.rag_store import RagStore


QUERY_TEMPLATES = {
    "지원동기": "{department}에 지원하게 된 계기, 관심 분야가 생긴 경험, 진로 선택 이유",
    "전공적합성": "{department}와 관련된 과목 활동, 프로젝트, 독서, 동아리, 탐구 경험",
    "학업역량": "교과 학습 성취, 심화 학습, 개념 이해, 자기주도적 학습 경험",
    "탐구역량": "문제를 정하고 조사, 실험, 분석, 발표, 보고서 작성으로 확장한 경험",
    "협업/리더십": "팀 프로젝트, 조별 활동, 역할 분담, 갈등 조정, 발표 주도, 리더십 경험",
    "인성": "성실성, 책임감, 배려, 봉사, 규칙 준수, 공동체 기여 경험",
    "진로계획": "{department} 진학 후 배우고 싶은 분야, 장래 희망, 진로 목표와 연결된 활동",
}

QUESTION_CATEGORIES = list(QUERY_TEMPLATES)
MAX_REWRITE_ROUNDS = 2
LOW_SCORE_THRESHOLD = 4
EVALUATION_CRITERIA = {
    "student_record_grounding": "학생부 근거성",
    "interview_realism": "면접 현실성",
    "difficulty_appropriateness": "난이도 적절성",
    "question_specificity": "질문의 구체성",
}


QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "difficulty": {"type": "string"},
                },
                "required": [
                    "content",
                    "difficulty",
                ],
            },
        }
    },
    "required": ["questions"],
}


EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "criteria_scores": {
                        "type": "object",
                        "properties": {
                            key: {"type": "integer", "minimum": 0, "maximum": 10}
                            for key in EVALUATION_CRITERIA
                        },
                        "required": list(EVALUATION_CRITERIA),
                    },
                },
                "required": [
                    "index",
                    "criteria_scores",
                ],
            },
        }
    },
    "required": ["evaluations"],
}


class QuestionGenerationState(TypedDict, total=False):
    target_school: str
    target_major: str
    interview_type: str
    rewrite_round: int
    current_questions: list[dict[str, Any]]
    approved_questions: list[dict[str, Any]]
    rejected_questions: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]
    category_contexts: dict[str, list[str]]


class QuestionGenerationGraph:
    def __init__(self, *, llm: GeminiClient, rag_store: RagStore) -> None:
        self.llm = llm
        self.rag_store = rag_store
        self.graph = self._build_graph()

    async def close(self) -> None:
        return None

    def _build_graph(self):
        builder = StateGraph(QuestionGenerationState)
        builder.add_node("generate_questions_node", self.generate_questions_node)
        builder.add_node("evaluate_questions_node", self.evaluate_questions_node)
        builder.add_node("rewrite_questions_node", self.rewrite_questions_node)
        builder.add_node("finalize_node", self.finalize_node)

        builder.add_edge(START, "generate_questions_node")
        builder.add_edge("generate_questions_node", "evaluate_questions_node")
        builder.add_conditional_edges(
            "evaluate_questions_node",
            route_after_evaluation,
            {
                "rewrite_questions_node": "rewrite_questions_node",
                "finalize_node": "finalize_node",
            },
        )
        builder.add_edge("rewrite_questions_node", "evaluate_questions_node")
        builder.add_edge("finalize_node", END)
        return builder.compile()

    async def run(
        self,
        *,
        target_school: str,
        target_major: str,
        interview_type: str,
    ) -> QuestionGenerationState:
        initial_state: QuestionGenerationState = {
            "target_school": target_school,
            "target_major": target_major,
            "interview_type": interview_type,
            "rewrite_round": 0,
            "current_questions": [],
            "approved_questions": [],
            "rejected_questions": [],
            "evaluations": [],
            "category_contexts": {},
        }
        return await self.graph.ainvoke(initial_state)

    async def generate_questions_node(self, state: QuestionGenerationState) -> dict[str, Any]:
        results = await asyncio.gather(
            *[self._generate_category_questions(state, category) for category in QUESTION_CATEGORIES],
            return_exceptions=True,
        )

        questions: list[dict[str, Any]] = []
        category_contexts: dict[str, list[str]] = {}

        for category, result in zip(QUESTION_CATEGORIES, results):
            if isinstance(result, Exception):
                continue

            category_questions, context = result
            questions.extend(category_questions)
            category_contexts[category] = context

        return {
            "current_questions": questions,
            "category_contexts": category_contexts,
        }

    async def evaluate_questions_node(self, state: QuestionGenerationState) -> dict[str, Any]:
        current_questions = state.get("current_questions", [])
        if not current_questions:
            return {
                "evaluations": [],
                "approved_questions": state.get("approved_questions", []),
                "rejected_questions": [],
            }

        evaluations = await self._evaluate_questions(state, current_questions)
        approved = list(state.get("approved_questions", []))
        rejected: list[dict[str, Any]] = []

        evaluation_by_index = {
            int(item.get("index", -1)): item
            for item in evaluations
        }

        for index, question in enumerate(current_questions):
            evaluation = evaluation_by_index.get(index)
            if not evaluation:
                continue
            criteria_scores = normalize_criteria_scores(evaluation.get("criteria_scores"))
            needs_rewrite = has_low_criteria(criteria_scores)
            question_with_eval = {
                **question,
                "evaluation": {
                    "index": index,
                    "criteria_scores": criteria_scores,
                },
            }
            if needs_rewrite:
                rejected.append(question_with_eval)
            else:
                approved.append(question_with_eval)

        return {
            "evaluations": evaluations,
            "approved_questions": approved,
            "rejected_questions": rejected,
        }

    async def rewrite_questions_node(self, state: QuestionGenerationState) -> dict[str, Any]:
        rejected_questions = state.get("rejected_questions", [])
        rewrite_round = state.get("rewrite_round", 0) + 1
        results = await asyncio.gather(
            *[
                self._rewrite_single_question(
                    state=state,
                    category=question.get("category", "기타"),
                    rejected_question=question,
                    criteria_scores=question.get("evaluation", {}).get("criteria_scores", {}),
                    rewrite_round=rewrite_round,
                )
                for question in rejected_questions
            ],
            return_exceptions=True,
        )

        rewritten_questions: list[dict[str, Any]] = []

        for question, result in zip(rejected_questions, results):
            if isinstance(result, Exception):
                fallback = make_rewrite_fallback(question, rewrite_round)
                rewritten_questions.append(fallback)
                continue

            rewritten_questions.append(result)

        return {
            "rewrite_round": rewrite_round,
            "current_questions": rewritten_questions,
            "rejected_questions": [],
        }

    async def finalize_node(self, state: QuestionGenerationState) -> dict[str, Any]:
        approved = state.get("approved_questions", [])
        finalized = sorted(
            approved,
            key=lambda item: category_order(item.get("category", "기타")),
        )
        return {
            "approved_questions": finalized,
        }

    async def _generate_category_questions(
        self,
        state: QuestionGenerationState,
        category: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        query = query_for_category(state, category)
        context = await self.rag_store.search_record_chunks(
            query=query,
            llm=self.llm,
            limit=5,
        )
        few_shot = await self.rag_store.search_interview_questions(
            query=query,
            llm=self.llm,
            limit=5,
        )

        prompt = category_generation_prompt(
            state=state,
            category=category,
            context=context,
            few_shot=few_shot,
        )
        data = await self.llm.generate_json(prompt, schema=QUESTION_SCHEMA, timeout=90)
        questions = normalize_questions(data.get("questions", []), category)[:2]
        return questions, context

    async def _evaluate_questions(
        self,
        state: QuestionGenerationState,
        questions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = evaluation_prompt(state, questions)
        data = await self.llm.generate_json(prompt, schema=EVALUATION_SCHEMA, timeout=120)
        return data.get("evaluations", [])

    async def _rewrite_single_question(
        self,
        *,
        state: QuestionGenerationState,
        category: str,
        rejected_question: dict[str, Any],
        criteria_scores: dict[str, int],
        rewrite_round: int,
    ) -> dict[str, Any]:
        query = " | ".join(
            [
                query_for_category(state, category),
                rejected_question.get("content", ""),
                json.dumps(criteria_scores, ensure_ascii=False),
            ]
        )
        context = await self.rag_store.search_record_chunks(
            query=query,
            llm=self.llm,
            limit=5,
        )
        few_shot = await self.rag_store.search_interview_questions(
            query=query,
            llm=self.llm,
            limit=5,
        )
        prompt = rewrite_prompt(
            state=state,
            category=category,
            rejected_question=rejected_question,
            criteria_scores=criteria_scores,
            context=context,
            few_shot=few_shot,
        )
        data = await self.llm.generate_json(prompt, schema=QUESTION_SCHEMA, timeout=90)
        questions = normalize_questions(data.get("questions", []), category)
        if not questions:
            return make_rewrite_fallback(rejected_question, rewrite_round)
        return questions[0]


def route_after_evaluation(state: QuestionGenerationState) -> str:
    rejected_count = len(state.get("rejected_questions", []))
    rewrite_round = state.get("rewrite_round", 0)
    if rejected_count and rewrite_round < MAX_REWRITE_ROUNDS:
        return "rewrite_questions_node"
    return "finalize_node"


def category_generation_prompt(
    *,
    state: QuestionGenerationState,
    category: str,
    context: list[str],
    few_shot: list[str],
) -> str:
    return f"""당신은 학생부종합전형 대입 면접 질문 생성 전문가입니다.

지원 대학: {state.get("target_school", "")}
지원 학과: {state.get("target_major", "")}
전형 유형: {state.get("interview_type", "")}

생활기록부 검색 청크:
{format_list(context) or "- 관련 청크 없음"}

실제 면접 질문 예시:
{format_list(few_shot) or "- 예시 없음"}

지침:
- 아래 생활기록부 청크와 실제 면접 질문 예시를 참고해 적절한 질문을 정확히 2개 생성하세요.
- 질문은 생활기록부 청크에 실제로 적힌 활동 하나를 중심으로 작성하세요.
- difficulty가 기본인 질문 1개와 심화인 질문 1개를 생성하세요.
- 생활기록부에 명확히 있는 내용만 근거로 삼고, 없는 활동이나 세부사항을 만들지 마세요.
- 실제 면접 질문 예시의 내용을 참고하되 문장을 그대로 복사하지 마세요.
- 질문은 한 문장으로 짧게 쓰고, 활동 하나와 확인할 점 하나만 물어보세요.
- 질문 하나에 여러 요구사항을 넣지 마세요.
- "이유와 과정과 느낀 점을 모두 설명하세요"처럼 2개 이상의 답변 요구를 섞지 마세요.
- "경험을 바탕으로 사회 문제 해결에서 고려할 점", "앞으로 어떻게 대처할 것인지"처럼 활동과 느슨하게 이어지는 일반론 질문은 만들지 마세요.
- "이 경험이 역량 발전에 어떻게 기여했나요", "진로에 어떤 영향을 주었나요", "컴퓨터공학도로서 어떤 역량을 키웠나요"처럼 활동 밖의 자기평가 질문은 만들지 마세요.
- 심화 질문은 대학 전공 지식을 요구하는 질문이 아니라, 학생부 활동의 원리·과정·판단 이유를 한 단계 더 묻는 질문으로 작성하세요.
- 고등학생이 학생부 활동과 고등학교 수준의 배경지식으로 답할 수 있는 범위를 넘지 마세요.
- 기본 질문은 활동 내용 확인, 수행 과정, 배운 개념, 본인의 역할을 확인하는 수준으로 작성하세요.
- 심화 질문은 해당 활동에서 사용한 원리, 선택 이유, 문제 해결 과정, 한계 인식 중 하나만 묻는 수준으로 작성하세요.
- difficulty는 기본 또는 심화만 사용하세요.
- 각 질문 객체는 content와 difficulty만 포함하세요.
- JSON 외 텍스트는 출력하지 마세요.

출력 전 내부 검토:
- 각 질문이 생활기록부 chunk에 직접 근거가 있는지 확인하세요.
- 학생부에 없는 세부 기술을 전제하지 않았는지 확인하세요.
- 활동 하나만 다루는지 확인하세요.
- 확인할 점 하나만 묻는지 확인하세요.
- 실제 학생부종합전형 면접에서 자연스럽게 물을 수 있는지 확인하세요.
- 고등학생이 답변 가능한 수준인지 확인하세요.
- 위 조건을 만족하지 못하면 출력하지 말고 다시 작성하세요.
"""


def evaluation_prompt(state: QuestionGenerationState, questions: list[dict[str, Any]]) -> str:
    contexts = state.get("category_contexts", {})
    compact_context = {
        category: items[:4]
        for category, items in contexts.items()
    }
    return f"""당신은 대입 면접 질문 품질 평가자입니다.

평가 지표:
1. student_record_grounding / 학생부 근거성
- 생성된 면접 질문이 학생부 또는 검색된 학생부 chunk의 실제 내용에 얼마나 정확히 기반하고 있는지 평가한다.
- 높은 점수는 단순히 학생부 주제와 비슷한 질문이 아니라, 학생부에 기록된 구체적인 활동, 표현, 탐구 주제, 산출물, 역할, 과정을 직접 반영한 경우에만 부여한다.
- 학생부에 없는 내용을 질문자가 임의로 확장하거나, 학생부에는 단순히 언급된 수준인데 과도하게 전문적인 경험을 한 것처럼 묻는 경우 감점한다.
- 10점: 학생부의 특정 활동, 주제, 표현, 산출물, 역할이 정확히 반영되어 있으며, 질문의 핵심이 모두 학생부 근거에 기반함.
- 9점: 학생부의 구체적 활동에 명확히 기반하고 있으나, 일부 표현이 약간 일반화되어 있음.
- 8점: 학생부 활동과 직접 관련은 있으나, 질문이 학생부 표현보다 한 단계 확장되어 있음. 그래도 무리한 추정은 아님.
- 7점: 학생부 내용과 관련은 명확하지만, 특정 기록을 세밀하게 짚기보다는 활동 주제를 넓게 활용함.
- 6점: 학생부의 전반적 관심사나 진로 방향과 관련은 있으나, 특정 활동 근거가 약함.
- 5점: 학생부와 간접적으로 연결되지만, 해당 학생이 아니어도 비슷하게 받을 수 있는 질문임.
- 4점: 학생부에 일부 단어 또는 분야만 겹치며, 실제 기록과 질문의 연결이 약함.
- 3점: 학생부의 넓은 진로 관심과만 관련 있고, 구체적 근거가 거의 없음.
- 2점: 학생부에 없는 내용을 바탕으로 질문한 가능성이 큼.
- 1점: 학생부와 거의 무관한 일반 질문임.
- 0점: 학생부 내용과 명백히 충돌하거나, 존재하지 않는 활동을 사실처럼 전제함.
- 강제 감점: 학생부에 단순히 "조사했다" 수준인데 "구현했다", "설계했다", "검증했다"처럼 확장하면 최대 5점.
- 강제 감점: 학생부에 없는 산출물이나 결과를 전제하면 최대 4점.
- 강제 감점: 학생부 내용과 명백히 다른 활동을 전제하면 0점.

2. interview_realism / 면접 현실성
- 질문이 실제 학생부종합전형, 학과 면접, 서류 기반 면접에서 나올 법한지 평가한다.
- 좋은 면접 질문은 활동의 실제 수행 여부, 본인의 역할, 사고 과정, 배운 점, 전공 관심도, 기본 개념 이해, 후속 탐구 가능성을 확인해야 한다.
- 단순 지식시험, 대학 전공시험, 지나치게 전문적인 기술면접, 너무 추상적인 가치관 질문은 낮게 평가한다.
- 10점: 실제 학생부종합전형 면접에서 매우 자연스럽게 나올 법하며, 학생부 확인과 사고력 평가가 동시에 가능함.
- 9점: 실제 면접 질문으로 충분히 자연스럽고, 답변을 통해 학생의 경험과 이해도를 확인할 수 있음.
- 8점: 면접 질문으로 적절하나, 약간 기술적이거나 표현을 다듬으면 더 자연스러움.
- 7점: 면접에서 가능은 하지만 다소 전공 질문 또는 발표 질문처럼 느껴짐.
- 6점: 실제 면접보다는 교과 심화 질문에 가까움.
- 5점: 면접 질문으로 사용할 수는 있으나, 학생부 확인 질문으로서의 자연스러움이 부족함.
- 4점: 대학 전공 구술시험이나 기술면접에 가까움.
- 3점: 학생부 면접보다는 지식 암기 문제, 논술형 토론 주제, 과제 발표 주제에 가까움.
- 2점: 실제 면접관이 묻기에는 지나치게 부자연스럽거나 부담스러움.
- 1점: 면접 질문으로 거의 사용할 수 없음.
- 0점: 질문 형식이 아니거나, 면접 맥락과 무관함.

3. difficulty_appropriateness / 난이도 적절성
- 질문이 고등학생 학생부종합전형 면접 수준에서 답변 가능한지 평가한다.
- 좋은 질문은 고등학생이 자신의 활동 경험과 고등학교 수준의 배경지식을 바탕으로 사고하여 답할 수 있어야 한다.
- 단순 정의만 말하면 되는 질문은 낮게 평가하고, 반대로 대학 전공 지식이나 연구 수준을 요구하는 질문도 낮게 평가한다.
- 10점: 고등학생이 자신의 활동 경험을 바탕으로 충분히 답할 수 있으면서 사고력과 이해도를 잘 드러낼 수 있음.
- 9점: 약간의 심화 사고가 필요하지만 고등학생 면접 수준에서 매우 적절함.
- 8점: 다소 어렵거나 쉽지만 준비 가능한 범위 안에 있음.
- 7점: 고등학생 상위권에게는 가능하지만 평균적인 학생에게는 다소 부담스러움.
- 6점: 특정 개념을 모르면 답변이 어려우며, 활동 경험보다 지식 수준에 의존함.
- 5점: 답변 가능성은 있으나 난이도 조정이 필요함.
- 4점: 대학 1~2학년 전공 지식에 가까움.
- 3점: 고등학생 면접 질문으로는 과도하게 전문적임.
- 2점: 연구, 실무 개발, 대학 전공시험 수준을 요구함.
- 1점: 일반 고등학생이 답변하기 매우 부적절함.
- 0점: 질문의 난이도 판단이 불가능하거나 질문 자체가 성립하지 않음.

4. question_specificity / 질문의 구체성
- 질문의 배경, 대상, 답변 요구가 얼마나 명확한지 평가한다.
- 좋은 질문은 학생이 무엇을 중심으로 답해야 하는지 바로 알 수 있어야 한다.
- 다만 구체성이 높다는 이유만으로 좋은 질문은 아니다.
- 학생부에 없는 세부사항을 억지로 넣어 구체적으로 만든 경우에는 구체성 점수도 감점한다.
- 10점: 질문의 배경, 활동, 답변 요구가 명확하며, 답변 방향이 지나치게 좁거나 넓지 않음.
- 9점: 매우 명확하지만 답변 요구가 약간 많음.
- 8점: 대체로 구체적이고 답변 방향이 분명함.
- 7점: 질문은 이해 가능하지만 답변 범위가 조금 넓거나 요구사항이 2개 이상 섞여 있음.
- 6점: 질문 배경은 있으나 핵심 답변 요구가 다소 흐림.
- 5점: 답변은 가능하지만 일반적이고 구체성이 부족함.
- 4점: 무엇을 중심으로 답해야 하는지 애매함.
- 3점: 질문 범위가 지나치게 넓거나 추상적임.
- 2점: 여러 질문이 한 문장에 섞여 답변 방향이 혼란스러움.
- 1점: 매우 모호하거나 포괄적임.
- 0점: 질문으로서 의미가 불명확함.

각 기준은 0~10점으로 평가하세요.
5점은 보통 점수가 아니라 결함이 분명한 질문에 주는 점수입니다.
10점은 기준을 매우 엄격하게 만족할 때만 부여하세요.
지원 대학: {state.get("target_school", "")}
지원 학과: {state.get("target_major", "")}
전형 유형: {state.get("interview_type", "")}

카테고리별 생활기록부 청크:
{json.dumps(compact_context, ensure_ascii=False, indent=2)}

평가할 질문 목록:
{json.dumps(questions, ensure_ascii=False, indent=2)}

각 질문의 배열 index를 유지해서 평가하고, criteria_scores만 작성하세요.
JSON 외 텍스트는 출력하지 마세요.
"""


def rewrite_prompt(
    *,
    state: QuestionGenerationState,
    category: str,
    rejected_question: dict[str, Any],
    criteria_scores: dict[str, int],
    context: list[str],
    few_shot: list[str],
) -> str:
    question_summary = {
        "content": rejected_question.get("content", ""),
        "difficulty": rejected_question.get("difficulty", ""),
    }
    return f"""당신은 대입 면접 질문 재작성 전문가입니다.

아래 질문은 평가에서 기준을 만족하지 못했습니다. 생활기록부 청크에 실제로 적힌 활동을 중심으로 새 질문 1개만 다시 생성하세요.

지원 대학: {state.get("target_school", "")}
지원 학과: {state.get("target_major", "")}
전형 유형: {state.get("interview_type", "")}

기존 질문:
{json.dumps(question_summary, ensure_ascii=False, indent=2)}

평가 점수:
{json.dumps(criteria_scores, ensure_ascii=False, indent=2)}

재검색한 생활기록부 청크:
{format_list(context) or "- 관련 청크 없음"}

재검색한 실제 면접 질문 예시:
{format_list(few_shot) or "- 예시 없음"}

지침:
- 질문은 1개만 생성하세요.
- 평가 점수에서 낮은 기준을 판단해 보완하세요.
- 재검색한 생활기록부 청크와 실제 면접 질문 예시를 참고해 적절한 질문을 만드세요.
- 학생부에 없는 사실을 만들지 마세요.
- 질문은 한 문장으로 짧게 쓰고, 활동 하나와 확인할 점 하나만 물어보세요.
- "경험을 바탕으로 사회 문제 해결에서 고려할 점", "앞으로 어떻게 대처할 것인지"처럼 활동과 느슨하게 이어지는 일반론 질문은 만들지 마세요.
- "이 경험이 역량 발전에 어떻게 기여했나요", "진로에 어떤 영향을 주었나요", "컴퓨터공학도로서 어떤 역량을 키웠나요"처럼 활동 밖의 자기평가 질문은 만들지 마세요.
- 적당히 구체적으로 묻되 너무 길거나 깊게 파고들지 마세요.
- 고등학생이 학생부 활동과 고등학교 수준의 배경지식으로 답할 수 있는 범위를 넘지 마세요.
- 질문 객체는 content와 difficulty만 포함하세요.
- JSON 외 텍스트는 출력하지 마세요.
"""


def normalize_questions(raw_questions: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in raw_questions:
        content = item.get("content").strip()
        if not content:
            continue
        normalized.append(
            {
                "category": category,
                "content": content,
                "difficulty": item.get("difficulty"),
            }
        )
    return normalized


def query_for_category(state: QuestionGenerationState, category: str) -> str:
    template = QUERY_TEMPLATES.get(category, category)
    return template.format(department=state.get("target_major", ""))


def normalize_criteria_scores(value: Any) -> dict[str, int]:
    scores = {}
    for key in EVALUATION_CRITERIA:
        score = int(value[key])
        scores[key] = max(0, min(score, 10))
    return scores


def has_low_criteria(criteria_scores: dict[str, int]) -> bool:
    return any(score <= LOW_SCORE_THRESHOLD for score in criteria_scores.values())


def make_rewrite_fallback(question: dict[str, Any], rewrite_round: int) -> dict[str, Any]:
    category = question.get("category", "기타")
    return {
        **question,
        "content": f"{question.get('content', '')} 이 활동에서 본인이 내린 판단 기준과 그 판단이 결과에 미친 영향을 구체적으로 설명해 주세요.",
    }


def format_list(items: list[Any]) -> str:
    return "\n".join(f"- {item}" for item in items if str(item).strip())


def category_order(category: str) -> int:
    try:
        return QUESTION_CATEGORIES.index(category)
    except ValueError:
        return len(QUESTION_CATEGORIES)
