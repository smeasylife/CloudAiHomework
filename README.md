# LangGraph RAG 면접 토이 프로젝트

기말과제 제출/시연용으로 분리한 로컬 전용 미니 프로젝트입니다. 
## 포함 개념

- 생기부 PDF 파싱/청킹: PDF 파일 자체를 Gemini 2.5 Flash에 전달해 OCR과 카테고리별 청킹 수행
- 임베딩: Gemini embedding 768차원 벡터 생성
- RAG: 로컬 Chroma 컬렉션 2개 사용
  - `student_record_chunks`: 업로드한 생기부 청크
  - `interview_questions`: 실제 면접 질문 데이터
- LangGraph: 답변 분석, 분기, 검색, 질문 생성, 리포트 생성
- 정적 프론트: 브라우저에서 업로드부터 면접과 리포트까지 시연

## 실행

```bash
cd langgraph_rag_interview_toy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`에 Google API Key를 입력합니다.

```bash
uvicorn app.main:app --reload --port 8010
```

브라우저에서 `http://127.0.0.1:8010`을 엽니다.

## API

### `POST /api/ingest`

multipart form:

- `pdf`: 생활기록부 PDF
- `target_department`: 지원 학과

응답:

```json
{
  "record_id": "record-...",
  "chunk_count": 12,
  "question_seed_count": 328
}
```

### `POST /api/interview/start`

```json
{
  "record_id": "record-...",
  "target_university": "가천대학교",
  "target_department": "컴퓨터공학과",
  "difficulty": "Normal"
}
```

### `POST /api/interview/answer`

```json
{
  "session_id": "session-...",
  "answer": "답변 내용",
  "response_time": 42
}
```

### `GET /api/interview/{session_id}/report`

최종 리포트 또는 현재 면접 로그를 반환합니다.

## 코드 구조

```text
app/
  main.py          FastAPI 엔트리포인트
  ingestion.py     Gemini PDF OCR/청킹, Chroma 저장
  rag_store.py     Chroma 컬렉션과 검색 로직
  graph.py         LangGraph StateGraph 면접 분기
  llm.py           Gemini 텍스트 생성/JSON/임베딩 래퍼
  session_store.py 로컬 JSON 세션 저장소
static/
  index.html
  styles.css
  app.js
data/
  interview_questions.json
```

## Git 추적 제외

상위 프로젝트의 `.git/info/exclude`에 `langgraph_rag_interview_toy/`가 등록되어 있어 이 폴더는 Git에 추적되지 않습니다.
