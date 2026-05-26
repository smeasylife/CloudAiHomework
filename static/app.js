const state = {
  recordId: null,
  sessionId: null,
  questionStartedAt: null,
};

const $ = (id) => document.getElementById(id);

const ingestButton = $("ingestButton");
const startButton = $("startButton");
const sendButton = $("sendButton");
const reportButton = $("reportButton");
const answerInput = $("answerInput");
const answerForm = $("answerForm");
const messages = $("messages");

ingestButton.addEventListener("click", ingestPdf);
startButton.addEventListener("click", startInterview);
answerForm.addEventListener("submit", submitAnswer);
reportButton.addEventListener("click", showReport);
$("closeReportButton").addEventListener("click", () => $("reportPanel").classList.add("hidden"));

async function ingestPdf() {
  const file = $("pdfInput").files[0];
  const targetDepartment = $("departmentInput").value.trim();
  if (!file || !targetDepartment) {
    setStatus("ingestResult", "PDF와 지원 학과를 입력하세요.", true);
    return;
  }

  const formData = new FormData();
  formData.append("pdf", file);
  formData.append("target_department", targetDepartment);

  ingestButton.disabled = true;
  startButton.disabled = true;
  setStatus("ingestResult", "실제 면접 질문 DB 시딩, PDF 청킹, Gemini 임베딩을 진행 중입니다.");

  try {
    const data = await api("/api/ingest", { method: "POST", body: formData });
    state.recordId = data.record_id;
    setStatus(
      "ingestResult",
      `record_id: ${data.record_id}\n생기부 청크: ${data.chunk_count}개\n실제 면접 질문 시드: ${data.question_seed_count}개`
    );
    startButton.disabled = false;
  } catch (error) {
    setStatus("ingestResult", error.message, true);
  } finally {
    ingestButton.disabled = false;
  }
}

async function startInterview() {
  if (!state.recordId) return;
  startButton.disabled = true;
  addMessage("system", "LangGraph 세션을 생성하는 중입니다.");

  try {
    const data = await api("/api/interview/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        record_id: state.recordId,
        target_university: $("universityInput").value.trim(),
        target_department: $("departmentInput").value.trim(),
        difficulty: $("difficultyInput").value,
      }),
    });
    state.sessionId = data.session_id;
    messages.innerHTML = "";
    addMessage("interviewer", data.first_question);
    answerInput.disabled = false;
    sendButton.disabled = false;
    reportButton.disabled = false;
    state.questionStartedAt = Date.now();
  } catch (error) {
    addMessage("system", error.message, true);
    startButton.disabled = false;
  }
}

async function submitAnswer(event) {
  event.preventDefault();
  const answer = answerInput.value.trim();
  if (!answer || !state.sessionId) return;

  const responseTime = Math.max(1, Math.round((Date.now() - state.questionStartedAt) / 1000));
  answerInput.value = "";
  answerInput.disabled = true;
  sendButton.disabled = true;
  addMessage("user", answer);
  addMessage("system", "LangGraph가 답변 분석 → 분기 판단 → RAG 검색 → 다음 질문 생성을 실행 중입니다.");

  try {
    const data = await api("/api/interview/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        answer,
        response_time: responseTime,
      }),
    });
    updateGraphState(data);
    removeLastSystemMessage();

    if (data.is_finished) {
      addMessage("interviewer", "면접을 종료합니다. 수고하셨습니다.");
      await showReport();
      return;
    }

    addMessage("interviewer", data.next_question);
    answerInput.disabled = false;
    sendButton.disabled = false;
    state.questionStartedAt = Date.now();
    answerInput.focus();
  } catch (error) {
    removeLastSystemMessage();
    addMessage("system", error.message, true);
    answerInput.disabled = false;
    sendButton.disabled = false;
  }
}

async function showReport() {
  if (!state.sessionId) return;
  try {
    const data = await api(`/api/interview/${state.sessionId}/report`);
    $("reportOutput").textContent = JSON.stringify(data.report, null, 2);
    $("reportPanel").classList.remove("hidden");
  } catch (error) {
    addMessage("system", error.message, true);
  }
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    throw new Error(data.detail || "요청 처리 중 오류가 발생했습니다.");
  }
  return data;
}

function addMessage(role, text, isError = false) {
  const message = document.createElement("div");
  message.className = `message ${role === "user" ? "user" : "interviewer"}`;
  if (role === "system") message.dataset.system = "true";
  const roleLabel = role === "user" ? "지원자" : role === "system" ? "시스템" : "면접관";
  message.innerHTML = `<div class="role">${roleLabel}</div><p></p>`;
  message.querySelector("p").textContent = text;
  if (isError) message.querySelector("p").classList.add("error");
  messages.appendChild(message);
  messages.scrollTop = messages.scrollHeight;
}

function removeLastSystemMessage() {
  const systemMessages = messages.querySelectorAll("[data-system='true']");
  const last = systemMessages[systemMessages.length - 1];
  if (last) last.remove();
}

function updateGraphState(data) {
  $("topicState").textContent = data.current_sub_topic || "-";
  $("actionState").textContent = data.action || "-";
  $("timeState").textContent = `${data.remaining_time}초`;
}

function setStatus(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", isError);
}
