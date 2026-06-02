let hasRecord = false;
const $ = (id) => document.getElementById(id);

const ingestButton = $("ingestButton");
const pdfInput = $("pdfInput");
const generateButton = $("generateButton");

ingestButton.addEventListener("click", () => {
  pdfInput.value = "";
  pdfInput.click();
});
pdfInput.addEventListener("change", ingestPdf);
generateButton.addEventListener("click", generateQuestions);

loadRecordStatus();

async function loadRecordStatus() {
  setStatus("ingestResult", "저장된 생기부를 확인하는 중입니다.");

  try {
    const data = await api("/api/records/status");
    hasRecord = data.has_record;
    if (!hasRecord) {
      setStatus("ingestResult", "저장된 생기부가 없습니다.");
      return;
    }

    setStatus("ingestResult", "저장된 생기부가 있습니다.");
    generateButton.disabled = false;
  } catch (error) {
    setStatus("ingestResult", error.message, true);
  }
}

async function ingestPdf() {
  const file = pdfInput.files[0];
  if (!file) {
    setStatus("ingestResult", "PDF를 선택하세요.", true);
    return;
  }

  const formData = new FormData();
  formData.append("pdf", file);

  ingestButton.disabled = true;
  generateButton.disabled = true;
  setStatus("ingestResult", "처리 중입니다.");

  try {
    const data = await api("/api/ingest", { method: "POST", body: formData });
    hasRecord = true;
    setStatus("ingestResult", `청크 ${data.chunk_count}개\n면접 질문 DB ${data.question_seed_count}개`);
    generateButton.disabled = false;
  } catch (error) {
    setStatus("ingestResult", error.message, true);
  } finally {
    ingestButton.disabled = false;
  }
}

async function generateQuestions() {
  if (!hasRecord) return;

  generateButton.disabled = true;
  renderLoading();

  try {
    const data = await api("/api/questions/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_school: $("schoolInput").value.trim(),
        target_major: $("majorInput").value.trim(),
        interview_type: $("interviewTypeInput").value.trim(),
      }),
    });

    renderQuestions(data.questions || []);
  } catch (error) {
    renderError(error.message);
  } finally {
    generateButton.disabled = false;
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

function renderQuestions(questions) {
  const root = $("questionGroups");
  root.innerHTML = "";

  if (!questions.length) {
    root.innerHTML = "<p>생성된 질문이 없습니다.</p>";
    return;
  }

  renderQuestionSection(root, "기본 질문", questions.filter((question) => question.difficulty !== "심화"));
  renderQuestionSection(root, "심화 질문", questions.filter((question) => question.difficulty === "심화"));
}

function renderQuestionSection(root, title, questions) {
  if (!questions.length) return;

  const section = document.createElement("section");
  section.className = "question-section";
  section.innerHTML = "<h3></h3>";
  section.querySelector("h3").textContent = title;
  questions.forEach((question) => section.appendChild(questionCard(question)));
  root.appendChild(section);
}

function questionCard(question) {
  const card = document.createElement("article");
  card.className = "question-card";
  card.innerHTML = "<h4></h4>";
  card.querySelector("h4").textContent = question.content || "";
  return card;
}

function renderLoading() {
  $("questionGroups").innerHTML = '<p>질문 생성 중입니다.</p>';
}

function renderError(message) {
  $("questionGroups").innerHTML = `<p class="error"></p>`;
  $("questionGroups").querySelector(".error").textContent = message;
}

function setStatus(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", isError);
}
