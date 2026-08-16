# HR/IT Helpdesk Assistant

An AI-powered internal helpdesk assistant that answers employee questions from company
policy documents (RAG) and can take real actions — checking leave balances or raising
support tickets — when a question needs more than a document lookup (Agent).

Built for the Codenixia AI/ML Internship Technical Selection Challenge 2026.

---

## 1. Problem Statement

Employees in most organizations repeatedly ask HR and IT the same questions: "how many
sick days do I have left?", "how do I set up VPN?", "how do I get reimbursed for
travel?". Answering these manually costs HR/IT staff hours every week, and employees
often wait hours or days for answers to questions that are already written down in a
policy document somewhere they haven't read.

**Target users:** Employees at a mid-size company, and the HR/IT teams who currently
field these repetitive questions.

**Why this needs an AI-enabled solution, not just a search bar:** Employees ask
questions in natural language ("can I carry over my leave?") that rarely match the exact
wording of a policy document. A keyword search returns irrelevant sections or nothing;
an LLM grounded in the actual policy text can understand the intent and give a direct,
accurate answer — and unlike a static FAQ page, it can also take the next step
(raising a ticket, checking a live balance) instead of just pointing at a document.

**Proposed solution:** A RAG-based assistant with a small agent layer: it retrieves the
relevant policy text for a question, answers from that grounded context, and decides
when a question needs an action (checking a live leave balance, or opening a ticket for
an issue it can't resolve from documents alone) rather than just an answer.

**Expected outcome:** Reduced repetitive-question load on HR/IT, faster answers for
employees, with every answer traceable back to a specific policy document.

---

## 2. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Data Layer                     │
                    │  Policy docs (.md) → data_pipeline.py        │
                    │  load → validate → clean → chunk             │
                    └───────────────────┬───────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │           RAG Layer (rag.py)               │
                    │  TF-IDF vectorization → in-memory vector    │
                    │  store → cosine-similarity retrieval        │
                    └───────────────────┬───────────────────────┘
                                        │ top-k relevant chunks
                    ┌───────────────────▼───────────────────────┐
                    │         Agent Layer (agent.py)              │
                    │  question + retrieved context + tool defs   │
                    │  → Claude (Anthropic API) → answer OR       │
                    │    tool call → execute tool → final answer  │
                    └───────────┬───────────────────┬───────────┘
                                │                   │
                    ┌───────────▼──────┐  ┌─────────▼─────────┐
                    │  tools.py         │  │  Claude (LLM)      │
                    │  raise_ticket()   │  │  reasoning + NLG   │
                    │  check_leave_bal()│  └────────────────────┘
                    └───────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │          API Layer (main.py - FastAPI)      │
                    │  POST /ask   GET /health   GET /            │
                    └───────────────────────────────────────────┘
```

**Request flow:** `User question → POST /ask → retrieve() [RAG] → answer_question()
[Agent + Claude] → (optional tool execution) → JSON response with answer + sources +
actions taken`

---

## 3. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Data processing | Pure Python | Small, well-understood corpus — no need for heavier tooling |
| Embeddings / retrieval | scikit-learn TF-IDF + cosine similarity | Fully local, no external model downloads, sufficient for a small domain-specific corpus (see DECISION_LOG.md) |
| LLM | Claude (Anthropic API) | Strong tool-calling support and instruction-following for grounded Q&A |
| Agent / tool calling | Anthropic Messages API tools | Native tool-use loop, no extra agent framework needed for 2 tools |
| API | FastAPI | Async support, automatic validation via Pydantic, built-in OpenAPI docs |
| Testing | pytest | Standard, integrates with FastAPI's TestClient |
| Containerization | Docker | Reproducible execution across environments |

See `DECISION_LOG.md` for the reasoning behind each of these choices and the
alternatives considered.

---

## 4. Data / Knowledge Sources

- **Source:** Self-created markdown policy documents (`data/policies/`) modeled on
  typical HR/IT policies (leave, WFH, reimbursement, IT/VPN setup, onboarding).
- **Format:** Markdown, structured with `##` section headings, which the pipeline uses
  as natural chunk boundaries.
- **Data quality:** Each document is validated for non-empty content and the presence
  of structural headings before being processed; documents failing validation are
  logged and skipped rather than silently ignored.
- **Privacy/security:** No real employee or company data is used. The "leave balance"
  lookup uses a small mock dataset (`src/tools.py`) rather than any real HR system, and
  tickets are written to a local JSON file, not a production ticketing system.

---

## 5. AI/ML Approach

This problem is primarily **natural-language understanding and generation**, not
classification/regression — the goal is to interpret an open-ended question and produce
a grounded, natural-language answer, which is exactly what LLMs are suited for.
Classical ML (e.g. classifying questions into fixed categories) was considered and
rejected — see `DECISION_LOG.md` — since the category set would be arbitrary and RAG +
LLM already generalizes to unseen phrasings without retraining.

- **Retrieval:** TF-IDF vectors + cosine similarity over chunked policy documents.
- **Generation:** Claude, grounded strictly in retrieved context via the system prompt,
  with instructions to say "I don't know" rather than hallucinate when context is
  insufficient.
- **Agentic decision-making:** Claude is given two tools (`check_leave_balance`,
  `raise_ticket`) and decides, per-question, whether the answer requires a document
  lookup, a live data lookup, or an action — rather than every question being forced
  through the same path.

**Limitations:** TF-IDF retrieval is lexical, not semantic — a question phrased with
completely different vocabulary from the source text may retrieve poorly (e.g. "money
back for a client dinner" vs. "reimbursement"). The mock tools operate on fabricated
data, not a real HRMS/ITSM system. The LLM's answers are only as good as the context
retrieved — if retrieval misses the right chunk, the answer will reflect that gap.

---

## 6. Setup Instructions

### Prerequisites
- Python 3.10+ (or Docker)
- An Anthropic API key (https://console.anthropic.com/)

### Local setup
```bash
git clone <your-repo-url>
cd helpdesk-assistant
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run locally
```bash
cd src
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

### Run with Docker
```bash
docker build -t helpdesk-assistant .
docker run -p 8000:8000 --env-file .env helpdesk-assistant
```

---

## 7. API Usage

### Health check
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of sick leave do I get?"}'
```
```json
{
  "answer": "You get 10 days of paid sick leave per year...",
  "sources": [
    {"source_file": "leave_policy.md", "section_title": "Sick Leave", "score": 0.71}
  ],
  "actions_taken": []
}
```

### A question that triggers a tool call
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is my current leave balance?", "employee_id": "E1001"}'
```
The agent will call `check_leave_balance`, and `actions_taken` in the response will
show the tool call and its result.

---

## 8. Testing

```bash
pytest tests/ -v
```
Covers: health/root endpoints, input validation (empty/missing/overlong questions),
graceful failure when the API key is missing, data pipeline correctness (cleaning,
validation, chunking), and retrieval quality (relevant queries return matches,
out-of-domain queries don't force a false match).

---

## 9. Limitations

- TF-IDF retrieval is lexical; a larger or more varied document set would likely need
  semantic embeddings instead (see `DECISION_LOG.md`).
- The two tools (`raise_ticket`, `check_leave_balance`) use mock/local data, not real
  HR/IT systems.
- No authentication/authorization layer — not production-ready for handling real
  employee data as-is.
- No conversation memory across requests — each `/ask` call is independent.

## 10. Future Improvements

- Swap TF-IDF for dense embeddings + a proper vector DB (e.g. Chroma) if the document
  set grows large or varied.
- Add multi-turn conversation support (session/thread IDs).
- Add authentication and per-employee scoping for real deployment.
- Add a lightweight web frontend (Streamlit) in addition to the API.
- Add response caching for frequently asked questions to reduce LLM cost/latency.
