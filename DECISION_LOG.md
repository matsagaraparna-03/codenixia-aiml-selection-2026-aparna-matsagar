# Engineering Decision Log

---

**Decision:** Use TF-IDF (scikit-learn) for embeddings instead of a neural
sentence-embedding model.

**Reason:** The knowledge base is small (a few dozen chunks) and domain-specific
(HR/IT vocabulary is fairly consistent), which is the regime where sparse lexical
retrieval performs close to dense embeddings. TF-IDF also requires no model download
at runtime, making the app fully reproducible offline/in restricted networks (see
`DEBUGGING_REPORT.md`, Issue 1) and lighter to containerize.

**Alternative considered:** `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) with a
FAISS vector index.

**Why rejected (for now):** Adds a large dependency (PyTorch) and a runtime download
dependency on Hugging Face Hub, for a marginal retrieval-quality gain on a small,
lexically consistent corpus. This is documented as a clear upgrade path in
`README.md` — if the document set grows large or lexically diverse (e.g. multilingual
or highly paraphrased employee questions), dense embeddings would be worth the cost.

---

**Decision:** Use the Anthropic Messages API's native tool-calling for the agent,
rather than a dedicated agent framework (e.g. LangChain agents, CrewAI).

**Reason:** The agent only needs to choose between 2 tools and answering directly —
a native tool-use loop (call model → check `stop_reason` → execute tool → send result
back) is ~80 lines of plain Python and is easy to trace and explain end-to-end.

**Alternative considered:** LangChain's agent executor.

**Why rejected:** Would add a large dependency and an abstraction layer for a workflow
simple enough to write and fully understand directly. A framework becomes worth the
overhead once there are many tools, multi-agent coordination, or complex planning —
none of which apply here.

---

**Decision:** Use FastAPI for the API layer.

**Reason:** Built-in request validation via Pydantic (directly supports Milestone 10's
input-validation requirement), async support, and automatic OpenAPI docs at `/docs`
which are useful during the live technical defense.

**Alternative considered:** Flask.

**Why rejected:** Flask would need an add-on (e.g. Marshmallow/Pydantic integration) to
get the same request validation "for free," and lacks built-in async support.

---

**Decision:** Chunk documents by markdown section (`##` heading) rather than by a
fixed character/token window.

**Reason:** Section-based chunking keeps semantically related content together (e.g.
the entire "Sick Leave" section stays in one chunk), which produces more coherent
retrieval results than an arbitrary fixed-size window that might cut a policy
statement in half.

**Alternative considered:** Fixed-size sliding-window chunking (e.g. 500 characters
with 50-character overlap).

**Why rejected:** Would occasionally split a single policy rule across two chunks,
degrading retrieval and forcing the LLM to reason over incomplete context. Kept as a
documented fallback for any section that exceeds the max chunk size (see
`chunk_document()` in `data_pipeline.py`).

---

**Decision:** Use mock/local data for the agent's tools (`raise_ticket`,
`check_leave_balance`) instead of integrating a real HRMS/ITSM API.

**Reason:** No real company system or credentials are available for this project, and
the goal is to demonstrate the *agent decision-making pattern* (when to call a tool vs.
answer from documents), which doesn't depend on which backend the tool ultimately
calls.

**Alternative considered:** Integrate a free-tier ticketing API (e.g. a public
mock-API service).

**Why rejected:** Adds an external network dependency and account-setup overhead for
no benefit to what's actually being demonstrated. Swapping the mock functions in
`tools.py` for real API calls is a small, isolated change if this were extended.
