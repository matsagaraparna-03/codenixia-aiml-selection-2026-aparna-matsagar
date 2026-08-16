# AI Usage Disclosure

## AI Tools Used
- **Claude (Anthropic)** — used extensively for scaffolding this project's code,
  architecture, and documentation.

## Purpose / What Claude Was Used For
- Proposing the overall project idea (HR/IT helpdesk assistant) and mapping it to the
  10 required milestones.
- Writing the initial implementation of the data pipeline, RAG layer, agent/tool-calling
  logic, FastAPI application, tests, Dockerfile, and documentation files.
- Debugging a real issue encountered during development (Hugging Face model download
  failing in a network-restricted environment) and implementing the fix (switching to
  TF-IDF embeddings).
- Identifying and fixing a retrieval-quality bug (title-only chunks polluting search
  results).

## Important — Fill This Section In Yourself Before Submitting

This file, and this codebase, were produced with heavy AI assistance. The challenge
explicitly requires that **you understand what you submit** and may be asked to
explain, debug, or modify it live without AI help. Before submitting, you should:

1. **Read every file in `src/`** end to end and make sure you can explain, in your own
   words, what each function does and why.
2. **Actually run the project yourself** — install dependencies, get your own Anthropic
   API key, run `pytest`, hit the `/ask` endpoint, and build the Docker image. Fix
   anything that doesn't work in your environment.
3. **Make at least a few real changes of your own** — e.g. add a new policy document,
   add a third tool, tweak the chunking logic, add another test — and note them below.
   This is both good practice for the live defense and genuinely strengthens your
   understanding.
4. **Replace this paragraph** with your own honest account of:
   - Which parts you modified or rewrote yourself
   - Which parts you understand fully vs. still need to review
   - Any bugs *you* found and fixed beyond the two documented in `DEBUGGING_REPORT.md`

## Major AI Assistance Received
- Full initial implementation of all source files.
- Architecture and technology decisions (documented with reasoning in
  `DECISION_LOG.md` so they can be defended, not just cited).

## Changes Made By the Student
*(Replace this with your own entries before submitting.)*
- [ ] Reviewed and understood `data_pipeline.py`
- [ ] Reviewed and understood `rag.py`
- [ ] Reviewed and understood `tools.py` and `agent.py`
- [ ] Reviewed and understood `main.py`
- [ ] Ran the project end-to-end locally with my own API key
- [ ] Ran the project in Docker
- [ ] Made at least one meaningful code change of my own: ___________
