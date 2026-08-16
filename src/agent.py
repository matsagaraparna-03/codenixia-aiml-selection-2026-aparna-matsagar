"""
Agent orchestration layer (Milestone 5 + 7).

Uses Google Gemini (free-tier API, no credit card required) instead of a paid LLM API,
since this project needs to run at zero cost for testing/grading. See DECISION_LOG.md
for the reasoning.

Why an agent here (not just a plain RAG chatbot):
Some employee questions are answerable purely from policy documents ("how many sick
days do I get?" -> RAG). Others require the system to actually DO something the
documents can't answer on their own - look up a specific employee's live leave balance,
or open a ticket when an issue needs human follow-up. An agent that can decide between
"answer from knowledge" and "call a tool" is what makes the assistant genuinely useful
instead of just a document search box.

Flow:
  User question -> retrieve top-k policy chunks (RAG) -> send question + context + tool
  definitions to Gemini -> Gemini either answers directly (grounded in the retrieved
  context) or calls a tool -> if a tool is called, we execute it locally and send the
  result back to Gemini for a final natural-language response.
"""

from __future__ import annotations
import os
import logging
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag import retrieve
from tools import TOOL_DEFINITIONS, execute_tool

load_dotenv()

logger = logging.getLogger("agent")

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

SYSTEM_PROMPT = """You are an internal HR/IT Helpdesk Assistant for a company.

You answer employee questions using ONLY the policy context provided to you below -
do not invent policy details that aren't in the context. If the context doesn't contain
the answer, say so honestly rather than guessing.

You have tools available for actions that go beyond answering from documents:
- check_leave_balance: use only when the user gives a specific employee ID and wants
  their current balance (not general policy questions).
- raise_ticket: use only when there's a genuine unresolved issue needing human/system
  follow-up, not for questions you can already answer from the policy context.

Always cite which policy document your answer is based on when you use the context.
Keep answers concise and practical.
"""


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return genai.Client(api_key=api_key)


def _gemini_tool() -> types.Tool:
    """Convert our provider-agnostic TOOL_DEFINITIONS (in tools.py) into Gemini's
    FunctionDeclaration format."""
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters_json_schema=t["input_schema"],
        )
        for t in TOOL_DEFINITIONS
    ]
    return types.Tool(function_declarations=declarations)


def _format_context(chunks: List[dict]) -> str:
    if not chunks:
        return "(No relevant policy context was found for this question.)"
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source_file']} - {c['section_title']}]\n{c['text']}")
    return "\n\n".join(parts)


def answer_question(question: str, employee_id: str = "UNKNOWN", top_k: int = 3) -> dict:
    """
    Runs the full agent loop for a single question. Returns a dict with the final
    answer text, the sources used, and any actions (tool calls) taken.
    """
    if not question or not question.strip():
        raise ValueError("question cannot be empty")

    client = _client()
    tool = _gemini_tool()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tool],
    )

    retrieved_chunks = retrieve(question, top_k=top_k)
    context_text = _format_context(retrieved_chunks)

    user_message = (
        f"Employee ID (if relevant): {employee_id}\n\n"
        f"Relevant policy context:\n{context_text}\n\n"
        f"Employee question: {question}"
    )

    contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]
    actions_taken = []

    response = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)

    max_hops = 3
    hops = 0
    while response.function_calls and hops < max_hops:
        hops += 1
        # Append the model's turn (which contains the function_call part) to history
        contents.append(response.candidates[0].content)

        function_response_parts = []
        for fc in response.function_calls:
            logger.info("Agent calling tool: %s(%s)", fc.name, fc.args)
            try:
                result = execute_tool(fc.name, dict(fc.args))
                actions_taken.append({"tool": fc.name, "input": dict(fc.args), "result": result})
                function_response_parts.append(
                    types.Part.from_function_response(name=fc.name, response={"result": result})
                )
            except Exception as e:
                logger.exception("Tool execution failed: %s", fc.name)
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name, response={"error": str(e)}
                    )
                )

        contents.append(types.Content(role="user", parts=function_response_parts))
        response = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)

    final_text = (response.text or "").strip()

    return {
        "answer": final_text,
        "sources": [
            {"source_file": c["source_file"], "section_title": c["section_title"], "score": c["score"]}
            for c in retrieved_chunks
        ],
        "actions_taken": actions_taken,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = answer_question("How many days of sick leave do employees get?")
    print(result)