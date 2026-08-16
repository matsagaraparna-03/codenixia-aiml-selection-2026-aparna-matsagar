"""
Mock backend "systems" that the agent can act on (Milestone 7).

In a real company these would call the actual HR/ITSM APIs (e.g. Workday, ServiceNow).
Here they are simulated with a local JSON file so the whole project runs standalone
with no external accounts or credentials required to test the agent behavior.
"""

from __future__ import annotations
import os
import json
import uuid
import logging
from datetime import datetime

logger = logging.getLogger("tools")

HERE = os.path.dirname(os.path.abspath(__file__))
TICKETS_PATH = os.path.join(HERE, "..", "data", "tickets.json")

# Mock employee leave balances - stands in for an HRMS lookup
MOCK_LEAVE_BALANCES = {
    "E1001": {"name": "Aisha Khan", "annual_leave_days": 12.5, "sick_leave_days": 8},
    "E1002": {"name": "Rohan Mehta", "annual_leave_days": 4.0, "sick_leave_days": 10},
    "E1003": {"name": "Priya Nair", "annual_leave_days": 18.0, "sick_leave_days": 6},
}


def _load_tickets() -> list:
    if not os.path.exists(TICKETS_PATH):
        return []
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_tickets(tickets: list) -> None:
    os.makedirs(os.path.dirname(TICKETS_PATH), exist_ok=True)
    with open(TICKETS_PATH, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2)


def raise_ticket(issue_summary: str, category: str, employee_id: str = "UNKNOWN") -> dict:
    """Create a support ticket. Returns the created ticket record."""
    if not issue_summary or not issue_summary.strip():
        raise ValueError("issue_summary cannot be empty")

    ticket = {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "employee_id": employee_id,
        "category": category,
        "issue_summary": issue_summary.strip(),
        "status": "open",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    tickets = _load_tickets()
    tickets.append(ticket)
    _save_tickets(tickets)
    logger.info("Raised ticket %s for employee %s", ticket["ticket_id"], employee_id)
    return ticket


def check_leave_balance(employee_id: str) -> dict:
    """Look up an employee's current leave balance (mock HRMS lookup)."""
    record = MOCK_LEAVE_BALANCES.get(employee_id)
    if record is None:
        return {"found": False, "employee_id": employee_id,
                "message": "No record found for this employee ID."}
    return {"found": True, "employee_id": employee_id, **record}


# --- Tool schema exposed to the LLM (Anthropic tool-calling format) ---

TOOL_DEFINITIONS = [
    {
        "name": "raise_ticket",
        "description": (
            "Create a support ticket for an issue that the knowledge base cannot resolve, "
            "or that requires a human/system action (e.g. VPN not connecting after "
            "troubleshooting, a reimbursement stuck in approval, a benefits enrollment error). "
            "Only use this when the user's issue is a genuine problem needing follow-up, "
            "not for general policy questions that can be answered directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "issue_summary": {"type": "string", "description": "Short summary of the issue"},
                "category": {"type": "string", "enum": ["IT", "HR", "Finance"],
                             "description": "Which team should handle this ticket"},
                "employee_id": {"type": "string", "description": "Employee ID if known, else 'UNKNOWN'"},
            },
            "required": ["issue_summary", "category"],
        },
    },
    {
        "name": "check_leave_balance",
        "description": (
            "Look up an employee's current annual and sick leave balance by employee ID. "
            "Use this only when the user provides or asks about a specific employee ID's "
            "current balance - not for general leave policy questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "Employee ID, e.g. E1001"},
            },
            "required": ["employee_id"],
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """Dispatch a tool call by name. Raises ValueError for unknown tools."""
    if tool_name == "raise_ticket":
        return raise_ticket(**tool_input)
    if tool_name == "check_leave_balance":
        return check_leave_balance(**tool_input)
    raise ValueError(f"Unknown tool: {tool_name}")
