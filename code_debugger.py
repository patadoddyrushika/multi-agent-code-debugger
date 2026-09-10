import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# =========================
# API / MODEL SETUP
# =========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


# =========================
# DEBUG STATE
# =========================

class DebugState(TypedDict):
    code: str
    analysis: str
    fixed_code: str
    test_output: str
    review: str
    attempts: int
    final_answer: str


MAX_CODE_CHARS = 12000
MAX_OUTPUT_CHARS = 4000
MAX_ATTEMPTS = 2


# =========================
# HELPERS
# =========================

def get_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            str(x.get("text", ""))
            for x in content
            if isinstance(x, dict)
        ).strip()

    return str(content).strip()


def clean_code(text):
    lines = text.strip().splitlines()

    return "\n".join(
        line for line in lines
        if not line.strip().startswith("```")
    ).strip()


def run_python_code(code):
    code = code[:MAX_CODE_CHARS]

    blocked = [
        "os.system(",
        "subprocess.",
        "shutil.rmtree",
        "socket.",
        "urllib.request"
    ]

    if any(item.lower() in code.lower() for item in blocked):
        return "Execution blocked: potentially unsafe operation.", True

    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}", True

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "user_code.py"
        file_path.write_text(code, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=temp_dir
            )

            output = (
                (result.stdout or "") +
                (result.stderr or "")
            ).strip()[:MAX_OUTPUT_CHARS]

            return (
                output or "Program finished with no output.",
                result.returncode != 0
            )

        except subprocess.TimeoutExpired:
            return "Execution timed out after 5 seconds.", True

        except Exception as e:
            return f"Execution error: {e}", True


# =========================
# AGENT 1 — ANALYZER
# =========================

def analyzer_node(state):

    prompt = f"""
Analyze this Python code and identify the most likely bug.

CODE:
{state["code"]}

Give a short explanation of what is wrong and how it should be fixed.
"""

    response = llm.invoke(prompt)

    return {
        "analysis": get_text(response)
    }


# =========================
# AGENT 2 — FIXER
# =========================

def fixer_node(state):

    prompt = f"""
Fix the Python code below.

ORIGINAL CODE:
{state["code"]}

BUG ANALYSIS:
{state["analysis"]}

Return ONLY the complete corrected Python code.
Do not include markdown or explanations.
"""

    response = llm.invoke(prompt)

    return {
        "fixed_code": clean_code(get_text(response)),
        "attempts": state.get("attempts", 0) + 1
    }


# =========================
# AGENT 3 — TESTER
# =========================

def tester_node(state):

    output, failed = run_python_code(
        state["fixed_code"]
    )

    return {
        "test_output": output
    }


# =========================
# AGENT 4 — REVIEWER
# =========================

def reviewer_node(state):

    prompt = f"""
Review this Python debugging attempt.

ORIGINAL CODE:
{state["code"]}

FIXED CODE:
{state["fixed_code"]}

TEST OUTPUT:
{state["test_output"]}

Start your response with exactly PASS or RETRY.

Then give one short reason.
"""

    response = llm.invoke(prompt)

    return {
        "review": get_text(response)
    }


# =========================
# ROUTING
# =========================

def route_after_review(state):

    review = state.get("review", "").upper()

    if review.startswith("PASS"):
        return "final"

    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "final"

    return "retry"


# =========================
# AGENT 5 — FINALIZER
# =========================

def finalizer_node(state):

    prompt = f"""
Give the final result of this debugging task.

ORIGINAL CODE:
{state["code"]}

FIXED CODE:
{state["fixed_code"]}

TEST OUTPUT:
{state["test_output"]}

REVIEW:
{state["review"]}

Briefly explain:
- What the bug was
- What was fixed
- Whether the test passed

Then provide the final corrected code.
Do not claim success if the test failed.
"""

    response = llm.invoke(prompt)

    return {
        "final_answer": get_text(response)
    }


# =========================
# LANGGRAPH WORKFLOW
# =========================

graph = StateGraph(DebugState)

graph.add_node("analyzer", analyzer_node)
graph.add_node("fixer", fixer_node)
graph.add_node("tester", tester_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("finalizer", finalizer_node)

graph.add_edge(START, "analyzer")
graph.add_edge("analyzer", "fixer")
graph.add_edge("fixer", "tester")
graph.add_edge("tester", "reviewer")

graph.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "retry": "analyzer",
        "final": "finalizer"
    }
)

graph.add_edge("finalizer", END)

debug_graph = graph.compile()
