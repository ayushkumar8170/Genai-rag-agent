"""Tools the agent can call: knowledge-base search, a safe calculator, and the date."""

from __future__ import annotations

import ast
import datetime as dt
import operator
import re
from dataclasses import dataclass, field
from typing import Callable

from .config import Settings
from .models import Hit
from .retrieval import KnowledgeBase

SEARCH_TOOL = "search_knowledge_base"
CALCULATOR_TOOL = "calculator"
DATE_TOOL = "current_date"


@dataclass
class ToolResult:
    observation: str
    hits: list[Hit] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], ToolResult]


# --- calculator ---------------------------------------------------------------------
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST, depth: int = 0) -> float:
    if depth > 20:
        raise ValueError("expression too complex")
    if isinstance(node, ast.Expression):
        return _eval(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left, depth + 1), _eval(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ValueError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, depth + 1))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate arithmetic safely (no names, calls, or attribute access)."""
    expression = expression.strip().replace("^", "**").replace("×", "*").replace(",", "")
    if not expression or len(expression) > 200:
        return "Error: expression is empty or too long"
    try:
        value = _eval(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError, OverflowError) as exc:
        return f"Error: {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:.6g}" if isinstance(value, float) else str(value)


_EXPR_RE = re.compile(r"[\d.][\d\s+\-*/().^%×,]*[\d)]")
_OPERATOR_RE = re.compile(r"[+*/^×]|\s-\s")


def extract_expression(text: str) -> str | None:
    """Find an arithmetic expression such as ``12 * 29 + 3`` inside a sentence."""
    candidates = [m.group(0).strip() for m in _EXPR_RE.finditer(text) if _OPERATOR_RE.search(m.group(0))]
    return max(candidates, key=len) if candidates else None


# --- tool registry ------------------------------------------------------------------
def format_hits(hits: list[Hit], limit: int = 3, width: int = 220) -> str:
    lines = []
    for i, hit in enumerate(hits[:limit], start=1):
        snippet = " ".join(hit.chunk.text.split())
        lines.append(f"[{i}] {hit.chunk.doc_id}: {snippet[:width]}")
    return "\n".join(lines) if lines else "No results."


def build_tools(kb: KnowledgeBase, settings: Settings) -> dict[str, Tool]:
    def search(query: str) -> ToolResult:
        hits = kb.search(query, k=settings.top_k, mode="hybrid")
        return ToolResult(observation=format_hits(hits), hits=hits)

    def calc(expression: str) -> ToolResult:
        return ToolResult(observation=f"{expression.strip()} = {calculate(expression)}")

    def today(_: str) -> ToolResult:
        return ToolResult(observation=dt.date.today().isoformat())

    tools = [
        Tool(SEARCH_TOOL, "Search the company knowledge base. Input: a search query.", search),
        Tool(CALCULATOR_TOOL, "Evaluate an arithmetic expression. Input: e.g. '12 * 29 + 3'.", calc),
        Tool(DATE_TOOL, "Get today's date. Input: ignored.", today),
    ]
    return {t.name: t for t in tools}


def as_langchain_tools(tools: dict[str, Tool]):
    """Expose the registry as LangChain ``StructuredTool`` objects (for use with native tool calling)."""
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            func=lambda input, _t=t: _t.run(input).observation,
            name=t.name,
            description=t.description,
        )
        for t in tools.values()
    ]
