"""Tool-using RAG agent orchestrated as a LangGraph state machine.

Flow::

    sanitize -> plan -+-> act -+-> grade -+-> plan ...
                      |        |          +-> rewrite -> act
                      |        +-> plan / generate
                      +-> generate -> END

* ``sanitize``  redacts PII from the question before any model sees it.
* ``plan``      chooses the next tool (or "answer"); an LLM decides when one is configured,
                otherwise simple rules do.
* ``act``       runs the chosen tool.
* ``grade``     checks that retrieved context is relevant to the question.
* ``rewrite``   reformulates the search query once if the context was not relevant.
* ``generate``  writes a grounded, cited answer, or says it does not know.

The node functions and the routing functions are plain Python, so the same flow runs on
LangGraph (default) or on a minimal local executor (``AGENT_ENGINE=local``).
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypedDict

from .config import Settings
from .llm import ChatLLM
from .models import Hit
from .privacy import redact
from .retrieval import KnowledgeBase
from .text import tokenize
from .tools import CALCULATOR_TOOL, DATE_TOOL, SEARCH_TOOL, build_tools, extract_expression

logger = logging.getLogger(__name__)

END = "__end__"
ANSWER = "answer"
NOT_FOUND = "I couldn't find that in the knowledge base."

_DATE_RE = re.compile(r"\b(today'?s date|current date|what(?:'s| is) the date|what day is it)\b", re.I)


class AgentState(TypedDict, total=False):
    raw_question: str
    question: str  # sanitised question
    search_query: str
    redactions: dict
    observations: list  # [{"tool", "input", "output"}]
    pending: dict  # next action: {"tool", "input"}
    last_tool: str
    hits: list
    relevant: bool
    rewrites: int
    steps: int
    answer: str
    sources: list
    trace: list


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    engine: str = "local"


# --- helpers (pure functions, unit-tested) -------------------------------------------
def parse_decision(text: str, tool_names: set[str]) -> Optional[dict]:
    """Parse ``{"tool": ..., "input": ...}`` out of model output; ``None`` if invalid."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    tool = str(data.get("tool", "")).strip()
    if tool != ANSWER and tool not in tool_names:
        return None
    return {"tool": tool, "input": str(data.get("input", "")).strip()}


def grade_hits(
    informative_terms: set[str], hits: list[Hit], min_coverage: float, min_dense: float
) -> tuple[bool, float, float]:
    """Relevance gate.

    A question is answerable from the knowledge base if one of the top hits contains enough of the
    question's *informative* terms (see ``KnowledgeBase.informative_terms``), or if its dense
    similarity is high enough. Returns ``(relevant, term_coverage, max_dense_similarity)``.
    """
    if not hits:
        return False, 0.0, 0.0
    coverage = 0.0
    if informative_terms:
        coverage = max(len(informative_terms & set(tokenize(h.chunk.text))) / len(informative_terms) for h in hits[:3])
    max_dense = max(h.dense for h in hits)
    return (coverage >= min_coverage or max_dense >= min_dense), coverage, max_dense


def extractive_answer(
    question: str,
    hits: list[Hit],
    max_sentences: int = 3,
    weight: Optional[Callable[[str], float]] = None,
) -> str:
    """Offline answer: the retrieved sentences that best match the question.

    Sentences are scored by the summed ``weight`` (e.g. IDF) of the question terms they contain, so
    rare, topical words count for more than common ones. Weak matches are dropped.
    """
    weight = weight or (lambda _t: 1.0)
    q_terms = set(tokenize(question))
    scored = []
    seen: set[str] = set()  # overlapping chunks repeat their boundary sentences
    for hit_idx, hit in enumerate(hits):
        for sent_idx, raw in enumerate(re.split(r"(?<=[.!?])\s+|\n+", hit.chunk.text)):
            if raw.lstrip().startswith("#"):  # skip markdown headings
                continue
            sentence = raw.strip().lstrip("-* ").strip()
            if len(sentence) < 20 or sentence in seen:
                continue
            seen.add(sentence)
            score = sum(weight(t) for t in q_terms & set(tokenize(sentence)))
            if score > 0:
                scored.append((score, -hit_idx, -sent_idx, hit_idx, sent_idx, sentence))
    if not scored:
        return NOT_FOUND
    ranked = sorted(scored, reverse=True)
    cutoff = 0.6 * ranked[0][0]
    best = [s for s in ranked if s[0] >= cutoff][:max_sentences]
    best.sort(key=lambda s: (s[3], s[4]))
    return "\n".join(f"- {s[5]} [{s[3] + 1}]" for s in best)


def rewrite_query(question: str) -> str:
    """Keyword-only reformulation used when no LLM is configured."""
    return " ".join(tokenize(question)) or question


# --- the agent ---------------------------------------------------------------------
class RagAgent:
    def __init__(self, kb: KnowledgeBase, settings: Settings, llm: Optional[ChatLLM] = None) -> None:
        self.kb, self.settings, self.llm = kb, settings, llm
        self.tools = build_tools(kb, settings)
        self._graph = None
        self.nodes = {
            "sanitize": self.node_sanitize,
            "plan": self.node_plan,
            "act": self.node_act,
            "grade": self.node_grade,
            "rewrite": self.node_rewrite,
            "generate": self.node_generate,
        }
        # name -> next node, or (router, possible targets) for conditional edges
        self.flow: dict[str, Any] = {
            "sanitize": "plan",
            "plan": (self.route_after_plan, ("act", "generate")),
            "act": (self.route_after_act, ("grade", "plan", "generate")),
            "grade": (self.route_after_grade, ("plan", "rewrite")),
            "rewrite": "act",
            "generate": END,
        }

    # -- nodes ----------------------------------------------------------------------
    def node_sanitize(self, state: AgentState) -> dict:
        raw = state["raw_question"]
        result = redact(raw) if self.settings.redact_pii else None
        question = result.text if result else raw
        counts = result.counts if result else {}
        note = f"sanitize: redacted {sum(counts.values())} item(s) {counts}" if counts else "sanitize: nothing to redact"
        return {"question": question, "search_query": question, "redactions": counts, "trace": state.get("trace", []) + [note]}

    def node_plan(self, state: AgentState) -> dict:
        observations = state.get("observations", [])
        question = state["question"]
        if state.get("steps", 0) >= self.settings.max_agent_steps:
            decision = {"tool": ANSWER, "input": ""}
            note = "plan: step budget reached -> answer"
        else:
            decision = self._plan_with_llm(question, observations) if self.llm else None
            source = "llm" if decision else "rules"
            decision = decision or self._plan_with_rules(question, observations)
            if decision["tool"] == ANSWER and not observations:  # always ground the answer
                decision = self._plan_with_rules(question, observations)
            if any(o["tool"] == decision["tool"] and o["input"] == decision["input"] for o in observations):
                decision = {"tool": ANSWER, "input": ""}  # never repeat an identical call
            note = f"plan[{source}]: {decision['tool']}({decision['input'][:60]!r})" if decision["tool"] != ANSWER else f"plan[{source}]: answer"
        return {"pending": decision, "trace": state.get("trace", []) + [note]}

    def node_act(self, state: AgentState) -> dict:
        action = state["pending"]
        tool = self.tools[action["tool"]]
        result = tool.run(action["input"])
        update: dict[str, Any] = {
            "observations": state.get("observations", []) + [{"tool": tool.name, "input": action["input"], "output": result.observation}],
            "last_tool": tool.name,
            "steps": state.get("steps", 0) + 1,
            "trace": state.get("trace", []) + [f"act: {tool.name} -> {result.observation.splitlines()[0][:80] if result.observation else ''}"],
        }
        if tool.name == SEARCH_TOOL:
            update["hits"] = result.hits
            update["search_query"] = action["input"]
        return update

    def node_grade(self, state: AgentState) -> dict:
        informative = self.kb.informative_terms(set(tokenize(state["question"])))
        ok, coverage, dense = grade_hits(
            informative, state.get("hits", []), self.settings.min_coverage, self.settings.min_dense
        )
        note = f"grade: relevant={ok} (term coverage={coverage:.2f}, max dense={dense:.2f})"
        return {"relevant": ok, "trace": state.get("trace", []) + [note]}

    def node_rewrite(self, state: AgentState) -> dict:
        rewritten = self._rewrite_with_llm(state["question"]) if self.llm else None
        rewritten = rewritten or rewrite_query(state["question"])
        return {
            "pending": {"tool": SEARCH_TOOL, "input": rewritten},
            "rewrites": state.get("rewrites", 0) + 1,
            "trace": state.get("trace", []) + [f"rewrite: {rewritten[:80]!r}"],
        }

    def node_generate(self, state: AgentState) -> dict:
        question = state["question"]
        observations = state.get("observations", [])
        hits = self._select_context(question, state.get("hits", [])) if state.get("relevant") else []
        used_search = any(o["tool"] == SEARCH_TOOL for o in observations)
        other = [o for o in observations if o["tool"] != SEARCH_TOOL]
        notes = list(state.get("trace", []))

        if used_search and not hits and not other:
            answer = NOT_FOUND
        else:
            answer = None
            if self.llm:
                try:
                    answer = self.llm.complete(*self._generation_prompts(question, hits, other))
                    notes.append("generate: answered with the LLM")
                except Exception as exc:  # network/API failure should not break the agent
                    logger.warning("LLM generation failed (%s); using extractive answer.", exc)
                    notes.append("generate: LLM call failed, used extractive answer")
            if not answer:
                parts = []
                if hits:
                    parts.append(extractive_answer(question, hits, weight=self.kb.idf))
                parts.extend(f"{o['output']}" for o in other)
                answer = "\n".join(p for p in parts if p) or NOT_FOUND
                notes.append("generate: extractive answer")
        if self.settings.redact_pii:
            answer = redact(answer).text
        sources = [f"[{i}] {h.chunk.doc_id} (chunk {h.chunk.index})" for i, h in enumerate(hits, start=1)]
        return {"answer": answer, "sources": sources, "trace": notes}

    def _select_context(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Keep only hits that share an informative term with the question (or embed close to it)."""
        informative = self.kb.informative_terms(set(tokenize(question)))
        keep = [h for h in hits if h.dense >= self.settings.min_dense or informative & set(tokenize(h.chunk.text))]
        return keep or hits[:1]

    # -- routing --------------------------------------------------------------------
    def route_after_plan(self, state: AgentState) -> str:
        return "generate" if state["pending"]["tool"] == ANSWER else "act"

    def route_after_act(self, state: AgentState) -> str:
        if state.get("steps", 0) >= self.settings.max_agent_steps:
            return "generate"
        return "grade" if state.get("last_tool") == SEARCH_TOOL else "plan"

    def route_after_grade(self, state: AgentState) -> str:
        if state.get("relevant") or state.get("rewrites", 0) >= 1:
            return "plan"
        return "rewrite"

    # -- planning / prompting -------------------------------------------------------
    def _plan_with_rules(self, question: str, observations: list) -> dict:
        if observations:
            return {"tool": ANSWER, "input": ""}
        expression = extract_expression(question)
        if expression:
            return {"tool": CALCULATOR_TOOL, "input": expression}
        if _DATE_RE.search(question):
            return {"tool": DATE_TOOL, "input": ""}
        return {"tool": SEARCH_TOOL, "input": question}

    def _plan_with_llm(self, question: str, observations: list) -> Optional[dict]:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        system = (
            "You plan the next step for a knowledge-base assistant.\n"
            f"Tools:\n{tool_lines}\n"
            'Reply with ONLY a JSON object: {"tool": "<tool name or answer>", "input": "<tool input>"}.\n'
            'Use "answer" once the observations are enough to answer the question.'
        )
        seen = "\n".join(f"{o['tool']}({o['input']!r}) -> {o['output'][:400]}" for o in observations) or "(none yet)"
        try:
            reply = self.llm.complete(system, f"Question: {question}\n\nObservations so far:\n{seen}")
        except Exception as exc:
            logger.warning("LLM planning failed (%s); falling back to rules.", exc)
            return None
        return parse_decision(reply, set(self.tools))

    def _rewrite_with_llm(self, question: str) -> Optional[str]:
        try:
            reply = self.llm.complete(
                "Rewrite the question as a short keyword search query. Reply with the query only.", question
            )
        except Exception:
            return None
        return reply.strip().strip('"') or None

    @staticmethod
    def _generation_prompts(question: str, hits: list[Hit], other: list) -> tuple[str, str]:
        system = (
            "Answer the question using ONLY the context and tool results provided. "
            "Cite context passages like [1]. If they are insufficient, say you don't know."
        )
        context = "\n\n".join(f"[{i}] ({h.chunk.doc_id})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)) or "(none)"
        tool_text = "\n".join(o["output"] for o in other) or "(none)"
        return system, f"Context:\n{context}\n\nTool results:\n{tool_text}\n\nQuestion: {question}"

    # -- execution ------------------------------------------------------------------
    def build_graph(self):
        """Compile the flow into a LangGraph ``StateGraph``."""
        from langgraph.graph import END as LG_END
        from langgraph.graph import START, StateGraph

        graph = StateGraph(AgentState)
        for name, fn in self.nodes.items():
            graph.add_node(name, fn)
        graph.add_edge(START, "sanitize")
        for name, nxt in self.flow.items():
            if isinstance(nxt, str):
                graph.add_edge(name, LG_END if nxt == END else nxt)
            else:
                router, targets = nxt
                graph.add_conditional_edges(name, router, {t: t for t in targets})
        return graph.compile()

    def _run_local(self, state: AgentState) -> AgentState:
        node = "sanitize"
        for _ in range(50):
            state.update(self.nodes[node](state))
            nxt = self.flow[node]
            node = nxt if isinstance(nxt, str) else nxt[0](state)
            if node == END:
                return state
        raise RuntimeError("Agent exceeded its step limit")

    def ask(self, question: str) -> AgentResult:
        state: AgentState = {"raw_question": question, "observations": [], "steps": 0, "rewrites": 0, "trace": [], "hits": []}
        engine = self.settings.agent_engine.lower()
        if engine == "langgraph":
            try:
                if self._graph is None:
                    self._graph = self.build_graph()
            except ImportError:
                logger.warning("langgraph is not installed; running the flow on the local executor.")
                engine = "local"
        final = self._graph.invoke(state) if engine == "langgraph" else self._run_local(state)
        return AgentResult(
            answer=final["answer"],
            sources=final.get("sources", []),
            trace=final.get("trace", []),
            redactions=final.get("redactions", {}),
            tool_calls=[(o["tool"], o["input"]) for o in final.get("observations", [])],
            engine=engine,
        )
