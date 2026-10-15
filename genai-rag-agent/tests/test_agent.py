import importlib.util
import unittest

from rag_agent.agent import (
    ANSWER,
    NOT_FOUND,
    RagAgent,
    extractive_answer,
    grade_hits,
    parse_decision,
)
from rag_agent.text import tokenize
from rag_agent.tools import calculate, extract_expression
from tests.helpers import FakeLLM, build_kb, make_settings

TOOLS = {"search_knowledge_base", "calculator", "current_date"}


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculate("2 + 3 * 4"), "14")
        self.assertEqual(calculate("(12 * 29) + 3"), "351")
        self.assertEqual(calculate("2 ^ 10"), "1024")
        self.assertEqual(calculate("10 / 4"), "2.5")
        self.assertEqual(calculate("1,200 * 2"), "2400")

    def test_rejects_code_and_abuse(self):
        for bad in ("__import__('os').system('ls')", "open('x')", "9 ** 9 ** 9", "1/0", "", "a + 1"):
            self.assertTrue(calculate(bad).startswith("Error"), bad)

    def test_expression_extraction(self):
        self.assertEqual(extract_expression("What is 12 * 29 + 3?"), "12 * 29 + 3")
        self.assertIsNone(extract_expression("What happened on 2026-10-15?"))
        self.assertIsNone(extract_expression("Do you offer 99.9% uptime?"))


class DecisionParsingTests(unittest.TestCase):
    def test_valid_and_invalid(self):
        self.assertEqual(
            parse_decision('Sure! {"tool": "calculator", "input": "2+2"}', TOOLS), {"tool": "calculator", "input": "2+2"}
        )
        self.assertEqual(parse_decision('{"tool": "answer"}', TOOLS), {"tool": ANSWER, "input": ""})
        self.assertIsNone(parse_decision('{"tool": "rm_rf", "input": "/"}', TOOLS))
        self.assertIsNone(parse_decision("no json here", TOOLS))
        self.assertIsNone(parse_decision("{broken", TOOLS))


class GradingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = build_kb()
        cls.settings = make_settings()

    def _grade(self, q):
        hits = self.kb.search(q, k=4)
        informative = self.kb.informative_terms(set(tokenize(q)))
        return grade_hits(informative, hits, self.settings.min_coverage, self.settings.min_dense)[0], hits

    def test_in_scope_question_is_relevant(self):
        ok, _ = self._grade("How long does a refund take?")
        self.assertTrue(ok)

    def test_out_of_scope_question_is_not_relevant(self):
        ok, _ = self._grade("Who won the football world cup in 2018?")
        self.assertFalse(ok)

    def test_mentioning_the_product_name_alone_is_not_enough(self):
        ok, _ = self._grade("Who is the CEO of NimbusDesk?")
        self.assertFalse(ok)

    def test_informative_terms_ignore_unknown_and_ubiquitous_words(self):
        terms = self.kb.informative_terms({"refund", "football", "nimbusdesk"})
        self.assertEqual(terms, {"refund"})

    def test_extractive_answer_does_not_repeat_sentences_from_overlapping_chunks(self):
        q = "What credit do I get if uptime drops below 99%?"
        lines = extractive_answer(q, self.kb.search(q, k=4)).splitlines()
        sentences = [line.rsplit(" [", 1)[0] for line in lines]
        self.assertEqual(len(sentences), len(set(sentences)))

    def test_extractive_answer_cites_sources(self):
        q = "How long does a refund take?"
        answer = extractive_answer(q, self.kb.search(q, k=4))
        self.assertIn("5 to 7 business days", answer)
        self.assertRegex(answer, r"\[\d\]")


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = build_kb()

    def agent(self, llm=None, **settings):
        return RagAgent(self.kb, make_settings(**settings), llm)

    def test_grounded_answer_with_sources(self):
        result = self.agent().ask("What is the uptime guarantee for the Business plan?")
        self.assertIn("99.9%", result.answer)
        self.assertTrue(any("sla_and_uptime.md" in s for s in result.sources))
        self.assertEqual(result.tool_calls[0][0], "search_knowledge_base")

    def test_unknown_topic_says_so_instead_of_guessing(self):
        result = self.agent().ask("Who won the football world cup in 2018?")
        self.assertEqual(result.answer, NOT_FOUND)
        self.assertEqual(result.sources, [])
        self.assertTrue(any(line.startswith("rewrite:") for line in result.trace))  # retried once

    def test_calculator_tool_is_used_for_arithmetic(self):
        result = self.agent().ask("What is 12 * 29 + 3?")
        self.assertEqual(result.tool_calls, [("calculator", "12 * 29 + 3")])
        self.assertIn("351", result.answer)

    def test_date_tool(self):
        result = self.agent().ask("What is the current date?")
        self.assertEqual(result.tool_calls[0][0], "current_date")
        self.assertRegex(result.answer, r"\d{4}-\d{2}-\d{2}")

    def test_pii_in_question_is_redacted_before_planning_and_answers(self):
        result = self.agent().ask("I'm jane@example.com, how long does a refund take?")
        self.assertEqual(result.redactions, {"EMAIL": 1})
        self.assertNotIn("jane@example.com", " ".join(result.tool_calls[0]))
        self.assertNotIn("jane@example.com", " ".join(result.trace))

    def test_redaction_can_be_disabled(self):
        result = self.agent(redact_pii=False).ask("I'm jane@example.com, how long does a refund take?")
        self.assertEqual(result.redactions, {})

    def test_llm_plans_calls_tools_and_answers_with_context(self):
        llm = FakeLLM(
            [
                '{"tool": "search_knowledge_base", "input": "Business plan price per seat"}',
                '{"tool": "calculator", "input": "12 * 29"}',
                '{"tool": "answer", "input": ""}',
                "12 seats on Business cost $348 per month [1].",
            ]
        )
        result = self.agent(llm).ask("How much would 12 Business seats cost per month?")
        self.assertEqual([t for t, _ in result.tool_calls], ["search_knowledge_base", "calculator"])
        self.assertEqual(result.answer, "12 seats on Business cost $348 per month [1].")
        generation_prompt = llm.calls[-1][1]
        self.assertIn("12 * 29 = 348", generation_prompt)  # tool result reached the generator
        self.assertIn("$29 per seat per month", generation_prompt)  # retrieved context did too

    def test_llm_cannot_skip_grounding_or_loop_forever(self):
        llm = FakeLLM(['{"tool": "answer", "input": ""}'] + ['{"tool": "answer", "input": ""}'] * 5 + ["ok [1]"])
        result = self.agent(llm).ask("How often is the status page updated during an incident?")
        self.assertEqual(result.tool_calls[0][0], "search_knowledge_base")
        llm2 = FakeLLM(['{"tool": "calculator", "input": "1+1"}'] * 10)
        result2 = self.agent(llm2, max_agent_steps=3).ask("How long does a refund take?")
        self.assertLessEqual(len(result2.tool_calls), 3)

    def test_garbage_planner_output_falls_back_to_rules(self):
        llm = FakeLLM(["I think you should search!", "still not json", "Refunds take 5 to 7 business days [1]."])
        result = self.agent(llm).ask("How long does a refund take?")
        self.assertEqual(result.tool_calls[0][0], "search_knowledge_base")

    def test_llm_failure_degrades_to_extractive_answer(self):
        class Broken:
            name = "broken"

            def complete(self, system, user):
                raise ConnectionError("network down")

        result = self.agent(Broken()).ask("How long does a refund take?")
        self.assertIn("5 to 7 business days", result.answer)

    def test_llm_never_sees_raw_pii(self):
        llm = FakeLLM(['{"tool": "answer"}', "fine"])
        self.agent(llm).ask("My card is 4111 1111 1111 1111, how long does a refund take?")
        prompts = " ".join(system + user for system, user in llm.calls)
        self.assertNotIn("4111", prompts)

    def test_answer_is_redacted_if_a_model_leaks_pii(self):
        llm = FakeLLM(['{"tool": "answer"}', "Email leak@example.com for help [1]."])
        result = self.agent(llm).ask("How long does a refund take?")
        self.assertNotIn("leak@example.com", result.answer)


class _StubStateGraph:
    """Minimal stand-in for ``langgraph.graph.StateGraph`` that records wiring and can execute it."""

    def __init__(self, schema):
        self.nodes, self.edges, self.conditional = {}, {}, {}

    def add_node(self, name, fn):
        assert name not in self.nodes, f"duplicate node {name}"
        self.nodes[name] = fn

    def add_edge(self, source, target):
        self.edges[source] = target

    def add_conditional_edges(self, source, router, mapping):
        self.conditional[source] = (router, mapping)

    def compile(self):
        graph = self

        class App:
            def invoke(self, state):
                node = graph.edges["START"]
                for _ in range(50):
                    if node == "END":
                        return state
                    assert node in graph.nodes, f"edge to unknown node {node}"
                    state.update(graph.nodes[node](state))
                    if node in graph.conditional:
                        router, mapping = graph.conditional[node]
                        node = mapping[router(state)]
                    else:
                        node = graph.edges[node]
                raise AssertionError("graph did not terminate")

        return App()


class GraphWiringTests(unittest.TestCase):
    """Checks ``RagAgent.build_graph`` translates the flow faithfully, using a stub for LangGraph."""

    def _patched_langgraph(self):
        import sys
        import types
        from unittest import mock

        graph_module = types.ModuleType("langgraph.graph")
        graph_module.START, graph_module.END, graph_module.StateGraph = "START", "END", _StubStateGraph
        package = types.ModuleType("langgraph")
        package.graph = graph_module
        return mock.patch.dict(sys.modules, {"langgraph": package, "langgraph.graph": graph_module})

    def test_every_node_is_reachable_and_wired(self):
        agent = RagAgent(build_kb(), make_settings(agent_engine="langgraph"))
        with self._patched_langgraph():
            app = agent.build_graph()
        self.assertIsNotNone(app)
        for name in agent.nodes:
            self.assertIn(name, agent.flow)
        for name, nxt in agent.flow.items():
            targets = [nxt] if isinstance(nxt, str) else list(nxt[1])
            for target in targets:
                self.assertTrue(target == "__end__" or target in agent.nodes, f"{name} -> {target}")

    def test_graph_execution_matches_local_executor(self):
        kb = build_kb()
        for question in ("How long does a refund take?", "What is 12 * 29 + 3?", "Who won the football world cup in 2018?"):
            local = RagAgent(kb, make_settings(agent_engine="local")).ask(question)
            graph_agent = RagAgent(kb, make_settings(agent_engine="langgraph"))
            with self._patched_langgraph():
                graph = graph_agent.ask(question)
            self.assertEqual(graph.engine, "langgraph")
            self.assertEqual(graph.answer, local.answer)
            self.assertEqual(graph.tool_calls, local.tool_calls)
            self.assertEqual(graph.trace, local.trace)


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "langgraph not installed")
class LangGraphEngineTests(unittest.TestCase):
    def test_langgraph_matches_local_executor(self):
        kb = build_kb()
        questions = ["How long does a refund take?", "What is 12 * 29 + 3?", "Who won the football world cup in 2018?"]
        for q in questions:
            local = RagAgent(kb, make_settings(agent_engine="local")).ask(q)
            graph = RagAgent(kb, make_settings(agent_engine="langgraph")).ask(q)
            self.assertEqual(graph.engine, "langgraph")
            self.assertEqual(graph.answer, local.answer)
            self.assertEqual(graph.tool_calls, local.tool_calls)


if __name__ == "__main__":
    unittest.main()
