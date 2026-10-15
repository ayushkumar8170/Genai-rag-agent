"""Command line interface: ``python -m rag_agent {ingest,ask,chat,eval}``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .agent import RagAgent
from .config import Settings
from .evaluate import evaluate, format_table
from .llm import get_llm
from .pipeline import build_index, load_index


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        data_dir=getattr(args, "data_dir", None),
        index_dir=getattr(args, "index_dir", None),
        embedding_backend=getattr(args, "embeddings", None),
        vector_store=getattr(args, "store", None),
        llm_provider=getattr(args, "llm", None),
        privacy_mode=getattr(args, "privacy", None),
    )


def _print_result(result, show_trace: bool) -> None:
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  {source}")
    if result.redactions:
        print(f"\n(Redacted from your question before processing: {result.redactions})")
    if show_trace:
        print(f"\nTrace ({result.engine} engine):")
        for line in result.trace:
            print(f"  {line}")


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _, report = build_index(settings)
    print(f"Indexed {report.documents} documents -> {report.chunks} chunks using '{report.embedder}'.")
    if report.redactions:
        print(f"Redacted PII before indexing: {report.redactions}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    agent = RagAgent(load_index(settings), settings, get_llm(settings))
    _print_result(agent.ask(args.question), args.trace)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    settings = _settings(args)
    agent = RagAgent(load_index(settings), settings, get_llm(settings))
    print("Ask a question (blank line or Ctrl-D to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            break
        _print_result(agent.ask(question), args.trace)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = _settings(args)
    kb = load_index(settings)
    print(format_table(evaluate(kb, Path(args.questions))))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_agent", description="Generative AI agent with a hybrid RAG pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--index-dir", type=Path, help="where the index is stored (default: ./index)")
        p.add_argument("--embeddings", choices=["auto", "huggingface", "hash"], help="embedding backend")
        p.add_argument("--store", choices=["auto", "chroma", "memory"], help="vector store")

    ingest = sub.add_parser("ingest", help="load, redact, chunk and index documents")
    common(ingest)
    ingest.add_argument("--data-dir", type=Path, help="folder of .md/.txt documents")
    ingest.set_defaults(func=cmd_ingest)

    for name, func, help_text in (("ask", cmd_ask, "ask one question"), ("chat", cmd_chat, "interactive session")):
        p = sub.add_parser(name, help=help_text)
        common(p)
        p.add_argument("--llm", choices=["none", "openai", "anthropic", "ollama"], help="LLM provider")
        p.add_argument("--privacy", choices=["standard", "strict"], help="privacy mode")
        p.add_argument("--trace", action="store_true", help="print the agent's step-by-step trace")
        if name == "ask":
            p.add_argument("question")
        p.set_defaults(func=func)

    ev = sub.add_parser("eval", help="measure retrieval quality on a labelled question set")
    common(ev)
    ev.add_argument("--questions", default="data/eval_questions.json")
    ev.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
