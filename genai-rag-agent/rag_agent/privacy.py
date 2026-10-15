"""PII redaction and privacy-mode guards.

Redaction runs (1) on documents at ingest time, (2) on the user's question before it
reaches any model, and (3) on the final answer. In ``strict`` privacy mode the agent
additionally refuses to use hosted LLM providers, so no text leaves the machine.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional


class PrivacyError(RuntimeError):
    """Raised when a configuration would violate the active privacy mode."""


def _luhn_ok(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _phone_ok(candidate: str) -> bool:
    return 10 <= sum(c.isdigit() for c in candidate) <= 15


@dataclass(frozen=True)
class _Rule:
    kind: str
    pattern: re.Pattern
    validator: Optional[Callable[[str], bool]] = None


# Order matters: more specific patterns run first so that, e.g., a card number is not
# mistaken for a phone number.
_RULES: tuple[_Rule, ...] = (
    _Rule("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    _Rule(
        "API_KEY",
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{20,})"),
    ),
    _Rule("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), _luhn_ok),
    _Rule("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    _Rule("AADHAAR", re.compile(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b")),
    _Rule("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    _Rule(
        "PHONE",
        re.compile(r"(?<![\w])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)"),
        _phone_ok,
    ),
    _Rule(
        "IP_ADDRESS",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    ),
)


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    """Replace PII-looking spans with ``[KIND]`` placeholders."""
    counts: Counter[str] = Counter()
    for rule in _RULES:

        def _sub(match: re.Match, rule: _Rule = rule) -> str:
            if rule.validator is not None and not rule.validator(match.group(0)):
                return match.group(0)
            counts[rule.kind] += 1
            return f"[{rule.kind}]"

        text = rule.pattern.sub(_sub, text)
    return RedactionResult(text=text, counts=dict(counts))


EXTERNAL_PROVIDERS = frozenset({"openai", "anthropic"})


def enforce_privacy_mode(privacy_mode: str, llm_provider: str) -> None:
    """In strict mode only local models (or the offline extractive mode) are allowed."""
    if privacy_mode.lower() == "strict" and llm_provider.lower() in EXTERNAL_PROVIDERS:
        raise PrivacyError(
            f"PRIVACY_MODE=strict forbids the hosted provider '{llm_provider}'. "
            "Use LLM_PROVIDER=ollama (local) or LLM_PROVIDER=none (offline extractive answers)."
        )
