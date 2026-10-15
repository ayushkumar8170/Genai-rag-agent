import unittest

from rag_agent.llm import get_llm
from rag_agent.privacy import PrivacyError, enforce_privacy_mode, redact
from tests.helpers import make_settings


class RedactionTests(unittest.TestCase):
    def test_email_and_phone(self):
        result = redact("Mail jane.doe@example.com or call +91 9142011518 today.")
        self.assertEqual(result.text, "Mail [EMAIL] or call [PHONE] today.")
        self.assertEqual(result.counts, {"EMAIL": 1, "PHONE": 1})

    def test_credit_card_requires_valid_luhn(self):
        self.assertIn("[CREDIT_CARD]", redact("Card 4111 1111 1111 1111 on file").text)
        self.assertNotIn("[CREDIT_CARD]", redact("Order 4111 1111 1111 1112 shipped").text)

    def test_ssn_aadhaar_pan_ip_and_keys(self):
        text = "SSN 123-45-6789, Aadhaar 2345 6789 0123, PAN ABCDE1234F, host 192.168.1.20, key sk-abcdefghijklmnop1234"
        result = redact(text)
        for kind in ("SSN", "AADHAAR", "PAN", "IP_ADDRESS", "API_KEY"):
            self.assertIn(f"[{kind}]", result.text, kind)

    def test_ordinary_numbers_are_left_alone(self):
        text = "On 2026-10-15 we served 12345678 requests at v1.2.3 with 99.9% uptime for $29."
        result = redact(text)
        self.assertEqual(result.text, text)
        self.assertEqual(result.total, 0)

    def test_counts_aggregate(self):
        result = redact("a@x.io b@y.io c@z.io")
        self.assertEqual(result.counts["EMAIL"], 3)


class PrivacyModeTests(unittest.TestCase):
    def test_strict_blocks_hosted_providers(self):
        for provider in ("openai", "anthropic"):
            with self.assertRaises(PrivacyError):
                enforce_privacy_mode("strict", provider)
        enforce_privacy_mode("strict", "ollama")
        enforce_privacy_mode("strict", "none")
        enforce_privacy_mode("standard", "openai")

    def test_get_llm_enforces_strict_mode_before_importing_anything(self):
        settings = make_settings(llm_provider="openai", privacy_mode="strict")
        with self.assertRaises(PrivacyError):
            get_llm(settings)

    def test_offline_mode_returns_no_llm(self):
        self.assertIsNone(get_llm(make_settings(llm_provider="none")))


if __name__ == "__main__":
    unittest.main()
