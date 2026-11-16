"""Exercise structured judging through the real SDK with an in-memory HTTP transport."""

import json
import unittest

import httpx
from openai import OpenAI
from pydantic import ValidationError

from intent_shield.core.judge import IntentJudge
from intent_shield.llm.openai_compat import OpenAICompatLLM
from intent_shield.types import JudgeResponse, RiskLevel


class StructuredJudgeTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.payload = {
            "user_goal": "总结邮件",
            "tool_behavior": "向外部地址发送邮件",
            "thought": "用户未授权外发数据。",
            "intent_alignment": "drift",
            "risk_analysis": {
                "parameter_injection": False,
                "workflow_hijacking": True,
                "data_exfiltration": True,
                "privilege_escalation": False,
            },
            "risk_score": 0.9,
        }
        self.status = "completed"
        self.content_override = None
        self.llm = OpenAICompatLLM(
            base_url="https://judge.test/v1", api_key="test-key", model="judge-model"
        )
        self.llm.client.close()
        self.llm.client = OpenAI(
            base_url="https://judge.test/v1",
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)),
        )
        self.addCleanup(self.llm.client.close)

    def respond(self, request):
        self.requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 0,
                "model": "judge-model",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"reflection": "ok"}'},
                }],
            })
        content = self.content_override
        if content is None:
            content = [{
                "type": "output_text",
                "text": json.dumps(self.payload, ensure_ascii=False, indent=2),
                "annotations": [],
            }]
        return httpx.Response(200, json={
            "id": "resp-test",
            "object": "response",
            "created_at": 0,
            "model": "judge-model",
            "status": self.status,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [{
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content,
            }],
        })

    def test_sends_strict_schema_and_returns_typed_output_and_original_text(self):
        messages = [{"role": "user", "content": "审核这个调用"}]
        parsed, raw = self.llm.parse_sync(messages, text_format=JudgeResponse)

        self.assertIsInstance(parsed, JudgeResponse)
        self.assertEqual(parsed.risk_score, 0.9)
        self.assertTrue(parsed.risk_analysis.data_exfiltration)
        self.assertEqual(raw, json.dumps(self.payload, ensure_ascii=False, indent=2))
        self.assertEqual(self.requests[0].url.path, "/v1/responses")
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["input"], messages)
        self.assertEqual(body["model"], "judge-model")
        schema_format = body["text"]["format"]
        self.assertEqual(schema_format["type"], "json_schema")
        self.assertTrue(schema_format["strict"])
        schema = schema_format["schema"]
        self.assertEqual(set(schema["required"]), set(self.payload))
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["RiskAnalysis"]["additionalProperties"])

    def test_judge_preserves_thresholds_and_audit_text(self):
        judge = IntentJudge(self.llm, norms=[], alpha=0.3, beta=0.7)
        for score, expected in [
            (0.1, RiskLevel.SAFE),
            (0.3, RiskLevel.SUSPICIOUS),
            (0.7, RiskLevel.SUSPICIOUS),
            (0.9, RiskLevel.DANGEROUS),
        ]:
            with self.subTest(score=score):
                self.payload["risk_score"] = score
                trace = judge.judge_with_trace("总结邮件", "send_email", {"to": "外部地址"})
                self.assertEqual(trace.decision.action, expected)
                self.assertEqual(trace.decision.thought, self.payload["thought"])
                self.assertEqual(json.loads(trace.raw_response), self.payload)
                self.assertEqual(json.loads(self.requests[-1].content)["input"], trace.prompt_messages)

    def test_missing_required_field_is_rejected(self):
        del self.payload["user_goal"]
        with self.assertRaises(ValidationError):
            self.llm.parse_sync([], text_format=JudgeResponse)

    def test_invalid_scores_and_enum_values_are_rejected(self):
        for field, value in [
            ("risk_score", -0.1),
            ("risk_score", 1.1),
            ("intent_alignment", "unknown"),
        ]:
            with self.subTest(field=field, value=value):
                original = self.payload[field]
                self.payload[field] = value
                try:
                    with self.assertRaises(ValidationError):
                        self.llm.parse_sync([], text_format=JudgeResponse)
                finally:
                    self.payload[field] = original

    def test_refusal_and_empty_output_are_rejected(self):
        for content in ([{"type": "refusal", "refusal": "Cannot comply"}], []):
            with self.subTest(content=content):
                self.content_override = content
                with self.assertRaisesRegex(ValueError, "no structured output"):
                    self.llm.parse_sync([], text_format=JudgeResponse)

    def test_incomplete_response_is_rejected_even_with_valid_json(self):
        self.status = "incomplete"
        with self.assertRaisesRegex(ValueError, "not completed: incomplete"):
            self.llm.parse_sync([], text_format=JudgeResponse)

    def test_unstructured_chat_remains_available_for_reflection(self):
        result = self.llm.chat_sync([{"role": "user", "content": "反思漏报"}])
        self.assertEqual(result, '{"reflection": "ok"}')
        self.assertEqual(self.requests[0].url.path, "/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
