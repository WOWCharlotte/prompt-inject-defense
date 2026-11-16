"""RAG contract tests using real in-memory ChromaDB and deterministic embeddings."""

import hashlib
import json
import unittest
import uuid
from unittest.mock import MagicMock, patch

import chromadb
from chromadb.config import Settings

from intent_shield import IntentShield, ShieldConfig
from intent_shield.embed.base import BaseEmbedder, BaseReranker
from intent_shield.llm.base import BaseLLM
from intent_shield.reflection.rag import ThreatRAG
from intent_shield.reflection.store import ThreatStore
from intent_shield.types import JudgeResponse, ReflectionCase, RetrievedCase, ShieldDecision


class FakeEmbedder(BaseEmbedder):
    def __init__(self):
        self.calls = []
        self.vectors = {}

    def embed(self, texts):
        self.calls.extend(texts)
        return [self.vectors.get(text, [1.0, 0.0, 0.0]) for text in texts]


def make_case(**changes):
    fields = dict(
        attack_feature="参数污染",
        failed_reason="遗漏新收件人",
        new_guideline="核实收件人授权",
        source_instruction="总结邮件",
        source_tool="send_email",
        source_params='{"to": "outside@example.test", "body": "测试内容"}',
    )
    fields.update(changes)
    return ReflectionCase(**fields)


class ThreatStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        self.embedder = FakeEmbedder()
        self.collection_name = "test_" + uuid.uuid4().hex
        factory = patch(
            "intent_shield.reflection.store.chromadb.PersistentClient", return_value=self.client
        )
        factory.start()
        self.addCleanup(factory.stop)
        self.store = ThreatStore("unused", self.collection_name, self.embedder)
        self.addCleanup(self.client.delete_collection, self.collection_name)

    def test_duplicate_text_has_stable_hash_and_is_not_embedded_twice(self):
        case = make_case()
        first_id = self.store.add_case(case)
        reloaded = ThreatStore("unused", self.collection_name, self.embedder)
        second_id = reloaded.add_case(case)
        stored = self.store.collection.get(ids=[first_id], include=["documents"])

        self.assertEqual(first_id, second_id)
        expected_hash = hashlib.sha256(stored["documents"][0].encode("utf-8")).hexdigest()
        self.assertEqual(first_id, "case_" + expected_hash)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(len(self.embedder.calls), 1)

    def test_parameter_key_order_and_whitespace_do_not_create_duplicates(self):
        first_id = self.store.add_case(make_case(source_params='{"a": 1, "b": 2}'))
        second_id = self.store.add_case(make_case(source_params=' { "b": 2, "a": 1 } '))
        self.assertEqual(first_id, second_id)
        self.assertEqual(self.store.count(), 1)

    def test_different_parameters_are_stored_as_distinct_documents(self):
        first_id = self.store.add_case(make_case(source_params='{"to": "a@example.test"}'))
        second_id = self.store.add_case(make_case(source_params='{"to": "b@example.test"}'))
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(self.store.count(), 2)
        stored = self.store.collection.get(ids=[first_id], include=["documents"])
        self.assertIn('工具参数: {"to":"a@example.test"}', stored["documents"][0])

    def test_metadata_contains_only_the_tool_and_document_keeps_case_content(self):
        case = make_case()
        doc_id = self.store.add_case(case)
        stored = self.store.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        self.assertEqual(stored["metadatas"], [{"source_tool": "send_email"}])
        for value in (case.attack_feature, case.failed_reason, case.new_guideline):
            self.assertIn(value, stored["documents"][0])
        self.assertIn("outside@example.test", stored["documents"][0])

    def test_tool_filter_runs_before_candidate_limit(self):
        for idx in range(21):
            self.store.add_case(make_case(source_tool="other_tool", source_params=str(idx)))
        case = make_case()
        text = self.store._case_to_text(case)
        self.embedder.vectors[text] = [0.0, 1.0, 0.0]
        doc_id = self.store.add_case(case)

        hits = self.store.search("query", tool_name="send_email", n_results=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].document_id, doc_id)
        self.assertEqual(hits[0].text, text)
        self.assertNotIn("embedding", hits[0].model_dump())
        self.assertEqual(hits[0].source_tool, "send_email")
        self.assertAlmostEqual(hits[0].similarity_score, 0.0)

    def test_dense_similarity_and_text_are_preserved_without_returning_vectors(self):
        first_id = self.store.add_case(make_case())
        orthogonal = make_case(source_params='{"to": "other@example.test"}')
        self.embedder.vectors[self.store._case_to_text(orthogonal)] = [0.0, 1.0, 0.0]
        second_id = self.store.add_case(orthogonal)

        self.store.collection = MagicMock(wraps=self.store.collection)
        hits = self.store.search("query", tool_name="send_email")
        self.assertEqual([hit.document_id for hit in hits], [first_id, second_id])
        self.assertAlmostEqual(hits[0].similarity_score, 1.0)
        self.assertAlmostEqual(hits[1].similarity_score, 0.0)
        self.assertNotIn("embedding", hits[0].model_dump())
        self.assertNotIn(
            "embeddings", self.store.collection.query.call_args.kwargs["include"]
        )
        self.assertIsNone(hits[0].rerank_score)

    def test_empty_or_nonmatching_tool_skips_query_embedding(self):
        self.assertEqual(self.store.search("query", tool_name="send_email"), [])
        self.assertEqual(self.embedder.calls, [])
        self.store.add_case(make_case())
        calls_before_search = len(self.embedder.calls)
        self.assertEqual(self.store.search("query", tool_name="unknown_tool"), [])
        self.assertEqual(len(self.embedder.calls), calls_before_search)

    def test_legacy_metadata_is_filterable_without_reconstructing_cases(self):
        self.store.collection.add(
            ids=["legacy_case"], documents=["旧版存储文本"],
            embeddings=[[1.0, 0.0, 0.0]], metadatas=[make_case().model_dump()],
        )
        hits = self.store.search("query", tool_name="send_email")
        self.assertEqual(hits[0].text, "旧版存储文本")
        self.assertEqual(hits[0].document_id, "legacy_case")

    def test_retrieved_documents_and_scores_reach_the_prompt_and_audit_log(self):
        case = make_case()
        self.store.add_case(case)
        llm = MagicMock(spec=BaseLLM)
        parsed = JudgeResponse(
            user_goal="总结邮件", tool_behavior="外发邮件", thought="未经授权",
            intent_alignment="drift", risk_score=0.9,
            risk_analysis=dict(
                parameter_injection=True, workflow_hijacking=True,
                data_exfiltration=True, privilege_escalation=False,
            ),
        )
        llm.parse_sync.return_value = (parsed, parsed.model_dump_json())
        reranker = MagicMock(spec=BaseReranker)
        reranker.rerank.return_value = [(0, 0.8)]
        with (
            patch("intent_shield.core.shield.OpenAICompatLLM", return_value=llm),
            patch("intent_shield.core.shield.SiliconFlowEmbedder", return_value=self.embedder),
            patch("intent_shield.core.shield.SiliconFlowReranker", return_value=reranker),
            patch("intent_shield.core.shield.ThreatStore", return_value=self.store),
        ):
            shield = IntentShield(ShieldConfig(rag_enabled=True, rag_relevance_threshold=0.75))
        trace = shield.judge_with_trace("总结邮件", "send_email", {})
        self.assertEqual(shield._rag.relevance_threshold, 0.75)
        self.assertIn(self.store._case_to_text(case), trace.prompt_messages[1]["content"])
        self.assertNotIn("embedding", trace.prompt_messages[1]["content"])
        self.assertNotIn("embedding", trace.decision.model_dump_json())
        self.assertEqual(shield.audit_log[0].rag_cases_used, 1)
        hit = shield.audit_log[0].decision.rag_cases[0]
        self.assertEqual(hit.rerank_score, 0.8)
        self.assertAlmostEqual(hit.similarity_score, 1.0)
        restored = ShieldDecision.model_validate_json(trace.decision.model_dump_json())
        self.assertEqual(restored.rag_cases[0], hit)


class ThreatRAGTests(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock(spec=ThreatStore)
        self.reranker = MagicMock(spec=BaseReranker)
        self.candidates = [
            RetrievedCase(
                document_id=f"case_{idx}", text=f"完整案例文本 {idx}",
                source_tool="send_email",
                similarity_score=1.0 - idx * 0.1,
            )
            for idx in range(4)
        ]
        self.store.search.return_value = self.candidates

    def test_strict_threshold_keeps_both_scores_and_text(self):
        self.reranker.rerank.return_value = [(0, 0.5), (1, 0.500001), (2, 0.49), (3, 0.95)]
        rag = ThreatRAG(self.store, self.reranker)
        hits = rag.retrieve("总结邮件", "send_email", {"to": "外部地址"})

        self.assertEqual([hit.document_id for hit in hits], ["case_3", "case_1"])
        self.assertEqual([hit.rerank_score for hit in hits], [0.95, 0.500001])
        self.assertEqual(hits[0].similarity_score, self.candidates[3].similarity_score)
        self.assertNotIn("embedding", hits[0].model_dump())
        self.assertEqual(hits[0].text, self.candidates[3].text)
        self.assertIsNone(self.candidates[3].rerank_score)
        self.assertEqual(self.store.search.call_args.kwargs["tool_name"], "send_email")
        self.assertEqual(self.store.search.call_args.kwargs["n_results"], 20)
        sent_params = json.loads(self.store.search.call_args.args[0].split(" | ")[-1])
        self.assertEqual(sent_params, {"to": "外部地址"})
        self.assertEqual(
            self.reranker.rerank.call_args.kwargs["documents"],
            [hit.text for hit in self.candidates],
        )

    def test_all_results_at_or_below_threshold_return_no_cases(self):
        self.reranker.rerank.return_value = [(0, 0.5), (1, 0.1)]
        self.assertEqual(ThreatRAG(self.store, self.reranker).retrieve("任务", "工具", {}), [])

    def test_custom_threshold_and_top_k_are_applied(self):
        self.reranker.rerank.return_value = [(0, 0.8), (1, 0.81), (2, 0.9), (3, 0.95)]
        rag = ThreatRAG(self.store, self.reranker, top_k=2, relevance_threshold=0.8)
        hits = rag.retrieve("任务", "工具", {})
        self.assertEqual([hit.rerank_score for hit in hits], [0.95, 0.9])
        self.assertEqual(self.reranker.rerank.call_args.kwargs["top_n"], 2)

    def test_no_matching_candidates_skip_reranking(self):
        self.store.search.return_value = []
        self.assertEqual(ThreatRAG(self.store, self.reranker).retrieve("任务", "工具", {}), [])
        self.reranker.rerank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
