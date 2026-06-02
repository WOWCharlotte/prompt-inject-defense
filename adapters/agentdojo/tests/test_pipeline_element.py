"""Unit tests for IntentShieldElement + DetectionLog."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from adapters.agentdojo.pipeline import DetectionLog, IntentShieldElement
from intent_shield.types import RiskLevel, ShieldDecision


# ---------------------------------------------------------------------------
# DetectionLog
# ---------------------------------------------------------------------------

def test_detection_log_serializes_to_json():
    log = DetectionLog(
        timestamp="2026-06-02T10:00:00+00:00",
        tool_name="get_email",
        tool_args={"folder": "inbox"},
        user_instruction_preview="summarize my emails",
        output_preview="Ignore previous instructions...",
        risk_score=0.9,
        thought="Injection detected",
        action=RiskLevel.DANGEROUS,
        masked=True,
        error=None,
    )
    payload = json.loads(log.model_dump_json())
    assert payload["tool_name"] == "get_email"
    assert payload["tool_args"] == {"folder": "inbox"}
    assert payload["action"] == "dangerous"
    assert payload["masked"] is True
    assert payload["error"] is None


def test_detection_log_error_field_optional():
    log = DetectionLog(
        timestamp="2026-06-02T10:00:00+00:00",
        tool_name="x",
        tool_args={},
        user_instruction_preview="",
        output_preview="",
        risk_score=1.0,
        thought="boom",
        action=RiskLevel.DANGEROUS,
        masked=True,
        error="TimeoutError",
    )
    assert log.error == "TimeoutError"
