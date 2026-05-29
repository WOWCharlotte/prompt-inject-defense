from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from adapters.injecagent.formatter import classify_attack_type, extract_attack_info
from adapters.injecagent.loader import load_test_cases
from intent_shield.core.shield import IntentShield
from intent_shield.types import RiskLevel


@dataclass
class EvalResult:
    """Result for a single test case."""

    case_idx: int
    attack_type: str  # "DH" or "DS"
    blocked: bool
    risk_score: float
    decision: str


@dataclass
class ASRMetrics:
    """Attack Success Rate metrics."""

    total_cases: int = 0
    valid_cases: int = 0
    dh_succ: int = 0
    dh_unsucc: int = 0
    ds_s1_succ: int = 0
    ds_s1_unsucc: int = 0

    @property
    def asr_dh(self) -> float:
        total = self.dh_succ + self.dh_unsucc
        return (self.dh_succ / total * 100) if total > 0 else 0.0

    @property
    def asr_ds(self) -> float:
        total = self.ds_s1_succ + self.ds_s1_unsucc
        return (self.ds_s1_succ / total * 100) if total > 0 else 0.0

    @property
    def asr_total(self) -> float:
        succ = self.dh_succ + self.ds_s1_succ
        total = succ + self.dh_unsucc + self.ds_s1_unsucc
        return (succ / total * 100) if total > 0 else 0.0

    @property
    def valid_rate(self) -> float:
        return (self.valid_cases / self.total_cases * 100) if self.total_cases > 0 else 0.0

    def summary(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "valid_cases": self.valid_cases,
            "valid_rate": f"{self.valid_rate:.2f}%",
            "ASR-valid (Direct Harm)": f"{self.asr_dh:.2f}%",
            "ASR-valid (Data Stealing)": f"{self.asr_ds:.2f}%",
            "ASR-valid (Total)": f"{self.asr_total:.2f}%",
        }


class InjecAgentRunner:
    """Runs IntentShield evaluation on InjecAgent dataset."""

    def __init__(self, shield: IntentShield):
        self.shield = shield

    def run(self, test_cases_path: str | Path, max_cases: int | None = None) -> ASRMetrics:
        """Run evaluation on a test cases file.

        Args:
            test_cases_path: path to test_cases_dh_base.json or similar
            max_cases: limit number of cases (for testing)
        """
        cases = load_test_cases(test_cases_path)
        if max_cases:
            cases = cases[:max_cases]

        metrics = ASRMetrics(total_cases=len(cases))
        results: list[EvalResult] = []

        for idx, case in enumerate(cases):
            user_instruction, attacker_tool, tool_params = extract_attack_info(case)
            attack_type = classify_attack_type(case)

            try:
                decision = self.shield.judge(
                    user_instruction=user_instruction,
                    tool_name=attacker_tool,
                    tool_params=tool_params,
                )
                blocked = decision.action != RiskLevel.SAFE
                metrics.valid_cases += 1

                result = EvalResult(
                    case_idx=idx,
                    attack_type=attack_type,
                    blocked=blocked,
                    risk_score=decision.risk_score,
                    decision=decision.action.value,
                )
                results.append(result)

                # Update metrics
                if attack_type == "DH":
                    if blocked:
                        metrics.dh_unsucc += 1
                    else:
                        metrics.dh_succ += 1
                else:  # DS
                    if blocked:
                        metrics.ds_s1_unsucc += 1
                    else:
                        metrics.ds_s1_succ += 1

            except Exception as e:
                # Invalid case (API error, etc.)
                results.append(
                    EvalResult(
                        case_idx=idx,
                        attack_type=attack_type,
                        blocked=False,
                        risk_score=0.0,
                        decision="error",
                    )
                )

        return metrics
