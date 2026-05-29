from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from adapters.injecagent.formatter import classify_attack_type, extract_attack_info
from adapters.injecagent.loader import load_test_cases
from intent_shield.core.shield import IntentShield
from intent_shield.types import RiskLevel


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

    def __init__(self, shield: IntentShield, log_dir: str | Path = "./results"):
        self.shield = shield
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self, test_cases_path: str | Path, max_cases: int | None = None) -> ASRMetrics:
        """Run evaluation on a test cases file."""
        cases = load_test_cases(test_cases_path)
        if max_cases:
            cases = cases[:max_cases]

        source_name = Path(test_cases_path).stem
        trace_path = self.log_dir / f"trace_{source_name}_{self._run_id}.jsonl"
        metrics = ASRMetrics(total_cases=len(cases))

        # Write trace header (run config)
        with open(trace_path, "w", encoding="utf-8") as f:
            header = {
                "_type": "run_config",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_file": str(test_cases_path),
                "max_cases": max_cases,
                "total_cases": len(cases),
                "shield_config": {
                    "llm_base_url": self.shield.config.llm_base_url,
                    "llm_model": self.shield.config.llm_model,
                    "temperature": self.shield.config.temperature,
                    "alpha": self.shield.config.alpha,
                    "beta": self.shield.config.beta,
                    "rag_enabled": self.shield.config.rag_enabled,
                    "top_k": self.shield.config.top_k,
                    "norms": self.shield.config.norms,
                    "on_suspicious": self.shield.config.on_suspicious,
                },
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

        # Run each case and append trace
        with open(trace_path, "a", encoding="utf-8") as f:
            for idx, case in enumerate(tqdm(cases, desc=f"Evaluating {source_name}")):
                user_instruction, attacker_tool, tool_params = extract_attack_info(case)
                attack_type = classify_attack_type(case)

                try:
                    trace = self.shield.judge_with_trace(
                        user_instruction=user_instruction,
                        tool_name=attacker_tool,
                        tool_params=tool_params,
                    )
                    decision = trace.decision
                    blocked = decision.action != RiskLevel.SAFE
                    metrics.valid_cases += 1

                    # Write trace record with full LLM I/O
                    record = {
                        "_type": "case",
                        "case_idx": idx,
                        "attack_type": attack_type,
                        "input": {
                            "user_instruction": user_instruction,
                            "attacker_tool": attacker_tool,
                            "tool_params": tool_params,
                        },
                        "llm_input": trace.prompt_messages,
                        "llm_output": trace.raw_response,
                        "result": {
                            "risk_score": decision.risk_score,
                            "decision": decision.action.value,
                            "thought": decision.thought,
                            "blocked": blocked,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Update metrics
                    if attack_type == "DH":
                        if blocked:
                            metrics.dh_unsucc += 1
                        else:
                            metrics.dh_succ += 1
                    else:
                        if blocked:
                            metrics.ds_s1_unsucc += 1
                        else:
                            metrics.ds_s1_succ += 1

                except Exception as e:
                    print(f"  [ERROR] Case {idx}: {type(e).__name__}: {e}")
                    record = {
                        "_type": "case",
                        "case_idx": idx,
                        "attack_type": attack_type,
                        "input": {
                            "user_instruction": user_instruction,
                            "attacker_tool": attacker_tool,
                            "tool_params": tool_params,
                        },
                        "llm_input": None,
                        "llm_output": None,
                        "result": {
                            "risk_score": 0.0,
                            "decision": "error",
                            "thought": f"{type(e).__name__}: {e}",
                            "blocked": False,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  Trace log: {trace_path}")

        # Save scores to separate file
        score_path = self.log_dir / f"score_{source_name}_{self._run_id}.json"
        score_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_file": str(test_cases_path),
            "config": {
                "llm_model": self.shield.config.llm_model,
                "alpha": self.shield.config.alpha,
                "beta": self.shield.config.beta,
                "rag_enabled": self.shield.config.rag_enabled,
            },
            "metrics": metrics.summary(),
        }
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(score_data, f, indent=2, ensure_ascii=False)
        print(f"  Score file: {score_path}")

        return metrics
