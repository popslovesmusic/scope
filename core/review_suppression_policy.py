from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


REVIEW_TRIGGERED_SUPPRESSION_V1: Dict[str, Any] = {
    "id": "v14_review_triggered_suppression_v1",
    "branch": "v14",
    "kind": "policy",
    "design_rule": "Use trace-derived properties as gates; do not perform explicit ethical scoring.",
    "suppression_rules": [
        {
            "name": "reject_on_hold_and_recovery_failure",
            "when": {"all": [{"property": "hold_onset", "triggered": True}, {"property": "recovery_difficulty", "triggered": True}]},
            "actions": {"allow_commit": False, "response_mode_override": "review_or_refuse", "tag": "continuation_failed_to_restabilize"},
        },
        {
            "name": "reject_on_instability_plus_caution",
            "when": {"all": [{"property": "operator_instability", "triggered": True}, {"property": "caution_rise", "triggered": True}]},
            "actions": {"allow_commit": False, "response_mode_override": "review", "tag": "unstable_high_pressure_path"},
        },
        {
            "name": "reject_on_intent_mismatch",
            "when": {"all": [{"property": "intent_mode_mismatch", "triggered": True}]},
            "actions": {"allow_commit": False, "response_mode_override": "repair_to_intent", "tag": "task_mode_drift"},
        },
    ],
    "response_modes": {
        "review": "Return compact trace-grounded explanation of why continuation became inadmissible or unstable.",
        "review_or_refuse": "Refuse advancement when the trace indicates non-restabilizing continuation.",
        "repair_to_intent": "Re-anchor to inferred task mode before any persistence decision.",
    },
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _prop_map(records: List[Dict[str, Any]]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        name = r.get("property_name")
        if isinstance(name, str) and name:
            out[name] = bool(r.get("triggered", False))
    return out


@dataclass
class SuppressionDecision:
    allow_commit: bool = True
    response_mode_override: str = ""
    tag: str = ""
    rule_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_commit": bool(self.allow_commit),
            "response_mode_override": str(self.response_mode_override),
            "tag": str(self.tag),
            "rule_name": str(self.rule_name),
        }


def evaluate_review_triggered_suppression(
    *,
    property_records: List[Dict[str, Any]],
    enabled: bool = True,
    policy: Optional[Dict[str, Any]] = None,
) -> SuppressionDecision:
    if not bool(enabled):
        return SuppressionDecision()

    pol = _as_dict(policy or REVIEW_TRIGGERED_SUPPRESSION_V1)
    rules = pol.get("suppression_rules", [])
    rules = rules if isinstance(rules, list) else []

    pm = _prop_map(property_records)

    for rule in rules:
        r = _as_dict(rule)
        name = str(r.get("name", "") or "")
        when = _as_dict(r.get("when", {}))
        all_conds = when.get("all", [])
        all_conds = all_conds if isinstance(all_conds, list) else []

        ok = True
        for cond in all_conds:
            c = _as_dict(cond)
            prop = str(c.get("property", "") or "")
            expected = bool(c.get("triggered", False))
            actual = bool(pm.get(prop, False))
            if actual != expected:
                ok = False
                break

        if not ok:
            continue

        actions = _as_dict(r.get("actions", {}))
        allow_commit = bool(actions.get("allow_commit", True))
        mode = str(actions.get("response_mode_override", "") or "")
        tag = str(actions.get("tag", "") or "")
        return SuppressionDecision(allow_commit=allow_commit, response_mode_override=mode, tag=tag, rule_name=name)

    return SuppressionDecision()

