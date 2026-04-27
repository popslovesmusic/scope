
"""
Response renderer for SBLLM v14 terminal shell.

This version is aligned to the actual runtime_output structure produced by
interfaces/chat_shell.py:

    runtime_output = {
        "prompt": ...,
        "seed": ...,
        "state": {
            "signature": {...},
            "corridor": {...},
            "orientation": {...},
            "reasoning": {...},
        },
        "trace": [...],
        "output": {
            "class_scores": ...,
            "selected_class": int,
            "confidence": float,
        },
    }

It preserves the legacy function interface expected by chat_shell.py:

    - render_response(...)
    - render_readback(...)
    - render_trace_summary(...)
    - render_debug_panel(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_present(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _fmt_float(value: Any, ndigits: int = 3) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "n/a"


def _truthy(mapping: Dict[str, Any], *keys: str, default: bool = False) -> bool:
    return bool(_first_present(mapping, *keys, default=default))


@dataclass
class RendererConfig:
    reply_prefix: str = "Reply:"
    trace_prefix: str = "Trace:"
    default_trace_items: int = 2
    max_trace_items: int = 12
    reply_templates: Dict[str, str] = field(default_factory=dict)
    fallback_reply: str = "I processed your input."
    unknown_reply: str = "I do not yet have a strong natural-language reply for that input."
    prefer_model_reply_keys: Sequence[str] = field(default_factory=lambda: (
        "reply",
        "response_text",
        "text",
        "message",
        "readable_reply",
        "natural_language_reply",
    ))


class ResponseRenderer:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        rr = _as_dict(_first_present(cfg, "response_renderer", "renderer", default={}))
        self.cfg = RendererConfig(
            reply_prefix=str(_first_present(rr, "reply_prefix", default="Reply:")),
            trace_prefix=str(_first_present(rr, "trace_prefix", default="Trace:")),
            default_trace_items=int(_first_present(rr, "default_trace_items", default=2)),
            max_trace_items=int(_first_present(rr, "max_trace_items", default=12)),
            reply_templates=_as_dict(_first_present(rr, "reply_templates", default={})),
            fallback_reply=str(_first_present(rr, "fallback_reply", default="I processed your input.")),
            unknown_reply=str(
                _first_present(
                    rr,
                    "unknown_reply",
                    default="I do not yet have a strong natural-language reply for that input.",
                )
            ),
        )

    # -----------------------------
    # Intent + state interpretation
    # -----------------------------
    def classify_intent(self, user_text: str) -> str:
        """
        Lightweight deterministic intent classifier.

        Categories:
          greeting | question | instruction | explanation_request | meta | unknown
        """
        text = (user_text or "").strip().lower()
        if not text:
            return "unknown"

        if text in {"hello", "hi", "hey", "yo"}:
            return "greeting"
        if re.search(r"\bhow\s+are\s+you\b", text):
            return "greeting"
        if re.search(r"\bwhat('?s| is)\s+up\b", text):
            return "greeting"
        if re.search(r"\bhow('?s| is)\s+it\s+going\b", text):
            return "greeting"

        if text in {"thanks", "thank you", "thx"}:
            return "meta"  # handled as a conversational meta response

        if re.search(r"\b(system status|status|engine|sbllm|trace|debug|operator scores|hold semantics)\b", text):
            return "meta"

        if re.match(r"^(explain|describe|give|show|teach)\b", text):
            return "explanation_request"

        if re.match(r"^(run|set|turn|toggle|enable|disable)\b", text):
            return "instruction"

        if text.endswith("?") or re.match(r"^(how|why|what|when|where)\b", text):
            return "question"

        return "unknown"

    def interpret_state_tokens(self, output: Dict[str, Any]) -> Dict[str, str]:
        """
        Map internal state signals into simple semantic tokens.
        """
        operator = self.extract_operator(output)
        control = self.extract_control(output)
        caution = float(_first_present(control, "caution", default=0.0) or 0.0)
        recovery = float(_first_present(control, "recovery", default=0.0) or 0.0)
        hold = bool(_first_present(control, "hold", default=False))

        op_map = {
            "++": "aligned",
            "+-": "mixed_tension",
            "--": "opposed",
            "-+": "unstable_transition",
        }
        alignment_state = op_map.get(str(operator), "unknown_alignment")

        if caution < 0.3:
            stability_state = "confident"
        elif caution <= 0.6:
            stability_state = "cautious"
        else:
            stability_state = "guarded"

        if recovery < 0.1:
            adaptation_state = "low_adaptation"
        elif recovery <= 0.5:
            adaptation_state = "adapting"
        else:
            adaptation_state = "recovering"

        if hold:
            stability_state = "holding_steady"

        return {
            "alignment_state": alignment_state,
            "stability_state": stability_state,
            "adaptation_state": adaptation_state,
        }

    def build_state_summary(self, output: Dict[str, Any]) -> str:
        tokens = self.interpret_state_tokens(output)
        a = tokens["alignment_state"]
        s = tokens["stability_state"]
        r = tokens["adaptation_state"]

        alignment_phrase = {
            "aligned": "aligned",
            "mixed_tension": "mixed",
            "opposed": "slightly opposed",
            "unstable_transition": "unsettled",
            "unknown_alignment": "unclear",
        }.get(a, "unclear")

        stability_phrase = {
            "confident": "stable",
            "cautious": "slightly cautious",
            "guarded": "guarded",
            "holding_steady": "holding steady",
        }.get(s, "steady")

        adaptation_phrase = {
            "low_adaptation": "not changing much",
            "adapting": "adapting",
            "recovering": "recovering",
        }.get(r, "adapting")

        # Keep this short and human-readable.
        if stability_phrase == "holding steady":
            return f"{stability_phrase} and {alignment_phrase}"
        if adaptation_phrase in {"adapting", "recovering"}:
            return f"{stability_phrase} and {alignment_phrase}, {adaptation_phrase}"
        return f"{stability_phrase} and {alignment_phrase}"

    # -----------------------------
    # Core extraction helpers
    # -----------------------------
    def _output_block(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return _as_dict(_first_present(output, "output", default={}))

    def _state_block(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return _as_dict(_first_present(output, "state", default={}))

    def _signature_block(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return _as_dict(_first_present(self._state_block(output), "signature", default={}))

    def _corridor_block(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return _as_dict(_first_present(self._state_block(output), "corridor", default={}))

    def _orientation_block(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return _as_dict(_first_present(self._state_block(output), "orientation", default={}))

    def _reasoning_block(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return _as_dict(_first_present(self._state_block(output), "reasoning", default={}))

    def extract_trace(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        trace = _as_list(_first_present(output, "trace", default=[]))
        if trace and all(isinstance(x, dict) for x in trace):
            return trace
        return []

    def extract_selected_class(self, output: Dict[str, Any]) -> Any:
        out_block = self._output_block(output)
        return _first_present(
            out_block,
            "selected_class",
            default=_first_present(output, "selected_class", "predicted_class", "class_id", default="n/a"),
        )

    def extract_confidence(self, output: Dict[str, Any]) -> Any:
        out_block = self._output_block(output)
        return _first_present(
            out_block,
            "confidence",
            default=_first_present(output, "confidence", "class_confidence", default=None),
        )

    def extract_dual_confidence(self, output: Dict[str, Any]) -> Dict[str, Any]:
        out_block = self._output_block(output)
        practical = _first_present(out_block, "practical_confidence", default=self.extract_confidence(output))
        structural = _first_present(out_block, "structural_confidence", default=None)
        return {"practical": practical, "structural": structural}

    def extract_operator(self, output: Dict[str, Any]) -> str:
        orientation = self._orientation_block(output)
        value = _first_present(
            orientation,
            "active_operator",
            default=_first_present(output, "operator", "selected_operator", "op", default=None),
        )

        if value is not None:
            return str(value)

        trace = self.extract_trace(output)
        if trace:
            value = _first_present(trace[-1], "selected_operator", "op", "operator", default=None)
            if value is not None:
                return str(value)

        return "n/a"

    def extract_active_component(self, output: Dict[str, Any]) -> Any:
        signature = self._signature_block(output)
        reasoning = self._reasoning_block(output)
        return _first_present(
            signature,
            "active_component_id",
            default=_first_present(output, "active_component", "active_component_id", default=_first_present(reasoning, "active_component_id", default="n/a")),
        )

    def extract_control(self, output: Dict[str, Any]) -> Dict[str, Any]:
        signature = self._signature_block(output)

        caution = _first_present(signature, "caution_scalar", default=_first_present(output, "caution", default=None))
        raw_caution = _first_present(signature, "raw_caution_scalar", default=_first_present(output, "raw_caution", default=None))
        recovery = _first_present(signature, "recovery_scalar", default=_first_present(output, "recovery", default=None))
        hold = _first_present(signature, "hold_state", default=_truthy(output, "hold", default=False))

        return {
            "caution": caution,
            "raw_caution": raw_caution,
            "recovery": recovery,
            "hold": hold,
        }

    # -----------------------------
    # Reply
    # -----------------------------
    def derive_reply(self, output: Dict[str, Any], user_text: str = "") -> str:
        # v15 semantic initialization: intent-aware replies driven by lightweight classification,
        # without changing core runtime behavior. For science/explanations, prefer explicit model
        # or semantic_readout replies if present; for conversational intents, respond briefly and
        # state-aware.
        text = (user_text or "").strip()
        intent = self.classify_intent(text)
        if intent == "greeting":
            return f"I'm doing {self.build_state_summary(output)}. How can I help?"
        if intent == "meta":
            lower = text.lower()
            if lower in {"thanks", "thank you", "thx"}:
                return "Youre welcome."
            selected_class = self.extract_selected_class(output)
            confidence = self.extract_confidence(output)
            operator = self.extract_operator(output)
            control = self.extract_control(output)
            return (
                "System status: "
                f"op={operator} "
                f"caution={_fmt_float(control.get('caution'))} "
                f"recovery={_fmt_float(control.get('recovery'))} "
                f"hold={control.get('hold')} "
                f"class={selected_class} conf={_fmt_float(confidence)}"
            )

        for key in self.cfg.prefer_model_reply_keys:
            value = _first_present(output, key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for block_name in ("output", "response", "render", "state"):
            block = _as_dict(_first_present(output, block_name, default={}))
            for key in self.cfg.prefer_model_reply_keys:
                value = _first_present(block, key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        selected_class = self.extract_selected_class(output)
        if selected_class is not None and selected_class != "n/a":
            template = self.cfg.reply_templates.get(str(selected_class))
            if template:
                return template

        low = (user_text or "").strip().lower()
        if intent == "instruction":
            return f"OK. ({self.build_state_summary(output)}) What should I run or change?"
        if intent in {"question", "explanation_request"} and low:
            return f"Heres a quick take. ({self.build_state_summary(output)}) Ask a specific science topic and Ill explain it clearly."
        if low:
            return self.cfg.unknown_reply
        return self.cfg.fallback_reply

    # -----------------------------
    # Renderers
    # -----------------------------
    def render_response(self, output: Dict[str, Any], user_text: str = "") -> str:
        return f"{self.cfg.reply_prefix} {self.derive_reply(output, user_text=user_text)}"

    def render_readback(self, output: Dict[str, Any]) -> str:
        selected_class = self.extract_selected_class(output)
        dual = self.extract_dual_confidence(output)
        confidence = dual.get("practical", None)
        structural_conf = dual.get("structural", None)
        operator = self.extract_operator(output)
        active_component = self.extract_active_component(output)
        control = self.extract_control(output)

        signature = self._signature_block(output)
        corridor = self._corridor_block(output)
        reasoning = self._reasoning_block(output)

        components = _first_present(signature, "components", default=[])
        comp_count = len(components) if isinstance(components, list) else _first_present(output, "component_count", default="n/a")
        hold_semantics = _first_present(reasoning, "hold_semantics", default="n/a")
        wrap = _first_present(corridor, "wraparound_lattice", default="n/a")

        conf_line = f"Selected class: {selected_class} (confidence={_fmt_float(confidence)})"
        if structural_conf is not None:
            conf_line = (
                f"Selected class: {selected_class} (confidence={_fmt_float(confidence)} "
                f"practical={_fmt_float(confidence)} structural={_fmt_float(structural_conf)})"
            )

        mp = _as_dict(_first_present(output, "misleading_positive", default={}))
        mp_flag = bool(_first_present(mp, "flagged", default=False))
        mp_hint = ""
        if mp_flag:
            mp_hint = f" | misleading_positive=True gap={_fmt_float(_first_present(mp, 'gap', default=None))}"

        lines = [
            conf_line + mp_hint,
            f"Operator: {operator} | Active component: {active_component}",
            (
                "Control: "
                f"caution={_fmt_float(control.get('caution'))} "
                f"(raw={_fmt_float(control.get('raw_caution'))}) | "
                f"recovery={_fmt_float(control.get('recovery'))} | "
                f"hold={control.get('hold')}"
            ),
            (
                "Readback: "
                f"comp_id={active_component} comps={comp_count} | "
                f"caution={_fmt_float(control.get('caution'))} "
                f"recovery={_fmt_float(control.get('recovery'))} "
                f"hold={control.get('hold')} | "
                f"hold_semantics={hold_semantics} wrap={wrap}"
            ),
        ]
        return "\n".join(lines)

    def render_trace_summary(self, output: Dict[str, Any], mode: str = "compact", limit: Optional[int] = None) -> str:
        trace = self.extract_trace(output)
        if not trace:
            return ""

        if limit is None:
            limit = self.cfg.default_trace_items if mode == "compact" else self.cfg.max_trace_items

        items = trace[-int(limit):]

        def fmt(item: Dict[str, Any]) -> str:
            phase = _first_present(item, "phase", "phase_index", "timestep", default="?")
            op = _first_present(item, "selected_operator", "op", "operator", default="n/a")
            shift = _first_present(item, "shift", "shift_metric", default=None)
            blocks = _first_present(item, "blocks", "blocked_count", default=0)
            caution = _first_present(item, "caution", "caution_scalar", default=None)
            recovery = _first_present(item, "recovery", "rec", default=None)
            hold = _first_present(item, "hold", default=False)
            return (
                f"phase={phase} op={op} shift={_fmt_float(shift)} "
                f"blocks={blocks} caution={_fmt_float(caution)} "
                f"rec={_fmt_float(recovery)} hold={hold}"
            )

        parts = [fmt(item) for item in items]
        if mode == "full":
            return self.cfg.trace_prefix + "\n" + "\n".join(parts)
        return self.cfg.trace_prefix + " " + " | ".join(parts)

    def render_debug_panel(self, output: Dict[str, Any]) -> str:
        trace = self.extract_trace(output)
        signature = self._signature_block(output)
        reasoning = self._reasoning_block(output)
        orientation = self._orientation_block(output)

        operator_scores = _as_dict(_first_present(orientation, "operator_scores", default={}))
        return " ".join(
            [
                "Debug:",
                f"trace_items={len(trace)}",
                f"zero_crossings={_first_present(signature, 'zero_crossings', default='n/a')}",
                f"mode_index_len={len(_as_list(_first_present(signature, 'mode_index', default=[])))}",
                f"operator_scores={len(operator_scores)}",
                f"phase_index={_first_present(reasoning, 'phase_index', default='n/a')}",
            ]
        )


def _renderer(config: Optional[Dict[str, Any]] = None) -> ResponseRenderer:
    return ResponseRenderer(config=config)


def render_response(output: Dict[str, Any], user_text: str = "", config: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    return _renderer(config).render_response(output=output, user_text=user_text)


def render_readback(output: Dict[str, Any], config: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    return _renderer(config).render_readback(output=output)


def render_trace_summary(
    output: Dict[str, Any],
    mode: str = "compact",
    limit: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> str:
    return _renderer(config).render_trace_summary(output=output,
 mode=mode, limit=limit)


def render_debug_panel(output: Dict[str, Any], config: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    return _renderer(config).render_debug_panel(output=output)
