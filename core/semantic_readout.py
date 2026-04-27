from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return default


def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _fmt(x: Any, ndigits: int = 3) -> str:
    try:
        return f"{float(x):.{int(ndigits)}f}"
    except Exception:
        return "n/a"


@dataclass(frozen=True)
class SemanticReadoutConfig:
    enabled: bool = True
    backend: str = "local"  # local | openai_compatible
    style: str = "hs_science"
    max_sentences: int = 4
    include_followup_question: bool = True
    caution_hedge_threshold: float = 0.65
    hold_explain: bool = True
    openai_base_url: str = "https://api.openai.com"
    openai_model: str = ""
    openai_timeout_s: float = 12.0


def _load_cfg(config: Optional[Dict[str, Any]]) -> SemanticReadoutConfig:
    cfg = _as_dict(config or {})
    sr = _as_dict(_first(cfg, "semantic_readout", default={}))
    oc = _as_dict(_first(sr, "openai_compatible", "openai", default={}))
    return SemanticReadoutConfig(
        enabled=bool(_first(sr, "enabled", default=True)),
        backend=str(_first(sr, "backend", default="local")),
        style=str(_first(sr, "style", default="hs_science")),
        max_sentences=int(_first(sr, "max_sentences", default=4)),
        include_followup_question=bool(_first(sr, "include_followup_question", default=True)),
        caution_hedge_threshold=float(_first(sr, "caution_hedge_threshold", default=0.65)),
        hold_explain=bool(_first(sr, "hold_explain", default=True)),
        openai_base_url=str(_first(oc, "base_url", default="https://api.openai.com")).rstrip("/"),
        openai_model=str(_first(oc, "model", default="")),
        openai_timeout_s=float(_first(oc, "timeout_s", default=12.0)),
    )


def _extract_runtime_summary(runtime_output: Dict[str, Any]) -> Dict[str, Any]:
    state = _as_dict(_first(runtime_output, "state", default={}))
    signature = _as_dict(_first(state, "signature", default={}))
    orientation = _as_dict(_first(state, "orientation", default={}))
    reasoning = _as_dict(_first(state, "reasoning", default={}))
    out = _as_dict(_first(runtime_output, "output", default={}))

    caution = _clip01(_first(signature, "caution_scalar", default=0.0))
    raw_caution = _clip01(_first(signature, "raw_caution_scalar", default=0.0))
    recovery = _clip01(_first(signature, "recovery_scalar", default=0.0))
    hold = bool(_first(signature, "hold_state", default=False))

    return {
        "selected_class": _first(out, "selected_class", default="n/a"),
        "confidence": _first(out, "confidence", default="n/a"),
        "operator": _first(orientation, "active_operator", default="n/a"),
        "active_component_id": _first(signature, "active_component_id", default="n/a"),
        "component_count": len(_first(signature, "components", default=[]) or []),
        "caution": float(caution),
        "raw_caution": float(raw_caution),
        "recovery": float(recovery),
        "hold": bool(hold),
        "hold_semantics": _first(reasoning, "hold_semantics", default="n/a"),
    }


_SCIENCE_SNIPPETS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bphotosynthesis\b", re.I), "Photosynthesis is how plants use sunlight to turn water and carbon dioxide into sugar (stored energy), releasing oxygen as a byproduct."),
    (re.compile(r"\brespiration\b", re.I), "Cellular respiration is how cells break down sugar to make usable energy (ATP), usually using oxygen and producing carbon dioxide and water."),
    (re.compile(r"\bmitosis\b", re.I), "Mitosis is cell division that makes two identical cells, used for growth and repair."),
    (re.compile(r"\bmeiosis\b", re.I), "Meiosis is cell division that makes sperm/egg cells with half the DNA, creating genetic variation."),
    (re.compile(r"\bdna\b", re.I), "DNA is the molecule that stores genetic instructions. Genes are DNA segments that help build proteins."),
    (re.compile(r"\bevolution\b", re.I), "Evolution is change in a population over generations. Natural selection favors traits that help survival and reproduction in a given environment."),
    (re.compile(r"\bgravity\b", re.I), "Gravity is the attractive force between masses. On Earth, it pulls objects toward the planets center."),
    (re.compile(r"\bsky\b.*\bblue\b|\bwhy\b.*\bsky\b.*\bblue\b", re.I), "The sky looks blue because air molecules scatter short-wavelength (blue) light more than long-wavelength (red) light (Rayleigh scattering)."),
    (re.compile(r"\bplate tectonics\b|\btectonic\b", re.I), "Plate tectonics explains how Earths crust is split into moving plates, causing earthquakes, volcanoes, and mountain building."),
    (re.compile(r"\bclimate change\b|\bglobal warming\b", re.I), "Climate change is long-term warming and related shifts in weather patterns, mainly driven today by increased greenhouse gases from human activity."),
    (re.compile(r"\bchemical reaction\b|\breaction\b", re.I), "A chemical reaction rearranges atoms: old bonds break and new bonds form. Matter is conserved even though substances change."),
)


def _is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"hello", "hi", "hey", "yo"}


def _is_thanks(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"thanks", "thank you", "thx"}


def _is_question(text: str) -> bool:
    t = (text or "").strip()
    return t.endswith("?") or t.lower().startswith(("why ", "how ", "what ", "when ", "where "))


def _local_reply(*, prompt: str, runtime_output: Dict[str, Any], cfg: SemanticReadoutConfig) -> str:
    summary = _extract_runtime_summary(runtime_output)
    caution = float(summary["caution"])
    recovery = float(summary["recovery"])
    hold = bool(summary["hold"])

    if _is_greeting(prompt):
        return "Hiask me a science question (biology, chemistry, physics, Earth/space), and Ill explain it in a few sentences."
    if _is_thanks(prompt):
        return "Youre welcome."

    snippet = None
    for pat, text in _SCIENCE_SNIPPETS:
        if pat.search(prompt or ""):
            snippet = text
            break

    hedge = caution >= float(cfg.caution_hedge_threshold)
    sentences = []

    if snippet:
        sentences.append(snippet)
    else:
        if _is_question(prompt):
            sentences.append("Heres a quick high-school level take, plus what the v14 engine is doing under the hood.")
        else:
            sentences.append("Got it. Heres a short explanation and a quick state readback from the v14 engine.")

    if hedge:
        sentences.append("Im being a bit cautious here (moderate caution), so I may need one more detail to be precise.")

    if cfg.hold_explain and hold:
        sentences.append("The engine is in HOLD, meaning its intentionally avoiding major state updates for stability.")

    sentences.append(
        "Engine snapshot: "
        f"op={summary['operator']} "
        f"comp={summary['active_component_id']} "
        f"caution={_fmt(caution)} "
        f"recovery={_fmt(recovery)} "
        f"conf={_fmt(summary['confidence'])}."
    )

    if cfg.include_followup_question:
        if snippet:
            sentences.append("Want an example, a diagram-style description, or a practice question?")
        else:
            sentences.append("What grade level and which part should we focus on (definition, mechanism, or example)?")

    max_s = max(1, int(cfg.max_sentences))
    return " ".join(sentences[:max_s]).strip()


def _openai_compatible_reply(*, prompt: str, runtime_output: Dict[str, Any], cfg: SemanticReadoutConfig) -> Optional[str]:
    api_key = os.environ.get("SEMANTIC_READOUT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key or not cfg.openai_model:
        return None

    summary = _extract_runtime_summary(runtime_output)
    system = (
        "You are a helpful high-school science tutor. "
        "Answer concisely (25 sentences), use simple language, and ask 1 short follow-up question. "
        "If you are uncertain, say so briefly. "
        "Do not mention internal engine implementation unless the user asks."
    )
    user = (
        f"User prompt: {prompt}\n\n"
        "Context (do not expose unless asked):\n"
        + json.dumps(
            {
                "engine": "sbllm_v14",
                "operator": summary["operator"],
                "active_component_id": summary["active_component_id"],
                "caution": summary["caution"],
                "recovery": summary["recovery"],
                "hold": summary["hold"],
                "confidence": summary["confidence"],
            },
            sort_keys=True,
        )
    )

    payload = {
        "model": cfg.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.4,
        "max_tokens": 200,
    }

    req = urllib.request.Request(
        url=f"{cfg.openai_base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=float(cfg.openai_timeout_s)) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
        choices = data.get("choices",
 [])
        if choices:
            msg = choices[0].get("message", {}) or {}
            content = msg.get("content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
        return None
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        return None


def generate_reply(
    *,
    prompt: str,
    runtime_output: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a natural-language reply from runtime state.

    Deterministic local mode is the default. Optional LLM backend can be enabled
    via config + environment variables without changing engine behavior.
    """
    cfg = _load_cfg(config)
    if not cfg.enabled:
        return ""

    backend = (cfg.backend or "local").strip().lower()
    if backend in {"openai", "openai_compatible"}:
        reply = _openai_compatible_reply(prompt=prompt, runtime_output=runtime_output, cfg=cfg)
        if reply:
            return reply
        # If backend is misconfigured/unavailable, fall back to deterministic local.
        return _local_reply(prompt=prompt, runtime_output=runtime_output, cfg=cfg)

    return _local_reply(prompt=prompt, runtime_output=runtime_output, cfg=cfg)

