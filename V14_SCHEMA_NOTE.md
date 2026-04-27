# v14 Schema Note (Practical)

This note describes the high-level structure used by `sim_v14_stage1.py` and the terminal shell (`interfaces/chat_shell.py`).

## Runtime Trace
- Trace is a list of per-phase dicts emitted by `core/reasoning_loop.py:run_reasoning()`.
- Allowed keys are enumerated in `core/trace_schema.py` as `TRACE_STEP_KEYS`.

## Artifact Shape (High Level)
The runtime output produced by `sim_v14_stage1.py` follows:
- `state.signature`: signed_field + derived overlays (amplitude/polarity/zero_crossings/components) + control scalars (caution/hold/recovery)
- `state.corridor`: window/resistances + topology/cancellation energies + lattice mode + caution/recovery influence metrics
- `state.orientation`: chosen operator + raw/diffused operator score summaries
- `state.reasoning`: phase index + phase_history (trace) + control metadata (hold_semantics, symmetry_mode_used)
- `state.output`: readout heads and confidence

This is intentionally **human-readable**, not a stable external API.

