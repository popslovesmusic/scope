# SBLLM v14  Status & Function List

Date: 2026-04-12

This repo is the **v14** executable baseline: signed-field-first runtime + topology corridor + components + trace salience/caution/hold/recovery + terminal shell + baseline freeze pack.

## Status

- **Engine:** stable (`core/` + `sim_v14_stage1.py`)
- **Shell:** stable (`interfaces/chat_shell.py`, `scripts/run_sbllmv14_cli.py`)
- **Trace:** complete and inspectable (compact + full)
- **Semantic reply:** present (deterministic local; optional OpenAI-compatible backend)
- **Memory layer (continuation bias):** implemented as **residue-based continuation training (v14.1)**; enabled by default in terminal config, disabled by default in scaffold config
- **Regression anchor:** baseline snapshot + manifest + pytest coverage

## Primary Entry Points (What You Run)

- Terminal shell: `python scripts/run_sbllmv14_cli.py`
- One-shot run (writes artifact JSON): `python sim_v14_stage1.py --config config/config_v14_scaffold.json --input "hello" --seed 123 --out v14_output.json`
- Regenerate baseline regression snapshot: `python scripts/gen_v14_baseline_snapshot.py --write`
- Run tests: `python -B -m pytest -q`

## Shell Commands (Interactive)

In `python scripts/run_sbllmv14_cli.py`:

- `/help`  command list
- `/trace`  compact last-turn trace
- `/trace 5`  last 5 phases
- `/trace full`  full per-phase dicts (no raw array spam by default)
- `/trace highlights`  show inferred triggered highlights (recent turns)
- `/trace review`  verbose highlight review (recent turns)
- `/props`  show triggered trace properties (inferred)
- `/props full`  show all properties + supporting fields
- `/turn`  show last-turn turn_record JSON (trace + highlights + residue + commit outcome)
- `/turn pack N`  show recent turn_record pack JSON
- `/state`  compact state readback
- `/debug on|off`  debug panel
- `/memory`  memory panel (only if enabled)
- `/memory reset`  clear persistent memory state
- `/reset`  reset session history
- `/seed <n>`  deterministic seed
- `/config <path>`  switch config JSON
- `/save <name>` / `/load <name>`  session save/load in `sessions/`
- `/quit`  exit

## Core Runtime Functions (Capabilities)

### Signed-field-first dynamics

- Persistent `signed_field` is the core state (`core/signature_state.py`).
- Family diffusion runs on `signed_field`, then amplitude is re-derived (`core/diffusion_ops.py`, `core/reasoning_loop.py`).

### Operators / orientation family

- Orientation operators: `++`, `--`, `+-`, `-+` (`core/orientation_ops.py`).
- Optional *score-space* operator diffusion (orientation diffusion) before selection (`core/diffusion_ops.py:diffuse_operator_scores`, `core/reasoning_loop.py`).
- Symmetry handling knobs are consumed during selection (`config/*` + `core/reasoning_loop.py:_select_operator`).

### Corridor gating (topology-aware admissibility)

- Dynamic corridor window and resistances (`core/corridor_gate.py`).
- Topology support, cancellation penalties, signed-run support.
- Wrap vs non-wrap lattice behavior is threaded through topology helpers (`use_wraparound_lattice`).
- Caution and recovery can modulate corridor tightness/reopening.

### Components (persistent structural targets)

- Extract signed components, match across phases, pick active component (`core/component_ops.py`, `core/reasoning_loop.py`).
- Component-local targeting blends local/global influence.
- Component-local caution scoring is configurable (`mean|peak|blended`).

### Trace salience ? caution ? hold ? recovery

- Trace salience (recent window) and trace similarity (`core/trace_ops.py`).
- Caution field and scalar:
  - `raw_caution_scalar` vs bounded/applied caution
  - `caution_release_scalar`
  - `caution_after_recovery`
- Hold:
  - threshold/carry reasons
  - `hold_semantics` = `decay` or `freeze`
  - controlled release under recovery (`hold_release_counter`, `hold_release_reason`)
- Recovery:
  - `recovery_scalar` and `recontextualization_score`
  - recovery field for corridor modulation

## Readout / Output

- Binary demo readout head (`core/readout_heads.py`) for quick end-of-turn scalar output.
- Terminal readback shows operator, component id, caution/recovery/hold, confidence.
- When memory is enabled, each normal turn also prints a one-line `MemoryDelta:` hint showing whether a residue committed and what priors/shift are now present.

## Semantic Reply (Natural-Language Projection)

### Deterministic local mode (default)

- `core/semantic_readout.py` generates a short high-school science reply from:
  - prompt + state snapshot (operator, caution, recovery, hold, confidence)
  - simple topic snippets for common science questions

### Optional OpenAI-compatible backend (opt-in)

- Backend is optional and auto-falls back to local if not configured.
- Config: `semantic_readout.backend = "openai_compatible"`
- Environment variables:
  - `OPENAI_API_KEY` (or `SEMANTIC_READOUT_API_KEY`)
- Config also needs:
  - `semantic_readout.openai_compatible.base_url`
  - `semantic_readout.openai_compatible.model`

### Intent-aware conversational routing

- `interfaces/response_renderer.py` includes a lightweight intent classifier so prompts like `how are you?` produce a short conversational response.
- Science/explanation prompts still prefer explicit `runtime_output["reply"]` (from `core/semantic_readout.py`) when present.

## Memory Layer (Continuation Bias; Not Trace)

Implemented as a residue-qualified continuation bias layer (no transcript replay, no gradients):

- Files:
  - `core/memory_layer.py`
  - `sessions/memory_state.json` (persistent memory state)
  - `sessions/turn_residue.jsonl` (append-only per-turn residues)
- v14.1 training definition:
  - write **raw residue** every turn
  - **qualify** via stability evaluator + ratchet gate
  - **commit** only qualified residue into persistent memory with persistence duration + decay/reinforcement
- Minimal v1 biases (derived from committed residue only):
  - `operator_bias` (soft priors on operator selection)
  - `caution_baseline_shift` (soft baseline on bounded caution)
- Injection points:
  - **Before reasoning:** `interfaces/chat_shell.py` injects priors into config (`memory_*` keys)
  - **In reasoning:** `core/reasoning_loop.py` applies mild additive priors
- Defaults:
  - `config/config_v14_scaffold.json` ? `memory_layer.enabled=false`
  - `config/config_v14_terminal.json` ? `memory_layer.enabled=true`

## Baseline Freeze Pack (Regression Anchor)

- `baseline/v14_baseline_snapshot.json`  deterministic compact snapshot
- `baseline/v14_baseline_manifest.json`  sha256/bytes manifest for baseline-relevant files
- Generator: `scripts/gen_v14_baseline_snapshot.py`
- Tests:
  - `tests/test_baseline_freeze.py` asserts snapshot/manifest match current code/config

## Tests

- Pytest suite in `tests/` covers:
  - diffusion ops, corridor gating, polarity/zero crossings
  - components, trace salience/caution/hold/recovery
  - shell commands + save/load
  - semantic readout (local + backend fallback)
  - memory layer primitives
  - baseline freeze regression

## Key Config Files

- `config/config_v14_scaffold.json`  reference runtime config (semantic readout + memory disabled by default)
- `config/config_v14_terminal.json`  terminal defaults (semantic readout enabled local; memory enabled + visible)

## Known Limits (By Design)

- Readout head is a demo classifier; it is not a full conversational model.
- Optional LLM backend is best-effort and not required for deterministic operation.
- Memory layer is intentionally low-bandwidth and deformational; it should remain weaker than the immediate input unless explicitly tuned.
