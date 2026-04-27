# v14 Status (Stage 1 ? Stage 2 ? Phase 1 ? Phase 2 ? Phase 3)

Date: 2026-04-11

## Scope
Stage 1 implements the `v14.json` first milestone only surface: 12-slot signature field, 4 orientation operators, dynamic corridor derived from peaks/bands, bounded writeback reasoning loop, and a corridor-supported binary readout with a readable per-phase trace.

## Implemented
- **Signature-native state**
  - `core/signature_state.py`: `SignatureState` with `amplitude`, `phase`, `stability`, `step`, `input_trace`.
  - Update rule (conceptual): `(1-?)*signature + ?*gated_delta - ?*boundary_penalty` with amplitude clipped to `[0, 1]`.
- **Peak + band detection**
  - `core/peak_detector.py`: `detect_peaks_and_bands()` with `min_height`, `min_distance`, `merge_radius`, and simple band expansion.
- **Dynamic corridor + gating**
  - `core/corridor_gate.py`: peak/band-derived corridor window + entry/exit resistance; threshold-based admissibility blocks entry updates outside the corridor.
- **Orientation operators**
  - `core/orientation_ops.py`: `++`, `--`, `+-`, `-+` as read/write transforms (stage-1 write is inverse of read) with phase shift metadata.
- **Reasoning loop + trace**
  - `core/reasoning_loop.py`: per-phase operator scoring, symmetry-aware selection, gating, writeback, stability/phase updates, and a phase-local trace record.
- **Binary readout**
  - `core/readout_heads.py`: binary class scores using corridor-supported energy over configured family heads; outputs `class_scores`, `selected_class`, `confidence`.
- **Runner + artifact**
  - `sim_v14_stage1.py`: deterministic input encoding, runs reasoning phases, builds a final corridor snapshot, emits `v14_stage1_state_v1` + trace to `--out`.
- **Config + tests**
  - `config/config_v14_scaffold.json`: adds `peak_detection`, `projection`, `corridor`, `symmetry_handling`, `output_model`.
  - `tests/`: replaces placeholders with basic correctness tests; `python -m pytest -q` passes locally.

## Stage 2 Patch: Native Diffusion (Applied)
Patch source: `v14_patch.json` (`v14_stage2_add_diffusion`)

### What changed
- **Family-space diffusion term**
  - `core/diffusion_ops.py`: 1D family-lattice Laplacian + diffusion update with boundary modes `wrap|reflect|clamp`.
  - `core/reasoning_loop.py`: diffusion runs *after* writeback and *before* phase-local peak/band refresh.
- **Bands now reflect topology**
  - `core/peak_detector.py`: optional wraparound band expansion (`use_wraparound_lattice`) so bands can span the 12-slot boundary.
- **Corridor derives more from band support**
  - `core/corridor_gate.py`: corridor width and window bump incorporate detected band width and support mass (when amplitude is provided).
- **Trace extended**
  - `core/trace_schema.py`: adds `diffusion_energy`, `spread_measure`, `band_widths_after`.

### Config knobs added (defaults in scaffold)
- `family_diffusion` (default `0.12`): diffusion strength in family space.
- `diffusion_boundary_mode` (default `"wrap"`): lattice boundary behavior for diffusion.
- `use_wraparound_lattice` (default `true`): cyclic lattice for peaks/bands/corridor distances and topology helpers.
- `orientation_diffusion` (default `0.0`): optional score-space diffusion across the four operator scores before final selection.
- `hold_semantics` (default `"decay"`): when hold triggers, either allow relaxation/diffusion (`"decay"`) or freeze the signature surface for that phase (`"freeze"`).

### Observed result (smoke run)
Repro:
`python sim_v14_stage1.py --config config/config_v14_scaffold.json --input "diffusion demo" --out v14_output.json`

Example last-phase trace values from that run:
- `diffusion_energy`: `0.195971...`
- `spread_measure`: `1.633095...`
- `band_widths_after`: `[8, 8]`

## Implications
- **Bands should become nontrivial** even when peak thresholds are conservative, because diffusion produces connected support rather than only peak-adjacent expansion.
- **Corridor becomes topology-coupled**: wider / higher-support bands now directly widen the admissibility window, reducing brittle peak-only gating.
- **Stability vs. distinctness tradeoff**: higher `family_diffusion` increases smoothing (more stable bands/corridor) but can reduce sharp peak separation and may flatten class-head energy differences.
- **Trace becomes diagnostic**: `diffusion_energy` and `spread_measure` let you verify diffusion is active (nonzero) and not numerically inert.

## Phase 1 Patch: Polarized Signature Consolidation (Applied)
Patch source: `v14_patch2.json` (`v14_phase1_polarized_signature_consolidation`)

### What changed
- **Polarized overlay fields (additive)**
  - `core/signature_state.py`: adds `polarity`, `zero_crossings`, `mode_index`, `component_summary` as derived fields.
  - `core/polarity_ops.py`: polarity assignment, zero-crossing detection, run-label mode index, and a lightweight dominant component summary.
- **Reasoning trace extended**
  - `core/reasoning_loop.py`: recomputes polarity/zero-crossings after diffusion (using a signed source field) and records:
    - `dominant_polarity_before/after`
    - `zero_crossings_before/after`
    - `polarity_shift`
    - `dominant_component_summary_after`
- **Corridor bias hooks (small)**
  - `core/corridor_gate.py`: optional `corridor_polarity_consistency_bonus` and `corridor_zero_crossing_bonus` to gently bias the corridor without replacing band support.
- **Artifact output**
  - `sim_v14_stage1.py`: writes the new polarized fields into `state.signature.*`.

### Config knobs added (defaults in scaffold)
- `polarity_threshold` (default `0.01`)
- `zero_crossing_threshold` (default `0.02`)
- `corridor_zero_crossing_bonus` (default `0.05`)
- `corridor_polarity_consistency_bonus` (default `0.05`)
- `enable_polarized_signature_summary` (default `true`)

### Observed result (smoke run)
Repro:
`python sim_v14_stage1.py --config config/config_v14_scaffold.json --input "patch2 demo" --out v14_output.json`

Observed (example output from that run):
- `state.signature.polarity`: present (multi-slot +/0/- assignment)
- `state.signature.mode_index`: present (signed run labels)
- `state.signature.component_summary`: present (dominant family/phase/polarity + nearest zero-crossing if any)
- Trace includes `dominant_polarity_before/after` and `polarity_shift` (example run showed `polarity_shift = 0.0` in last phase)

## Implications (Phase 1)
- **Interpretability increases**: the same peak/band/corridor mechanics can now be read as *lobe structure* rather than only energy support.
- **Cancellation axes become explicit**: zero crossings (when present) provide a concrete candidate for shared emission/cancellation axes without changing readout.
- **Backward compatibility preserved**: peak/band outputs, corridor fields, diffusion metrics, and binary readout remain intact; the new fields are additive.
- **Polarity is derived from signed dynamics**: because amplitude is clipped to `[0, 1]`, polarity must be computed from a signed source (current implementation uses signed writeback + diffusion term). This is the main conceptual bridge toward v15 without destabilizing v14.

## How to run
- Demo run: `python sim_v14_stage1.py --config config/config_v14_scaffold.json --input "demo" --out v14_output.json`
- Tests: `python -m pytest -q`

## Output format (high level)
The output JSON (`--out`) contains:
- `state.version`: `v14_stage1_state_v1`
- `state.signature`: amplitude/phase/stability plus peak/band summaries
- `state.corridor`: window + entry/exit resistance + threshold
- `state.orientation`: selected operator + operator scores
- `state.reasoning.phase_history`: per-phase trace steps (operator choice, peak summaries, corridor hits/blocks, decision shift)
- `state.output`: class scores, selected class, confidence

## Known limitations (Stage 1)
- Band model is intentionally simple (non-wrapping band expansion; minimal band influence on corridor).
- Corridor penalty and gating are heuristic (intended to be readable + stable, not a final physics).
- Operator scoring is based on gated update energy minus boundary penalty; no learned scoring head.
- Tokens are summaries is represented via peaks/bands + trace, not a token generator.
- Only one scaffold config is present (the additional profiles referenced in `v14.json` are not created yet).

## Known limitations (Phase 1 polarized overlay)
- Zero crossings may be rare for some configs/inputs because the signed source is based on update dynamics, not a persistent signed amplitude field.
- Corridor polarity/zero-crossing bonuses are intentionally small and should remain so until a more stable polarized state representation is introduced.

## Phase 2 Patch: Persistent Signed Field (Applied)
Patch source: `v14_patch3.json` (`v14_phase2_persistent_signed_field`)

### What changed
- **`signed_field` is now primary**
  - `core/signature_state.py`: adds `signed_field` (persistent) and derives `amplitude = abs(signed_field)` for compatibility.
- **Diffusion and polarized summaries are intrinsic**
  - `core/reasoning_loop.py`: applies family diffusion to `signed_field`, then recomputes `polarity` and `zero_crossings` from the persistent signed surface (not from phase-local deltas).
- **Artifact output now includes `signed_field`**
  - `sim_v14_stage1.py`: persists `state.signature.signed_field` alongside derived amplitude and polarized diagnostics.

### Implications
- **Polarity and zero crossings become replay-stable** because they are computed from the persistent signed surface.
- **Amplitude is no longer the computational primitive**; it is an interpretive magnitude view for peaks/bands/corridor/readout compatibility.
- **Corridor remains backward compatible**: peaks/bands/corridor/readout still operate over magnitude (`abs(signed_field)`), while polarity/zero-crossing structure feeds optional corridor biases.

### Acceptance criteria checklist
- `signed_field`
 persists across phases: yes (primary state field, diffused in-place each phase).
- `amplitude` is derived rather than primary: yes (`amplitude = abs(signed_field)`).
- Polarity stable under replay: yes (derived from `signed_field`).
- Zero crossings structural: yes (derived from sign transitions of `signed_field`).
- Corridor backward compatible: yes (still uses magnitude peaks/bands with additive polarity/zero-crossing biases).
- Artifact includes signed_field + diagnostics: yes (`state.signature.signed_field` plus `polarity`, `zero_crossings`, `mode_index`, `component_summary`).

## Phase 3 Patch: Corridor Reanchoring v1 (Applied)
Patch set: `v14_phase3_corridor_reanchoring_v1` (proposed-next ? applied)

### What changed
- **Signed-topology corridor biasing (still compatible)**
  - `core/corridor_gate.py`: after the existing peak/band base window is built, a signed-topology support field and a localized zero-crossing penalty field are computed and blended in (before normalization).
  - Adds optional entry gating: when enabled, entry updates are reduced in topology-poor locations even if magnitude window is high.
- **Polarity run extraction**
  - `core/polarity_ops.py`: `extract_polarity_runs()` returns contiguous nonzero polarity regions (including wraparound merge).
- **Trace + artifact diagnostics**
  - `core/reasoning_loop.py` / `core/trace_schema.py`: emits `topology_support_energy`, `cancellation_penalty_energy`, `signed_run_count`, `largest_signed_run_width`, `corridor_topology_bias_applied`.
  - `sim_v14_stage1.py`: persists corridor summary metrics into `state.corridor.*`.

### Config knobs added (defaults in scaffold)
- `corridor_topology_support_gain` (default `0.18`)
- `corridor_zero_crossing_penalty` (default `0.12`)
- `corridor_run_support_gain` (default `0.10`)
- `corridor_crossing_decay` (default `1.5`)
- `corridor_require_signed_support_for_entry` (default `true`)

### Observed result (smoke run)
Repro:
`python sim_v14_stage1.py --config config/config_v14_scaffold.json --input "phase3 demo" --out v14_output.json`

Example last-phase trace values from that run:
- `topology_support_energy`: `5.183144...`
- `cancellation_penalty_energy`: `4.581490...`
- `signed_run_count`: `3`
- `largest_signed_run_width`: `4`
- `corridor_topology_bias_applied`: `true`

### Implications
- **Corridor is now topology-first (optionally)**: contiguous signed lobes widen admissibility over their interior; cancellation axes can narrow admissibility locally.
- **Backward compatibility remains**: setting the topology gains to `0.0` keeps corridor behavior approximately the same as the magnitude-only base window.
- **Entry becomes structurally admissible**: with `corridor_require_signed_support_for_entry=true`, entering a slot requires not just magnitude corridor openness but also local signed structure support.

## Next steps (suggested)
- Add the missing config profiles (`config_v14_binary_demo.json`, `config_v14_corridor_low/mid/high.json`) and a small symmetry-aware demo set.
- Improve band tracking (identity across phases; wrap-around bands on the lattice).
- Extend trace tooling (`scripts/run_v14_phase_trace.py`) to render phase summaries/plots.
- Add deterministic replay metadata in the saved artifact (seed, config hash, numpy version).
