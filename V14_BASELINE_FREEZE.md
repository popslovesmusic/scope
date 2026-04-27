# v14 Baseline Freeze Pack (Phase 9)

Date: 2026-04-11

This repo freezes **v14** as the executable baseline for demos and regression checks.

## Whats Frozen
- Runtime + tests under `core/`, `sim_v14_stage1.py`, and the terminal shell wrappers.
- Config pack:
  - `config/config_v14_scaffold.json` (scaffold/reference)
  - `config/config_v14_terminal.json` (terminal/research shell default)
- Baseline snapshot + manifest:
  - `baseline/v14_baseline_snapshot.json`
  - `baseline/v14_baseline_manifest.json`

## Update Policy
- Treat changes that modify `baseline/v14_baseline_snapshot.json` as **intentional breaking changes** unless explicitly approved.
- If you intentionally change behavior, regenerate the baseline snapshot with:
  - `python scripts/gen_v14_baseline_snapshot.py --write`

## Regression
- Run full tests: `python -B -m pytest -q`
- Run the baseline snapshot check only: `python -B -m pytest -q -k baseline_freeze`

