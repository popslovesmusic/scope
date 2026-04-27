# SBLLM v14 (Terminal)  Student-Friendly Instructions

This folder is a small research/demo project that lets you type text prompts and watch a **v14 signed-field** reasoning engine run step-by-step. You can use it like a simple chat app in the terminal, and you can inspect a compact trace when youre curious.

## 1) One-time setup

You need **Python 3.11+**.

Install the two main dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pytest
```

## 2) The easiest way to use it (interactive terminal shell)

From the repo root:

```powershell
python scripts/run_sbllmv14_cli.py
```

You should see:
`sbllm v14 shell. Type /help for commands.`

Now type a normal prompt like:
`hello`  
or  
`explain photosynthesis in one sentence`

### Helpful shell commands

- `/help`  show command list
- `/trace`  short trace summary for the last turn
- `/trace 5`  last 5 phase summaries
- `/trace full`  detailed trace for the last turn
- `/state`  compact state snapshot
- `/debug on` / `/debug off`  more/less diagnostics
- `/seed 123`  make the run deterministic (repeatable)
- `/config config/config_v14_terminal.json`  switch config
- `/save mysession`  saves to `sessions/mysession.json`
- `/load mysession`  loads from `sessions/mysession.json`
- `/reset`  clears the session history
- `/quit`  exit

## Optional: richer science answers (LLM backend)

By default, the shell uses a **deterministic local** `semantic_readout` reply (works offline).

To enable an optional OpenAI-compatible backend:

1) Set an API key in your environment:
   - PowerShell: `setx OPENAI_API_KEY "your_key_here"`
2) Edit `config/config_v14_terminal.json`:
   - Set `semantic_readout.backend` to `openai_compatible`
   - Set `semantic_readout.openai_compatible.model` to the model you want to use

If the backend is not configured (missing key/model) or is unavailable, it automatically falls back to the local deterministic reply.

## Optional: continuation memory (no transcript replay)

The shell can keep a small persistent memory state that biases the next turn **without replaying old text**.

- Enable it in `config/config_v14_terminal.json`:
  - Set `memory_layer.enabled` to `true`
- Inspect it in the shell:
  - `/memory`
  - `/memory reset`

Files (when enabled):
- `sessions/memory_state.json` (persistent low-bandwidth memory)
- `sessions/turn_residue.jsonl` (append-only distilled per-turn residues)

## 3) Run the engine once (non-interactive)

This runs one prompt and writes an output JSON artifact:

```powershell
python sim_v14_stage1.py --config config/config_v14_scaffold.json --input "hello" --seed 123 --out v14_output.json
```

## 4) Run the test suite

```powershell
python -B -m pytest -q
```

## 5) Baseline freeze pack (for regression checks)

This repo includes a baseline snapshot so you can detect accidental behavior changes.

- Generate/update the baseline files:

```powershell
python scripts/gen_v14_baseline_snapshot.py --write
```

- Files written:
  - `baseline/v14_baseline_snapshot.json`
  - `baseline/v14_baseline_manifest.json`

If tests fail because the baseline changed, you usually have two options:
1) Fix the code so behavior matches the stored baseline again, or  
2) If the behavior change is *intentional*, regenerate the baseline with the command above.

## What youre looking at (plain language)

- The engine keeps a numeric field state and updates it in phases.
- It can become more cautious (caution), sometimes pause changes (hold), and optionally recover (recovery).
- The trace commands show you *what happened each phase* without dumping huge arrays.

If you want, tell me what youre trying to demo (school talk? science fair? research notes?) and Ill tailor a best default config + best commands cheat-sheet for that use case.
