
# v14 Spectral Signature Native

## Overview
v14 shifts from token-led continuation to signature-led state evolution.
Signature is primary. Tokens are derived summaries.

## Run
```bash
python sim_v14_stage1.py --config config/config_v14_scaffold.json --input "demo text" --out v14_output.json
```

## Structure
- core/: core logic
- config/: configs
- scripts/: helper runners
- tests/: basic tests

## First milestone
- 12-slot signature
- 4 operators
- corridor from peaks
- binary readout
- reasoning trace
