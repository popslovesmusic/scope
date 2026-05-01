# Continuation Stress Test Report (Patch 33-35 - Sweep: Default)

Date: 2026-04-29 20:54:16

Config: w_trace=0.75, w_inductive=0.15, w_anchor=0.03, damping=0.92, experimental_rotation=False

## 1. Controls
| Control | Normal VelCoherence | Shuffled VelCoherence | Result |
| :--- | :--- | :--- | :--- |
| Shuffle | 0.054 | -0.092 | PASSED |

## 2. Stress Test Results
| ID | Test Name | VelCoherence | StateSim | Freq MAE | Alignment | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S1e | S1_early | 0.009 | 0.248 | 0.4591 | 0.000 | DRIFTING |
| S1l | S1_late | 0.015 | 0.250 | 0.4654 | 0.000 | DRIFTING |
| S3 | S3 | -0.035 | 0.173 | 0.2321 | -0.000 | DRIFTING |
