# Patch 27: Recursive Motion Anchor Validation Report

Date: 2026-04-28 21:05:42

## 1. Leakage Audit & Controls
| Test | Normal Vel-PLV | Shuffled Vel-PLV | Result |
| :--- | :--- | :--- | :--- |
| Shuffle Control | 0.060 | 0.006 | PASSED |

## 2. Recursive Continuation Metrics
| ID | Test Name | PLV | Freq MAE | Floor Viol | Alignment | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S1e | S1_early | 0.028 | 0.0574 | 0.00 | 0.020 | |
| S1l | S1_late | 0.027 | 0.0790 | 0.12 | 0.016 | |
