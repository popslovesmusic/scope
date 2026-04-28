# Phase Geometry Validation Report (Patch 25)

Date: 2026-04-28 19:37:46
Raw Log Path: logs/feedback_trace_V2_RECOUPLE.jsonl

## 1. Metric Definitions
- **PLV (Phase Locking Value)**: Stability of angular relation [0, 1].
- **Phase Cosine**: Average alignment (1=in-phase, -1=anti-phase).
- **Freq MAE**: Mean velocity mismatch between teacher and student.
- **Alignment Score**: PLV penalized by frequency error.

## 2. Phase-by-Phase Geometry
| Phase | Mismatch | Cosine | PLV | Freq MAE | Slips | Floor Viol | Alignment | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 0.1831 | 0.869 | 0.878 | 0.0965 | 0 | 0.01 | 0.708 | **in_phase** |
| Disconnect | 0.1418 | 0.939 | 0.994 | 0.0574 | 0 | 0.94 | 0.880 | **frozen_false_stability** |
| Recouple | 0.1136 | 0.910 | 0.918 | 0.0892 | 0 | 0.01 | 0.754 | **in_phase** |

## 3. Claim Verification
### Overall Claim: ❌ FAILED RIGOROUS CLAIM

#### Failure Diagnosis:
- Frozen Stability: Model stopped moving to hide error.
