# Inductive Transformer v2 Validation Report

Date: 2026-04-28 19:19:40

## 1. Protocol Summary
- Train Duration: 20s
- Disconnect Duration: 10s
- Recouple Duration: 10s

## 2. Key Metrics
| Phase | Mismatch Mean | Phase Error | Drift | Reinforce Rate |
| :--- | :--- | :--- | :--- | :--- |
| Train | 0.1264 | 0.5508 | 0.0495 | 0.43 |
| Disconnect | 0.0762 | 1.0000 | 0.0000 | 0.36 |
| Recouple | 0.1561 | 0.3932 | 0.0476 | 0.18 |

## 3. Analysis
- **Continuation Stability**: Disconnect mismatch is 0.6x training levels.
- **Recoupling Shock**: First frame mismatch after recouple: 0.0000
