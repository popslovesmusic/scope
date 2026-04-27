# Native Wave-Residue Platform (v14) - Usage Document

## 1. Platform Overview
The Native Wave-Residue Platform is a weightless reasoning engine where **waves** are the primary training material and **residue topology** is the learned structure. It bridges high-performance C++ analog dynamics with the SBLLM v14 symbolic reasoning loop.

### Core Cycle:
1.  **Signal Generation:** Input waves (Audio, EEG, or synthetic) are injected into the system.
2.  **Analog Engine:** An AVX2-accelerated C++ field evolves in response to the signal.
3.  **Measurement (SignalScope):** Spatially distributed sensors measure the field's energy, gradient, and variance.
4.  **Symbolic Projection:** Measurements are encoded into **Hex Codes** (`local.global.meta`) and projected into a 12-channel **Spectral Signature**.
5.  **Reasoning (v14):** SBLLM v14 processes the signature within an admissibility corridor.
6.  **Learning (Imprinting):** Stable reasoning trajectories are "imprinted" into residue memory.
7.  **Feedback:** The reasoning state dynamically modulates the engine's physical control parameters.

---

## 2. Installation & Setup

### Requirements
*   **Python 3.11+**
*   **NumPy, SciPy, PySoundFile** (for audio/wave processing)
*   **C++ Compiler & PyBind11** (optional, for native AVX2 speed)

### Compilation (Optional)
The system includes a **Dummy Fallback** that allows it to run in pure Python if the C++ engine is not compiled. To build the native engine:
```powershell
# Requires MSVC or GCC with pybind11
g++ -O3 -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) bindings.cpp analog_universal_node_engine_avx2.cpp -o engine_bridge.so
```

---

## 3. Running the Platform

### Standard Execution
To run the full native loop (Signal -> Engine -> v14 -> Feedback):
```powershell
python -m native_platform.run_native_platform
```

### Audio Imprinting
To imprint a specific audio wave into the system memory:
```powershell
python imprint_v14.py --audio "your_file.wav" --config "config/config_v14_terminal.json"
```

---

## 4. System Architecture

### The "Hex" Face
The system represents its internal "thought" state through a readable 3-part hex string: `LOCAL.GLOBAL.META` (e.g., `#6B9301.#717B14.#90600E`).
*   **Local:** Reaction to the immediate raw wave.
*   **Global:** Averaged participation of the surrounding node field.
*   **Meta:** Long-term atmospheric stability of the system.

### The Admissibility Corridor
Reasoning only occurs within a "corridor" of validity. 
*   **Caution Rise:** If the input wave is too chaotic or unfamiliar, the system enters a "hold" state.
*   **Recovery:** If the system successfully maps the wave to an internal attractor, recovery energy increases.

### Learning (Weightless Imprinting)
Unlike traditional neural networks, there is no backpropagation. The system learns by:
1.  **Qualification:** Is the trajectory stable? Is the hex motif consistent?
2.  **Imprinting:** If qualified, the spectral signature is written to `residue_memory`.
3.  **Bias:** Future frames are biased by these residues, making familiar patterns "easier" to process (lower caution).

---

## 5. Feedback & Closed-Loop Control

The platform features a dual-layer feedback loop configured in `native_platform/feedback_config.json`:

1.  **Metric Feedback:** Rewards the engine when the reasoning loop is in a "Recovery" state and penalizes it during "Caution" spikes.
2.  **Directional Flow Feedback:** Reacts to the velocity (`V`) of the SignalScope. If the state is moving toward stability, the engine is encouraged; if it is flickering chaotically, the engine is dampened.

---

## 6. Observability & Logs

The system generates three primary output streams for every run:

1.  **`sessions/memory_state_<run_id>.json`**: The permanent "learned" residue.
2.  **`sessions/hex_trace_<run_id>.jsonl`**: Every frame's symbolic hex code and physical metrics (Coupling, Speed, Curvature).
3.  **`logs/feedback_trace_<run_id>.jsonl`**: The real-time record of how reasoning metrics modulated the engine's control pattern.

---

## 7. Exporting Code
For archival or system transfer, use the following aggregated files:
*   **`signal_scope_v14_hex_full.txt`**: The core symbolic v14 system.
*   **`native_wave_residue_platform_v14.txt`**: The full physical-to-symbolic bridge and feedback logic.

---

## 8. Interface Usage

### Command Line Interface (CLI)
The primary entry point for managing runs and data:
```powershell
# Start a new run
python -m native_platform.cli run --frames 500 --nodes 1024 --engine-steps 20

# View latest trajectory metrics
python -m native_platform.cli scope --tail 20

# Check residue memory status
python -m native_platform.cli memory

# Reset and backup memory
python -m native_platform.cli reset-memory

# Export complete run data
python -m native_platform.cli export-run --out run_bundle.zip
```

### Visual Dashboard
Read-only monitoring of the analog field and reasoning metrics:
```powershell
# Requires streamlit and pandas
streamlit run native_platform/dashboard.py
```
**Panels:**
*   Live W-state and Hex status.
*   Coupling (C) and Imbalance (E) time series.
*   Reasoning Caution vs. Recovery trends.
*   Learning/Imprint success rate.

### FastAPI Bridge
External control and state access:
```powershell
# Start the API server
uvicorn native_platform.api:app --reload
```
**Key Endpoints:**
*   `GET /status`: Current system state.
*   `GET /memory`: Full residue memory summary.
*   `POST /run`: Programmatic trigger for new imprinting runs.
*   `POST /memory/reset`: Remote memory clearing with automatic backup.
