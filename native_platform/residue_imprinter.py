
import os
import json
from core.memory_layer import (
    build_turn_residue,
    qualify_residue,
    apply_commit_gate_and_persistence,
    PersistentMemoryState
)

def qualify_and_commit(trace, state, memory, turn_id, config, metadata=None):
    """
    Qualifies the trace and commits to memory if stable.
    """
    mem_cfg = config.get("memory_layer", {})
    
    # 1. Build runtime output for building residue
    runtime_output = {
        "state": {
            "signature": {
                "caution_scalar": state.caution_scalar,
                "recovery_scalar": state.recovery_scalar,
                "hold_state": state.hold_state,
                "components": state.components
            }
        },
        "trace": trace
    }
    
    # 2. Build Residue
    residue = build_turn_residue(
        runtime_output=runtime_output,
        prompt_text=f"native_turn_{turn_id}",
        intent_category="instruction",
        reply_mode="research_shell",
        turn_id=turn_id
    )
    
    # Patch 15: Attach optional metadata (phi, hex, op_pressure) to residue
    if metadata:
        residue.metadata = metadata
        if 'phi' in metadata:
            residue.phi = metadata['phi']
        if 'hex' in metadata:
            residue.hex_code = metadata['hex']
        if 'delta_phi' in metadata:
            residue.hex_stability = 1.0 - metadata['delta_phi'] # proxy for stability
    
    # 3. Qualify
    # Use thresholds from config
    residue = qualify_residue(
        residue, 
        structured_input=False,
        epsilon=float(mem_cfg.get("qualify_epsilon", 0.1)),
        recovery_threshold=float(mem_cfg.get("qualify_recovery_threshold", 0.02)),
        max_switch_freq=float(mem_cfg.get("qualify_max_switch_freq", 0.5)),
        min_score=float(mem_cfg.get("qualify_min_score", 0.55)),
        min_admissibility=float(mem_cfg.get("qualify_min_admissibility", 0.0))
    )
    
    # 4. Commit
    memory, residue = apply_commit_gate_and_persistence(
        state=memory,
        residue=residue,
        base_duration=int(mem_cfg.get("base_persistence_duration", 3)),
        reinforce=int(mem_cfg.get("reinforce_duration", 2)),
        max_duration=int(mem_cfg.get("max_persistence_duration", 12))
    )
    
    return memory, residue
