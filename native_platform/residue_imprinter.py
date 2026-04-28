
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
    training_cfg = config.get("training_overrides", config.get("feedback", {}).get("training_overrides", {}))
    
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
    
    # Patch 15/16: Attach optional metadata (phi, hex, op_pressure) to residue
    if metadata:
        residue.metadata = metadata
        if 'phi' in metadata:
            residue.phi = metadata['phi']
        if 'hex' in metadata:
            residue.hex_code = metadata['hex']
        if 'delta_phi' in metadata:
            residue.hex_stability = 1.0 - metadata['delta_phi']
    
    # 3. Qualify
    # Patch 16: Apply training overrides if enabled
    if training_cfg.get("enabled", False):
        min_score = float(training_cfg.get("qualify_min_score", mem_cfg.get("qualify_min_score", 0.55)))
        recovery_threshold = float(training_cfg.get("qualify_recovery_threshold", mem_cfg.get("qualify_recovery_threshold", 0.02)))
        max_switch_freq = float(training_cfg.get("qualify_max_switch_freq", mem_cfg.get("qualify_max_switch_freq", 0.5)))
    else:
        min_score = float(mem_cfg.get("qualify_min_score", 0.55))
        recovery_threshold = float(mem_cfg.get("qualify_recovery_threshold", 0.02))
        max_switch_freq = float(mem_cfg.get("qualify_max_switch_freq", 0.5))

    residue = qualify_residue(
        residue, 
        structured_input=False,
        epsilon=float(mem_cfg.get("qualify_epsilon", 0.1)),
        recovery_threshold=recovery_threshold,
        max_switch_freq=max_switch_freq,
        min_score=min_score,
        min_admissibility=float(mem_cfg.get("qualify_min_admissibility", 0.0))
    )
    
    # Patch 16: Ensure diagnostics are available
    if not hasattr(residue, 'reject_reasons'):
        residue.reject_reasons = []
    if not hasattr(residue, 'qualification_reasons'):
        residue.qualification_reasons = []

    # 4. Commit
    memory, residue = apply_commit_gate_and_persistence(
        state=memory,
        residue=residue,
        base_duration=int(mem_cfg.get("base_persistence_duration", 3)),
        reinforce=int(mem_cfg.get("reinforce_duration", 2)),
        max_duration=int(mem_cfg.get("max_persistence_duration", 12))
    )
    
    return memory, residue
