
import os
import json
import numpy as np
from core.signature_state import SignatureState
from core.reasoning_loop import run_reasoning

class V14Bridge:
    def __init__(self, config_path="config/config_v14_terminal.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.signature_size = int(self.config.get("signature_size", 12))

    def run_turn(self, signature_12, orientation_bias=0.0):
        # 1. Initialize State
        state = SignatureState(self.signature_size)
        
        # Center signature and scale to [-1, 1] range for v14 signed_field
        state.signed_field = (signature_12 * 2.0) - 1.0
        state.derive_amplitude_from_signed()
        
        # 2. Run reasoning loop
        trace = run_reasoning(state, self.config)
        
        return trace, state
