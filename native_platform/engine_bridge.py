
import numpy as np
try:
    import engine_bridge
except ImportError:
    engine_bridge = None
    # print("Warning: engine_bridge.so/pyd not found. Native AVX2 engine will be unavailable.")

class EngineBridge:
    def __init__(self, num_nodes=100):
        self.num_nodes = num_nodes
        if engine_bridge:
            self.engine = engine_bridge.AnalogCellularEngineAVX2(num_nodes)
        else:
            self.engine = None
            self.dummy_outputs = np.zeros(num_nodes)
            self.integrator_state = np.zeros(num_nodes)
            self.reaction_enabled = True
            self.corridor_enabled = True

    def step(self, input_signal, control_pattern=1.0):
        """
        Processes one step of the engine (AVX2 if available).
        """
        if self.engine:
            return self.engine.processSignalWaveAVX2(input_signal, control_pattern)
        else:
            # Dummy dynamics mimicking C++ logic with toggles
            dt = 1.0 / 48000.0
            amplified = input_signal * control_pattern
            if self.reaction_enabled:
                amplified += 0.05 * np.tanh(self.integrator_state)
            
            self.integrator_state += amplified * 0.1 * dt
            self.integrator_state *= 0.999999
            
            if self.corridor_enabled:
                # Mock corridor gating
                mask = np.abs(self.integrator_state) > 8.0
                self.integrator_state[mask] *= 0.5
            
            # Simple output mapping
            self.dummy_outputs = self.integrator_state + np.random.normal(0, 0.001, self.num_nodes)
            return np.mean(self.dummy_outputs)

    def step_scalar(self, input_signal, control_pattern=1.0):
        """
        Processes one step using scalar reference path.
        """
        if self.engine:
            # Placeholder if we don't have a single step scalar exposed yet
            # In C++ we might need to expose it. 
            # For now we'll just run a 1-step mission if possible, 
            # but usually we use this for comparison.
            return self.step(input_signal, control_pattern) 
        else:
            # For dummy mode, scalar and normal are same
            return self.step(input_signal, control_pattern)

    def evolve(self, input_signal, control_pattern=1.0, steps=20):
        last = 0.0
        for _ in range(int(steps)):
            last = self.step(input_signal, control_pattern)
        return last

    def get_node_outputs(self):
        if self.engine:
            return np.array(self.engine.getNodeOutputs())
        return self.dummy_outputs

    def set_reaction_enabled(self, enabled):
        if self.engine:
            self.engine.setReactionEnabled(enabled)
        else:
            self.reaction_enabled = enabled

    def set_corridor_enabled(self, enabled):
        if self.engine:
            self.engine.setCorridorEnabled(enabled)
        else:
            self.corridor_enabled = enabled

    def get_field_statistics(self, previous_outputs=None):
        if self.engine:
            prev = previous_outputs.tolist() if previous_outputs is not None else []
            stats = self.engine.getFieldStatistics(prev)
            return {
                "mean": stats.mean,
                "variance": stats.variance,
                "gradient_energy": stats.gradient_energy,
                "state_delta": stats.state_delta,
                "total_energy": stats.total_energy
            }
        else:
            # Python implementation of field stats
            current = self.dummy_outputs
            mean = np.mean(current)
            variance = np.var(current)
            delta = np.mean(np.abs(current - previous_outputs)) if previous_outputs is not None else 0.0
            grad = np.mean(np.diff(current)**2) if len(current) > 1 else 0.0
            return {
                "mean": mean,
                "variance": variance,
                "gradient_energy": grad,
                "state_delta": delta,
                "total_energy": np.sum(current**2)
            }

    def set_integrator_state(self, value):
        if self.engine:
            self.engine.setIntegratorState(value)
        else:
            self.integrator_state.fill(value)
