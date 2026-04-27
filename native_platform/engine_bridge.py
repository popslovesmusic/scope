
import numpy as np
try:
    import engine_bridge
except ImportError:
    engine_bridge = None
    print("Warning: engine_bridge.so/pyd not found. Native AVX2 engine will be unavailable.")

class EngineBridge:
    def __init__(self, num_nodes=100):
        self.num_nodes = num_nodes
        if engine_bridge:
            self.engine = engine_bridge.AnalogCellularEngineAVX2(num_nodes)
        else:
            self.engine = None
            self.dummy_outputs = np.zeros(num_nodes)

    def step(self, input_signal, control_pattern=1.0):
        """
        Processes one step of the engine.
        """
        if self.engine:
            # We use processSignalWaveAVX2 as a single step proxy if runMission is too large
            return self.engine.processSignalWaveAVX2(input_signal, control_pattern)
        else:
            # Dummy dynamics: leaky integrator + noise
            self.dummy_outputs = 0.9 * self.dummy_outputs + 0.1 * input_signal + np.random.normal(0, 0.01, self.num_nodes)
            return np.mean(self.dummy_outputs)

    def get_node_outputs(self):
        if self.engine:
            return np.array(self.engine.getNodeOutputs())
        return self.dummy_outputs

    def get_node_states(self):
        # Placeholder for more complex state extraction if needed
        return self.get_node_outputs()
