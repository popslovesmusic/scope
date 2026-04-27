
class FeedbackAdapter:
  def __init__(self, config):
    self.config = config.get("feedback", {})
    self.bias = 1.0

  def update(self, state, residue):
    # derive bias from admissible outputs only
    caution = getattr(state, 'caution_scalar', 0.0)
    recovery = getattr(state, 'recovery_scalar', 0.0)

    delta = (recovery - caution)
    
    # Scale adjustment by gain
    gain = self.config.get("gain", 0.5)
    self.bias += delta * gain * 0.1
    
    # Apply decay towards neutral (1.0)
    decay = self.config.get("decay", 0.95)
    self.bias = 1.0 + (self.bias - 1.0) * decay

    # clamp
    min_b = self.config.get("min_bias", 0.5)
    max_b = self.config.get("max_bias", 2.0)
    self.bias = max(min_b, min(max_b, self.bias))

    return self.bias
