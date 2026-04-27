
def residue_bias(residue):
  if not residue or not getattr(residue, 'is_committed', False):
    return 1.0

  # Extract stability metrics from the v14 residue
  score = getattr(residue, 'stability_score', 0.0)
  # hex_stability was added in previous steps to the residue object
  stability = getattr(residue, 'hex_stability', 0.0)

  return 1.0 + 0.2 * (score + stability)
