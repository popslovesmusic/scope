
def operator_pressure(continuation_mismatch, continuation_mismatch_prev, C, E, V):
    """
    Maps phase flow into v14 operator pressure without forcing operator selection.
    
    Rising mismatch = continuation divergence pressure (+-)
    Falling mismatch = recoupling/recovery pressure (-+)
    Stable low mismatch = alignment pressure (++)
    High imbalance + high mismatch = inversion pressure (--)
    """
    pressure = {'++': 0.0, '--': 0.0, '+-': 0.0, '-+': 0.0}
    
    # 1. Rising mismatch -> rising divergence pressure (+-)
    if continuation_mismatch > continuation_mismatch_prev:
        pressure['+-'] += (continuation_mismatch - continuation_mismatch_prev) * 2.0
    
    # 2. Falling mismatch -> recovery pressure (-+)
    if continuation_mismatch < continuation_mismatch_prev:
        pressure['-+' ] += (continuation_mismatch_prev - continuation_mismatch) * 2.0
        
    # 3. Low mismatch and high coupling -> stability pressure (++)
    if continuation_mismatch < 0.1 and C > 0.8:
        pressure['++'] += (1.0 - continuation_mismatch) * 0.5
        
    # 4. High imbalance and high mismatch -> inversion pressure (--)
    if E > 0.5 and continuation_mismatch > 0.5:
        pressure['--'] += E * continuation_mismatch
        
    # Normalize pressures
    total = sum(pressure.values())
    if total > 1e-9:
        for k in pressure:
            pressure[k] /= total
            
    return pressure
