
def operator_pressure(delta_phi, delta_phi_prev, C, E, V):
    """
    Maps phase flow into v14 operator pressure without forcing operator selection.
    """
    pressure = {'++': 0.0, '--': 0.0, '+-': 0.0, '-+': 0.0}
    
    # 1. Rising mismatch -> rising divergence pressure (+-)
    if delta_phi > delta_phi_prev:
        pressure['+-'] += (delta_phi - delta_phi_prev) * 2.0
    
    # 2. Falling mismatch -> recovery pressure (-+)
    if delta_phi < delta_phi_prev:
        pressure['-+' ] += (delta_phi_prev - delta_phi) * 2.0
        
    # 3. Low mismatch and high coupling -> stability pressure (++)
    if delta_phi < 0.1 and C > 0.8:
        pressure['++'] += (1.0 - delta_phi) * 0.5
        
    # 4. High imbalance and high mismatch -> inversion pressure (--)
    if E > 0.5 and delta_phi > 0.5:
        pressure['--'] += E * delta_phi
        
    # Normalize pressures
    total = sum(pressure.values())
    if total > 1e-9:
        for k in pressure:
            pressure[k] /= total
    else:
        # Default to neutral/forward
        pressure['++'] = 1.0
            
    return pressure
