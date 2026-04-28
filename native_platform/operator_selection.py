import numpy as np

OPERATORS = ['++', '--', '+-', '-+']

def apply_operator(phi, op):
    if op == '++': return phi
    if op == '--': return -phi
    if op == '+-': return np.roll(phi, 1)
    if op == '-+': return np.roll(phi, -1)
    return phi

def select_operator(phi, phi_prev):
    if phi_prev is None:
        return '++', 0.0
        
    best_op = None
    best_cost = 1e9

    for op in OPERATORS:
        candidate = apply_operator(phi, op)
        # Cost is 1.0 - cosine similarity (dot product of normalized vectors)
        dot = float(np.dot(candidate, phi_prev))
        dot = max(-1.0, min(1.0, dot))
        cost = 1.0 - dot

        if cost < best_cost:
            best_cost = cost
            best_op = op

    return best_op, best_cost
