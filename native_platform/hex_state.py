
import numpy as np

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def w_to_hex(W):
    """
    Converts W=[w1, w2, w3] to #RRGGBB.
    """
    r = int(round(clamp01(W[0]) * 255))
    g = int(round(clamp01(W[1]) * 255))
    b = int(round(clamp01(W[2]) * 255))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def make_full_hex(local_W, global_W, meta_W):
    """
    Returns 'local.global.meta' hex string.
    """
    return "{}.{}.{}".format(w_to_hex(local_W), w_to_hex(global_W), w_to_hex(meta_W))
