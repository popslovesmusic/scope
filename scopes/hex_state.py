
def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def w_to_rgb(W):
    """
    W = [w1, w2, w3] where sum(W) = 1
    Maps to R, G, B [0, 255]
    """
    if not W or len(W) < 3:
        return (0, 0, 0)
    
    r = int(round(clamp01(W[0]) * 255))
    g = int(round(clamp01(W[1]) * 255))
    b = int(round(clamp01(W[2]) * 255))
    return (r, g, b)

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])

def w_to_hex(W):
    return rgb_to_hex(w_to_rgb(W))

def make_full_hex(local_W, global_W, meta_W):
    """
    Combines three W states into a 'local.global.meta' hex string.
    """
    return "{}.{}.{}".format(w_to_hex(local_W), w_to_hex(global_W), w_to_hex(meta_W))
