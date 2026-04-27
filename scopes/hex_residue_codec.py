
from typing import List, Dict, Any
from .hex_trace import HexTraceFrame

def compress_hex_path(frames: List[HexTraceFrame], window_size: int = 12) -> List[str]:
    """
    Returns a sequence of unique full_hex codes in the window.
    """
    if not frames:
        return []
    
    path = []
    last_code = None
    # Take the last window_size frames
    window = frames[-window_size:] if len(frames) > window_size else frames
    
    for f in window:
        if f.full_code != last_code:
            path.append(f.full_code)
            last_code = f.full_code
    return path

def detect_repeated_hex_motifs(frames: List[HexTraceFrame]) -> Dict[str, int]:
    """
    Counts occurrences of full_hex codes in the provided frames.
    """
    counts = {}
    for f in frames:
        counts[f.full_code] = counts.get(f.full_code, 0) + 1
    return counts

def score_hex_stability(frames: List[HexTraceFrame]) -> float:
    """
    Returns a stability score [0, 1] based on hex code consistency.
    Higher score means more repeated hex codes (less flickering).
    """
    if not frames:
        return 0.0
    
    counts = detect_repeated_hex_motifs(frames)
    max_freq = max(counts.values()) if counts else 0
    
    # Ratio of most frequent hex code to total frames
    stability = max_freq / float(len(frames))
    return stability

def make_hex_residue_candidate(frames: List[HexTraceFrame]) -> Dict[str, Any]:
    """
    Summarizes a sequence of hex frames for residue memory.
    """
    if not frames:
        return {}
    
    stability = score_hex_stability(frames)
    motifs = detect_repeated_hex_motifs(frames)
    compressed = compress_hex_path(frames)
    
    return {
        "hex_stability": stability,
        "dominant_motif": max(motifs, key=motifs.get) if motifs else None,
        "motif_count": len(motifs),
        "compressed_path": compressed,
        "frame_count": len(frames)
    }
