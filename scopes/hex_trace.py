
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class HexTraceFrame:
    t: int  # frame_index_or_timestamp
    local: str
    global_hex: str  # 'global' is a keyword in python
    meta: str
    full_code: str
    C: float  # coupling
    E_scope: float  # imbalance
    speed: float  # flow_speed
    curvature: float  # trajectory_curvature
    direction_8way: str  # N|NE|E|SE|S|SW|W|NW|CENTER
    operator: str  # ++|--|+-|-+
    qualified: bool
    events: List[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        # Rename global_hex to global in dict to match schema
        d["global"] = d.pop("global_hex")
        return d

    @classmethod
    def from_dict(cls, d):
        # Handle the 'global' key from JSON
        if "global" in d:
            d["global_hex"] = d.pop("global")
        return cls(**d)

def append_hex_trace_jsonl(path: str, frame: HexTraceFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(frame.to_dict()) + "\n")

def load_hex_trace_jsonl(path: str) -> List[HexTraceFrame]:
    if not os.path.exists(path):
        return []
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frames.append(HexTraceFrame.from_dict(json.loads(line)))
    return frames
