from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    turn_history: List[Dict[str, Any]] = field(default_factory=list)
    runtime_history: List[Dict[str, Any]] = field(default_factory=list)
    current_seed: Optional[int] = None
    current_config_path: str = "config/config_v14_terminal.json"
    debug_mode: bool = False
    inline_trace_mode: bool = True
    last_runtime_output: Optional[Dict[str, Any]] = None
    saved_state_reference: Optional[str] = None
    memory_state: Optional[Dict[str, Any]] = None
    memory_state_path: str = "sessions/memory_state.json"
    turn_residue_path: str = "sessions/turn_residue.jsonl"
    committed_residue_path: str = "sessions/committed_residue.jsonl"

    def add_turn(self, record: Dict[str, Any]) -> None:
        self.turn_history.append(record)

    def reset(self) -> None:
        self.turn_history = []
        self.runtime_history = []
        self.last_runtime_output = None
        self.saved_state_reference = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_history": list(self.turn_history),
            "runtime_history": list(self.runtime_history),
            "current_seed": self.current_seed,
            "current_config_path": self.current_config_path,
            "debug_mode": bool(self.debug_mode),
            "inline_trace_mode": bool(self.inline_trace_mode),
            "last_runtime_output": self.last_runtime_output,
            "saved_state_reference": self.saved_state_reference,
            "memory_state": self.memory_state,
            "memory_state_path": str(self.memory_state_path),
            "turn_residue_path": str(self.turn_residue_path),
            "committed_residue_path": str(self.committed_residue_path),
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SessionState":
        s = SessionState()
        s.turn_history = list(d.get("turn_history", []) or [])
        s.runtime_history = list(d.get("runtime_history", []) or [])
        s.current_seed = d.get("current_seed", None)
        s.current_config_path = str(d.get("current_config_path", s.current_config_path))
        s.debug_mode = bool(d.get("debug_mode", False))
        s.inline_trace_mode = bool(d.get("inline_trace_mode", True))
        s.last_runtime_output = d.get("last_runtime_output", None)
        s.saved_state_reference = d.get("saved_state_reference", None)
        s.memory_state = d.get("memory_state", None)
        s.memory_state_path = str(d.get("memory_state_path", s.memory_state_path))
        s.turn_residue_path = str(d.get("turn_residue_path", s.turn_residue_path))
        s.committed_residue_path = str(d.get("committed_residue_path", s.committed_residue_path))
        return s
