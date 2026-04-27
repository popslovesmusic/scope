import os
import json
import shutil
from datetime import datetime

def latest_jsonl(path_pattern):
    """
    Finds the most recent JSONL file matching the pattern and returns its last record.
    """
    # pattern example: "logs/feedback_trace_*.jsonl"
    import glob
    files = glob.glob(path_pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            return None
        return json.loads(lines[-1])

def tail_jsonl(path, n=20):
    """
    Returns the last N records from a JSONL file.
    """
    if not os.path.exists(path):
        return []
    
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-n:]:
            if line.strip():
                records.append(json.loads(line))
    return records

def find_latest_feedback_trace(logs_dir='logs'):
    import glob
    files = glob.glob(os.path.join(logs_dir, "feedback_trace_*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def find_latest_hex_trace(sessions_dir='sessions'):
    import glob
    files = glob.glob(os.path.join(sessions_dir, "hex_trace_*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_memory_summary(path='sessions/native_memory.json'):
    if not os.path.exists(path):
        return {"error": "Memory file not found"}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return {
        "turn_counter": data.get("turn_counter", 0),
        "qualified_residue_count": data.get("qualified_residue_count", 0),
        "committed_count": len(data.get("committed", [])),
        "operator_bias": data.get("operator_bias", {}),
        "caution_baseline_shift": data.get("caution_baseline_shift", 0.0)
    }

def backup_file(path):
    if not os.path.exists(path):
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.{timestamp}.bak"
    shutil.copy2(path, backup_path)
    return backup_path

def clear_memory(path='sessions/native_memory.json'):
    if os.path.exists(path):
        backup_file(path)
        # Reset to initial empty state
        initial_state = {
            "turn_counter": 0,
            "qualified_residue_count": 0,
            "committed": [],
            "operator_bias": {},
            "caution_baseline_shift": 0.0
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(initial_state, f, indent=2)
        return True
    return False
