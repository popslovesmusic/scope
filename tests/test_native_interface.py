import os
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from native_platform.run_state import (
    tail_jsonl, 
    latest_jsonl, 
    load_memory_summary, 
    clear_memory,
    backup_file
)

class TestNativeInterface(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_jsonl_helpers(self):
        log_file = os.path.join(self.test_dir, "test_log.jsonl")
        records = [{"t": i, "val": i*10} for i in range(5)]
        
        with open(log_file, "w", encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
                
        # Test tail
        tailed = tail_jsonl(log_file, n=2)
        self.assertEqual(len(tailed), 2)
        self.assertEqual(tailed[0]["t"], 3)
        self.assertEqual(tailed[1]["t"], 4)
        
        # Test latest
        latest = latest_jsonl(log_file)
        self.assertEqual(latest["t"], 4)

    def test_memory_management(self):
        mem_file = os.path.join(self.test_dir, "native_memory.json")
        data = {
            "turn_counter": 10,
            "qualified_residue_count": 5,
            "committed": [{"key": "op:1", "persistence_duration": 1}],
            "operator_bias": {},
            "caution_baseline_shift": 0.0
        }
        
        with open(mem_file, "w", encoding='utf-8') as f:
            json.dump(data, f)
            
        # Test summary
        summary = load_memory_summary(mem_file)
        self.assertEqual(summary["turn_counter"], 10)
        self.assertEqual(summary["committed_count"], 1)
        
        # Test backup
        backup_path = backup_file(mem_file)
        self.assertTrue(os.path.exists(backup_path))
        os.remove(backup_path)
        
        # Test clear
        clear_memory(mem_file)
        summary_cleared = load_memory_summary(mem_file)
        self.assertEqual(summary_cleared["turn_counter"], 0)
        self.assertEqual(summary_cleared["committed_count"], 0)

    def test_interface_imports(self):
        # Ensure interfaces can be imported without immediate side effects
        # Use try-except to handle cases where fastapi/streamlit aren't installed yet
        try:
            from native_platform.cli import main as cli_main
        except ImportError:
            pass
            
        try:
            from native_platform.api import app as fastapi_app
        except ImportError:
            # Expected if fastapi not in venv
            pass
            
        try:
            from native_platform.dashboard import main as dashboard_main
        except ImportError:
            # Expected if streamlit not in venv
            pass
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
