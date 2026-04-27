from interfaces.chat_shell import ChatShell


def test_shell_accepts_prompt_and_returns_readable_response(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        keep, out = shell.process_line("hello")
        assert keep is True
        assert "Selected class:" in out
        assert "confidence=" in out
    finally:
        os.chdir(old)


def test_trace_command_returns_compact_summary(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("hello")
        keep, out = shell.process_line("/trace")
        assert keep is True
        assert "phase=" in out
    finally:
        os.chdir(old)


def test_trace_highlights_and_review_commands(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("/memory reset")
        shell.process_line("hello")
        shell.process_line("hello")
        keep, out = shell.process_line("/trace highlights")
        assert keep is True
        assert out.startswith("Trace highlights:")
        keep2, out2 = shell.process_line("/trace review")
        assert keep2 is True
        assert out2.startswith("Trace highlights:")
    finally:
        os.chdir(old)


def test_props_command_reports_trace_properties(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("hello")
        keep, out = shell.process_line("/props")
        assert keep is True
        assert out.startswith("Trace properties:")
        keep2, out2 = shell.process_line("/props full")
        assert keep2 is True
        assert "caution_rise" in out2
        assert "operator_instability" in out2
        assert "residue_rejection" in out2
    finally:
        os.chdir(old)


def test_turn_record_command_returns_json(tmp_path):
    import os
    import json as _json

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("hello")
        keep, out = shell.process_line("/turn")
        assert keep is True
        payload = _json.loads(out)
        assert "trace_record" in payload
        assert "highlight_record" in payload
        assert "raw_residue_record" in payload
        assert "committed_residue_record" in payload

        keep2, out2 = shell.process_line("/turn pack 2")
        assert keep2 is True
        payload2 = _json.loads(out2)
        assert payload2.get("id") == "v14_turn_record_pack_v1"
        assert isinstance(payload2.get("records", []), list)
        assert len(payload2.get("records", [])) >= 1
    finally:
        os.chdir(old)


def test_memory_command_is_available_when_enabled(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        keep, out = shell.process_line("/memory")
        assert keep is True
        assert out.splitlines()[0].strip() == "Memory:"
    finally:
        os.chdir(old)


def test_memory_status_reports_enabled_and_writes(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("hello")
        keep, out = shell.process_line("/memory status")
        assert keep is True
        assert "enabled_effective=True" in out
        assert "memory_state_path=" in out
        assert "turn_residue_path=" in out
        assert "committed_residue_path=" in out
        assert "last_residue_appended=True" in out
    finally:
        os.chdir(old)


def test_memory_commits_stable_repetition_greeting(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("/memory reset")
        shell.process_line("hello")
        keep, out = shell.process_line("/memory status")
        assert keep is True
        assert "committed_residue_count=1" in out
        assert (tmp_path / "sessions" / "committed_residue.jsonl").exists()
        # Next turn should inject non-zero bias (effect is mild by design).
        shell.process_line("hello")
        keep2, out2 = shell.process_line("/memory status")
        assert keep2 is True
        l1 = None
        for line in out2.splitlines():
            if line.startswith("last_operator_bias_l1="):
                try:
                    l1 = float(line.split("=", 1)[1])
                except Exception:
                    l1 = None
                break
        assert l1 is not None
        assert l1 > 0.10
    finally:
        os.chdir(old)


def test_memory_toggle_off_disables_for_session(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("hello")
        keep_off, out_off = shell.process_line("/memory off")
        assert keep_off is True
        assert "enabled_effective=False" in out_off
        keep, out = shell.process_line("/memory")
        assert keep is True
        assert out.startswith("Memory: OFF")
    finally:
        os.chdir(old)


def test_debug_toggle_changes_output_depth(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("/debug on")
        keep, out = shell.process_line("hello")
        assert keep is True
        assert "Debug:" in out
        shell.process_line("/debug off")
        keep2, out2 = shell.process_line("hello again")
        assert keep2 is True
        assert "Debug:" not in out2
    finally:
        os.chdir(old)


def test_reset_clears_session_state(tmp_path):
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("hello")
        assert shell.session.turn_history
        shell.process_line("/reset")
        assert shell.session.turn_history == []
        assert shell.session.last_runtime_output is None
    finally:
        os.chdir(old)


def test_seed_command_changes_deterministic_behavior_as_expected(tmp_path):
    import os

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir(parents=True, exist_ok=True)
    run2.mkdir(parents=True, exist_ok=True)

    old = os.getcwd()
    try:
        os.chdir(run1)
        shell = ChatShell(config_path="config/config_v14_terminal.json")
        shell.process_line("/seed 1")
        shell.process_line("hello")
        out1 = shell.session.last_runtime_output
        shell.process_line("/seed 2")
        shell.process_line("hello")
        out2 = shell.session.last_runtime_output
        assert out1["output"]["selected_class"] in (0, 1)
        assert out2["output"]["selected_class"] in (0, 1)

        os.chdir(run2)
        shell2 = ChatShell(config_path="config/config_v14_terminal.json")
        shell2.process_line("/seed 1")
        shell2.process_line("hello")
        assert shell2.session.last_runtime_output["output"]["selected_class"] == out1["output"]["selected_class"]
    finally:
        os.chdir(old)


def test_save_and_load_restores_session_state(tmp_path):
    import os

    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        shell = ChatShell(config_path="config/config_v14_terminal.json", seed=123)
        shell.process_line("hello")
        shell.process_line("world")

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        shell.process_line("/save test1")
        assert (sessions_dir / "test1.json").exists()

        shell2 = ChatShell(config_path="config/config_v14_terminal.json")
        shell2.process_line("/load test1")
        assert len(shell2.session.turn_history) == 2
        assert shell2.session.last_runtime_output is not None
    finally:
        os.chdir(old)
