from core.semantic_readout import generate_reply


def test_semantic_readout_local_greeting_reply():
    out = {"state": {"signature": {}}, "output": {"selected_class": 0, "confidence": 0.5}}
    reply = generate_reply(
        prompt="hello",
        runtime_output=out,
        config={"semantic_readout": {"enabled": True, "backend": "local", "max_sentences": 3}},
    )
    assert isinstance(reply, str)
    assert reply.strip()
    assert "science" in reply.lower()


def test_semantic_readout_includes_engine_snapshot():
    out = {
        "state": {
            "signature": {"caution_scalar": 0.2, "recovery_scalar": 0.1, "hold_state": False, "active_component_id": 0, "components": []},
            "orientation": {"active_operator": "++"},
        },
        "output": {"selected_class": 1, "confidence": 0.75},
    }
    reply = generate_reply(
        prompt="why is the sky blue?",
        runtime_output=out,
        config={"semantic_readout": {"enabled": True, "backend": "local", "max_sentences": 4, "include_followup_question": False}},
    )
    assert "Engine snapshot:" in reply
    assert "caution=" in reply
    assert "recovery=" in reply


def test_semantic_readout_openai_backend_falls_back_without_key_or_model():
    out = {"state": {"signature": {}}, "output": {"selected_class": 0, "confidence": 0.5}}
    reply = generate_reply(
        prompt="test",
        runtime_output=out,
        config={"semantic_readout": {"enabled": True, "backend": "openai_compatible", "openai_compatible": {"model": ""}}},
    )
    assert isinstance(reply, str)
    assert reply.strip()

