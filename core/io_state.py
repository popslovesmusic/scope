
import json

def save_state(path, state_dict, trace):
    with open(path, "w") as f:
        json.dump({"state": state_dict, "trace": trace}, f, indent=2)


def save_session_state(path: str, session) -> None:
    """
    Save terminal session state/history without changing runtime artifact format.
    """
    payload = session.to_dict() if hasattr(session, "to_dict") else dict(session)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_session_state(path: str):
    """
    Load terminal session state/history.
    """
    from .session_state import SessionState

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return SessionState.from_dict(payload)
