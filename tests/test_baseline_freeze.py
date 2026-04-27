import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_baseline_snapshot_matches_current_runtime() -> None:
    from scripts.gen_v14_baseline_snapshot import compute_snapshot

    snapshot_path = REPO_ROOT / "baseline" / "v14_baseline_snapshot.json"
    assert snapshot_path.exists(), "missing baseline snapshot file; run scripts/gen_v14_baseline_snapshot.py --write"

    stored = _load_json(snapshot_path)
    computed = compute_snapshot(
        prompt=stored.get("prompt", "v14_baseline_freeze"),
        seed=int(stored.get("seed", 123)),
        config_paths=["config/config_v14_scaffold.json", "config/config_v14_terminal.json"],
    )

    stored.pop("generated_at", None)
    computed.pop("generated_at", None)

    assert computed == stored


def test_baseline_manifest_hashes_match_current_files() -> None:
    manifest_path = REPO_ROOT / "baseline" / "v14_baseline_manifest.json"
    snapshot_path = REPO_ROOT / "baseline" / "v14_baseline_snapshot.json"
    assert manifest_path.exists(), "missing baseline manifest file; run scripts/gen_v14_baseline_snapshot.py --write"
    assert snapshot_path.exists(), "missing baseline snapshot file; run scripts/gen_v14_baseline_snapshot.py --write"

    manifest = _load_json(manifest_path)

    snapshot_bytes = snapshot_path.read_bytes()
    assert manifest["snapshot"]["sha256"] == _sha256_bytes(snapshot_bytes)
    assert int(manifest["snapshot"].get("bytes", -1)) == int(len(snapshot_bytes))

    for entry in manifest.get("files", []):
        rel = entry["path"]
        path = REPO_ROOT / rel
        assert path.exists(), f"manifest lists missing file: {rel}"
        b = path.read_bytes()
        assert int(entry["bytes"]) == int(len(b)), f"byte size drift for {rel}"
        assert entry["sha256"] == _sha256_bytes(b), f"sha256 drift for {rel}"

