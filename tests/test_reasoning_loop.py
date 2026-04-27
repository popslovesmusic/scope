from core.reasoning_loop import _select_operator


def test_symmetry_disabled_selects_best_raw_operator():
    scores = {"++": 1.0, "--": 2.0, "+-": 0.0, "-+": -1.0}
    op, _shift = _select_operator(scores, {"symmetry_handling": {"enabled": False}})
    assert op == "--"


def test_magnitude_invariant_false_uses_pair_average_for_group_choice():
    # If using max, group ++|-- would win (max=10). If using average, it loses (avg=0).
    scores = {"++": 10.0, "--": -10.0, "+-": 1.0, "-+": 1.0}
    op, _shift = _select_operator(
        scores,
        {
            "symmetry_handling": {
                "enabled": True,
                "family_mode": "paired",
                "inversion_pairs": [["++", "--"], ["+-", "-+"]],
                "magnitude_invariant": False,
                "orientation_distinct": True,
            }
        },
    )
    assert op in ("+-", "-+")


def test_orientation_distinct_false_collapses_to_first_member_of_best_pair():
    scores = {"++": 0.0, "--": 1.0, "+-": 0.0, "-+": 0.0}
    op, _shift = _select_operator(
        scores,
        {
            "symmetry_handling": {
                "enabled": True,
                "family_mode": "paired",
                "inversion_pairs": [["++", "--"], ["+-", "-+"]],
                "magnitude_invariant": True,
                "orientation_distinct": False,
            }
        },
    )
    assert op == "++"


def test_unrecognized_family_mode_falls_back_to_raw_scoring():
    scores = {"++": 0.0, "--": 2.0, "+-": 1.0, "-+": 1.0}
    op, _shift = _select_operator(scores, {"symmetry_handling": {"enabled": True, "family_mode": "none"}})
    assert op == "--"

