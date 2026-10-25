import json

from basil3.aggregate import combine


def test_across_seed_delta_is_paired(tmp_path):
    files = []
    for seed in range(2):
        rows = [{"agent": a, "opponent": "greedy", "plan_rate": 0.1,
                 "expansions_per_move": 1, "imagination_calls_per_move": 5,
                 "ms_per_move": 2, "illegal_moves": 0}
                for a in ("morty", "citadel@voc:0.25")]
        uni = [{"agent": "morty", "opponent": "greedy", "scores": [0.2, 0.4]},
               {"agent": "citadel@voc:0.25", "opponent": "greedy", "scores": [0.3, 0.5]}]
        path = tmp_path / f"run_{seed}.json"
        path.write_text(json.dumps({"args": {"split": "test", "universes": 2},
                                    "universes": ["a", "b"], "per_universe": uni, "results": rows}))
        files.append(str(path))
    data = combine(files)
    r = next(r for r in data["results"] if r["agent"] != "morty")
    assert r["delta"] == 0.1
    assert r["delta_ci95"] == [0.1, 0.1]
