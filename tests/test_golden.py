import json
from pathlib import Path

from gittins_reference.golden import generate, render

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "spec" / "golden.json"


class TestGoldenCorpus:
    def test_checked_in_corpus_matches_regeneration_exactly(self):
        # The whole point of the corpus: any semantic drift in the reference
        # shows up here as a failing test and a reviewable vector diff. On a
        # deliberate semantics change, regenerate with
        #   uv run python -m gittins_reference.golden
        # and review the diff like code. Running this test on every CI
        # platform is also the cross-platform bit-identity check.
        assert GOLDEN_PATH.read_text(encoding="utf-8") == render()

    def test_floats_round_trip_bitwise_through_json(self):
        # The corpus contract says "parse to doubles, compare bitwise" —
        # so the file's own floats must survive a parse/serialize cycle.
        text = GOLDEN_PATH.read_text(encoding="utf-8")
        assert json.dumps(json.loads(text), sort_keys=True, indent=1) + "\n" == text

    def test_episode_exercises_every_resolution_kind(self):
        episode = generate()["sections"]["episode"]
        kinds = {r["kind"] for r in episode["resolutions"]}
        assert kinds == {"rewarded", "expired", "censored"}
        assert len(episode["events"]) == 11
        # Nothing left open: the episode is a complete, replayable history.
        assert episode["final"]["open_ids"] == []
        # The rejected attempts cover both resolution entry points, and each
        # names either an already-resolved decision or one never made — the
        # no-op paths an independent implementation must reproduce.
        assert {r["action"] for r in episode["rejected"]} == {"learn", "censor"}
        resolved = {r["decision_id"] for r in episode["resolutions"]}
        for attempt in episode["rejected"]:
            assert attempt["decision_id"] in resolved or attempt["decision_id"] == "fleet-a:99"

    def test_corpus_is_json_clean(self):
        # No NaN/Infinity anywhere: independent parsers must not need
        # non-standard JSON extensions.
        json.loads(render(), parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
