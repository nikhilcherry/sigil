"""Threshold calibration, measured offline against synthetic populations.

The point of the feature is that its reported rates are arithmetic over two
named populations rather than a number someone typed in, so these tests build
populations whose answers are known by hand and check the arithmetic.
"""

import json

import numpy as np
import pytest

from sigil.calibrate import (
    BORN_AFTER,
    Calibration,
    Distribution,
    _birth_year,
    _threshold_at_fpr,
    encode_portraits,
    genuine_similarities,
    impostor_similarities,
    lead_images,
    measure,
    sitelinks,
)
from sigil.identify import Identity, IdentityIndex


def _index(vectors, names=None):
    v = np.asarray(vectors, dtype=np.float32)
    names = names or [f"P{i}" for i in range(len(v))]
    return IdentityIndex(
        v,
        [Identity(name=n, qid=f"Q{i}", image_url="https://x/i.jpg",
                  source="en.wikipedia")
         for i, n in enumerate(names)],
        "fake",
    )


class FakeEncoder:
    name = "fake"
    model = "fake-model"


# ------------------------------------------------------------------ arithmetic


def test_a_distribution_reports_the_population_it_was_given():
    d = Distribution.of(np.array([0.0, 0.5, 1.0]), (50,))
    assert d.pairs == 3
    assert d.minimum == 0.0 and d.maximum == 1.0
    assert d.quantiles["p50"] == pytest.approx(0.5)


def test_genuine_pairs_are_every_within_identity_combination():
    sims, keys = genuine_similarities({
        "Qa": [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])],
        "Qb": [np.array([1.0, 0.0]), np.array([1.0, 0.0])],
    })
    # 3 choose 2 for Qa, 2 choose 2 for Qb.
    assert sims.size == 4
    assert sorted(round(s, 6) for s in sims) == [0.0, 0.0, 1.0, 1.0]
    assert {k[0] for k in keys} == {"Qa", "Qb"}


def test_an_identity_with_one_portrait_contributes_no_genuine_pair():
    sims, keys = genuine_similarities({"Qa": [np.array([1.0, 0.0])]})
    assert sims.size == 0 and keys == []


def test_genuine_pair_order_is_stable_across_dict_orderings():
    a, b = np.array([1.0, 0.0]), np.array([0.6, 0.8])
    first = genuine_similarities({"Qa": [a, b], "Qb": [b, a]})[1]
    second = genuine_similarities({"Qb": [b, a], "Qa": [a, b]})[1]
    assert first == second


def test_impostor_pairs_are_every_cross_identity_combination():
    sims, iu = impostor_similarities(_index([[1, 0], [0, 1], [1, 0]]))
    assert sims.size == 3  # 3 choose 2
    assert len(iu[0]) == 3
    assert sorted(round(float(s), 6) for s in sims) == [0.0, 0.0, 1.0]


def test_a_zero_vector_does_not_divide_by_zero():
    sims, _ = impostor_similarities(_index([[0, 0], [1, 0]]))
    assert np.isfinite(sims).all()


def test_the_threshold_for_a_target_false_positive_rate_is_a_quantile():
    impostor = np.linspace(0.0, 1.0, 1001)
    # 1% of this population sits at or above 0.99.
    assert _threshold_at_fpr(impostor, 0.01) == pytest.approx(0.99, abs=1e-3)


# ------------------------------------------------------------------- measuring


def _population(genuine_at, impostor_at):
    """Two-dimensional populations whose pair similarities are exactly chosen."""
    def pair(sim):
        theta = np.arccos(np.clip(sim, -1, 1))
        return [np.array([1.0, 0.0]),
                np.array([np.cos(theta), np.sin(theta)])]

    by_qid = {f"Q{i}": pair(s) for i, s in enumerate(genuine_at)}
    vectors = []
    for s in impostor_at:
        theta = np.arccos(np.clip(s, -1, 1))
        vectors += [[1.0, 0.0], [float(np.cos(theta)), float(np.sin(theta))]]
    return by_qid, _index(vectors or [[1.0, 0.0], [0.0, 1.0]])


def test_rates_at_a_threshold_are_counts_over_the_two_populations():
    by_qid, index = _population([0.9, 0.5, 0.2], [0.1])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.4)
    # Two of the three genuine pairs clear 0.4; the impostor pair does not.
    assert c.tpr == pytest.approx(2 / 3)
    assert c.genuine.pairs == 3
    assert c.fpr == 0.0


def test_measuring_with_no_genuine_pair_refuses_rather_than_reporting_zero():
    with pytest.raises(RuntimeError, match="two usable portraits"):
        measure(FakeEncoder(), _index([[1, 0], [0, 1]]),
                {"Qa": [np.array([1.0, 0.0])]}, threshold=0.4)


def test_near_identical_impostor_pairs_are_counted_and_reported_separately():
    """Two Wikidata entities for one human are a labelling artefact, not a miss."""
    by_qid, index = _population([0.9], [0.9999, 0.05])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.4)
    assert c.artefact_pairs >= 1
    assert c.fpr_excluding_artefacts < c.fpr
    assert c.artefact_examples[0]["similarity"] >= 0.99


def test_the_closest_impostor_pair_is_named_so_it_can_be_inspected():
    by_qid, _ = _population([0.9], [])
    index = _index([[1, 0], [1, 0], [0, 1]], names=["Twin A", "Twin B", "Other"])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.4)
    assert {c.artefact_examples[0]["a"], c.artefact_examples[0]["b"]} == {
        "Twin A", "Twin B"
    }


def test_the_hardest_genuine_pairs_come_back_lowest_first():
    by_qid, index = _population([0.95, 0.3, 0.6], [0.05])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.4)
    got = [h["similarity"] for h in c.hardest_genuine]
    assert got == sorted(got)
    assert got[0] == pytest.approx(0.3, abs=1e-3)


def test_the_curve_spans_the_useful_range_and_never_rises_with_the_threshold():
    by_qid, index = _population([0.9, 0.5, 0.2], [0.1, 0.6])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.4)
    assert c.curve[0]["threshold"] == pytest.approx(0.05)
    assert c.curve[-1]["threshold"] == pytest.approx(0.95)
    for a, b in zip(c.curve, c.curve[1:], strict=False):
        assert b["tpr"] <= a["tpr"] and b["fpr"] <= a["fpr"]


def test_the_provenance_of_the_genuine_set_is_recorded_not_implied():
    by_qid, index = _population([0.9, 0.5], [0.1])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.4,
                requested=50, born_after=1900)
    assert c.sampled_requested == 50
    assert c.born_after == 1900
    assert c.sampled_photographic == 2
    assert c.backend == "fake" and c.model == "fake-model"


def test_a_calibration_survives_a_save_and_load_round_trip(tmp_path):
    by_qid, index = _population([0.9, 0.4], [0.2])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.38, requested=7)
    path = tmp_path / "calibration.json"
    c.save(path)
    back = Calibration.load(path)
    assert back.to_dict() == c.to_dict()
    assert isinstance(back.genuine, Distribution)
    assert json.loads(path.read_text())["threshold"] == 0.38


def test_loading_a_missing_calibration_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="sigil calibrate"):
        Calibration.load(tmp_path / "nope.json")


# --------------------------------------------------------------------- harvest


def test_a_birth_year_is_read_from_wikidata_including_bc():
    def ent(time):
        return {"claims": {"P569": [
            {"mainsnak": {"datavalue": {"value": {"time": time}}}}]}}

    assert _birth_year(ent("+1926-04-21T00:00:00Z")) == 1926
    assert _birth_year(ent("-0356-07-20T00:00:00Z")) == -356


def test_a_missing_or_unparseable_birth_year_is_none_not_an_exception():
    assert _birth_year({}) is None
    assert _birth_year({"claims": {"P569": []}}) is None
    assert _birth_year({"claims": {"P569": [{"mainsnak": {}}]}}) is None
    assert _birth_year({"claims": {"P569": [
        {"mainsnak": {"datavalue": {"value": {"time": "+xxxx-01-01T00:00:00Z"}}}}
    ]}}) is None


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status_code = status

    def json(self):
        return self.body


class FakeSession:
    """Replays canned API payloads and records what was asked of it."""

    def __init__(self, payloads, status=200):
        self.payloads = payloads
        self.status = status
        self.asked = []

    def get(self, url, params=None, timeout=None):
        self.asked.append((url, dict(params or {})))
        return FakeResponse(self.payloads.get(url, {}), self.status)


def _entity(qid, year, wikis):
    ent = {"sitelinks": {f"{w}wiki": {"title": f"{qid}-{w}"} for w in wikis}}
    if year is not None:
        sign = "+" if year >= 0 else "-"
        ent["claims"] = {"P569": [{"mainsnak": {"datavalue": {"value": {
            "time": f"{sign}{abs(year):04d}-01-01T00:00:00Z"}}}}]}
    return ent


WIKIDATA = "https://www.wikidata.org/w/api.php"


def test_identities_from_before_photography_are_dropped_at_the_source():
    """The filter that took the measured true-positive rate from 74% to 95%.

    Left out, the hardest 'same person' pairs were a Roman bust of Alexander
    the Great against a Pompeian mosaic of him - two sculptors' opinions of one
    head, not two photographs of a face.
    """
    session = FakeSession({WIKIDATA: {"entities": {
        "Q1": _entity("Q1", 1980, ["en", "es"]),
        "Q2": _entity("Q2", -356, ["en"]),
        "Q3": _entity("Q3", None, ["en"]),
    }}})
    got = sitelinks(session, ["Q1", "Q2", "Q3"])
    assert set(got) == {"Q1"}
    assert got["Q1"] == {"enwiki": "Q1-en", "eswiki": "Q1-es"}


def test_the_era_filter_can_be_turned_off_deliberately():
    session = FakeSession({WIKIDATA: {"entities": {
        "Q1": _entity("Q1", 1980, ["en"]),
        "Q2": _entity("Q2", -356, ["en"]),
        "Q3": _entity("Q3", None, ["en"]),
    }}})
    assert set(sitelinks(session, ["Q1", "Q2", "Q3"], born_after=None)) == {
        "Q1", "Q2", "Q3"
    }


def test_the_default_era_cutoff_is_the_documented_constant():
    session = FakeSession({WIKIDATA: {"entities": {
        "old": _entity("old", BORN_AFTER - 1, ["en"]),
        "new": _entity("new", BORN_AFTER + 1, ["en"]),
    }}})
    assert set(sitelinks(session, ["old", "new"])) == {"new"}


def test_a_failed_api_batch_costs_coverage_not_the_run():
    assert sitelinks(FakeSession({}, status=503), ["Q1"]) == {}


def test_lead_images_collects_a_distinct_portrait_from_each_wikipedia():
    payloads = {
        "https://en.wikipedia.org/w/api.php": {"query": {"pages": {
            "1": {"title": "Q1-en", "original": {"source": "https://x/en.jpg"}}}}},
        "https://es.wikipedia.org/w/api.php": {"query": {"pages": {
            "1": {"title": "Q1-es", "original": {"source": "https://x/es.jpg"}}}}},
    }
    got = lead_images(FakeSession(payloads),
                      {"Q1": {"enwiki": "Q1-en", "eswiki": "Q1-es"}},
                      langs=("en", "es"))
    assert got["Q1"] == {"https://x/en.jpg", "https://x/es.jpg"}


def test_the_same_file_on_two_wikipedias_is_one_portrait_not_two():
    same = {"query": {"pages": {
        "1": {"title": "T", "original": {"source": "https://x/a.jpg"}}}}}
    got = lead_images(
        FakeSession({"https://en.wikipedia.org/w/api.php": same,
                     "https://es.wikipedia.org/w/api.php": same}),
        {"Q1": {"enwiki": "T", "eswiki": "T"}}, langs=("en", "es"))
    assert got["Q1"] == {"https://x/a.jpg"}


def test_an_article_with_no_lead_image_yields_nothing_for_that_language():
    got = lead_images(
        FakeSession({"https://en.wikipedia.org/w/api.php":
                     {"query": {"pages": {"1": {"title": "Q1-en"}}}}}),
        {"Q1": {"enwiki": "Q1-en"}}, langs=("en",))
    assert got["Q1"] == set()


def test_a_language_nobody_has_an_article_in_is_never_requested():
    session = FakeSession({})
    lead_images(session, {"Q1": {"enwiki": "Q1-en"}}, langs=("en", "ja"))
    assert not any("ja.wikipedia" in url for url, _ in session.asked)


# -------------------------------------------------------------------- encoding


class StubEncoder:
    """One face per image, its embedding chosen by the image's first byte."""

    name = "fake"
    model = "fake-model"

    def __init__(self, by_pixel):
        self.by_pixel = by_pixel
        self.calls = 0

    def detect_and_encode(self, image_bgr):
        from sigil.face import Face

        self.calls += 1
        vec = self.by_pixel.get(int(image_bgr[0, 0, 0]))
        if vec is None:
            return []
        return [Face(embedding=np.asarray(vec, dtype=np.float32),
                     bbox=[0, 0, 10, 10], det_score=0.9)]


def _stub_fetch(monkeypatch, by_url):
    import sigil.calibrate as cal

    monkeypatch.setattr(cal, "fetch_image", lambda s, u, t: by_url.get(u))
    monkeypatch.setattr(cal, "decode_image",
                        lambda blob: np.full((4, 4, 3), blob[0], dtype=np.uint8))


def test_byte_identical_portraits_are_not_counted_as_a_second_capture(monkeypatch):
    """Otherwise the encoder is scored against itself on one photograph."""
    _stub_fetch(monkeypatch, {"https://a": b"\x01", "https://b": b"\x01"})
    enc = StubEncoder({1: [1.0, 0.0]})
    got = encode_portraits(enc, {"Q1": {"https://a", "https://b"}})
    assert len(got["Q1"]) == 1
    assert enc.calls == 1, "the duplicate was decoded and encoded anyway"


def test_the_same_file_under_two_identities_is_kept_for_both(monkeypatch):
    """Dedup is per identity: one photo of two people is a member of each."""
    _stub_fetch(monkeypatch, {"https://a": b"\x01", "https://b": b"\x01"})
    got = encode_portraits(StubEncoder({1: [1.0, 0.0]}),
                           {"Q1": {"https://a"}, "Q2": {"https://b"}})
    assert len(got["Q1"]) == 1 and len(got["Q2"]) == 1


def test_a_portrait_with_no_detectable_face_is_skipped_not_fatal(monkeypatch):
    _stub_fetch(monkeypatch, {"https://a": b"\x01", "https://b": b"\x02"})
    got = encode_portraits(StubEncoder({1: [1.0, 0.0]}),
                           {"Q1": {"https://a", "https://b"}})
    assert len(got["Q1"]) == 1


def test_an_undownloadable_portrait_is_skipped_not_fatal(monkeypatch):
    _stub_fetch(monkeypatch, {"https://a": b"\x01"})
    got = encode_portraits(StubEncoder({1: [1.0, 0.0]}),
                           {"Q1": {"https://a", "https://gone"}})
    assert len(got["Q1"]) == 1


def test_an_undecodable_portrait_is_skipped_not_fatal(monkeypatch):
    import sigil.calibrate as cal

    monkeypatch.setattr(cal, "fetch_image", lambda s, u, t: b"\x01")
    monkeypatch.setattr(cal, "decode_image", lambda blob: None)
    assert encode_portraits(StubEncoder({1: [1.0, 0.0]}), {"Q1": {"https://a"}}) == {}


# ---------------------------------------------------- reading it back out


def _measured(threshold=0.38):
    by_qid, index = _population([0.9, 0.5], [0.1])
    return measure(FakeEncoder(), index, by_qid, threshold=threshold)


def test_rates_come_back_for_a_threshold_that_was_measured():
    got = _measured().rates_at(0.40)
    assert got is not None
    tpr, fpr = got
    assert 0.0 <= tpr <= 1.0 and 0.0 <= fpr <= 1.0


def test_a_threshold_outside_the_measured_range_returns_nothing():
    """Extrapolating past the data would present arithmetic as measurement."""
    assert _measured().rates_at(0.99) is None
    assert _measured().rates_at(-0.5) is None


def test_rates_snap_to_the_nearest_measured_point_not_between_two():
    c = _measured()
    at_forty = c.rates_at(0.40)
    assert c.rates_at(0.404) == at_forty
    assert c.rates_at(0.396) == at_forty


def test_a_calibration_with_no_curve_reports_nothing_rather_than_crashing():
    c = _measured()
    c.curve = []
    assert c.rates_at(0.38) is None


def test_the_match_panel_shows_the_measured_rate_when_one_exists(tmp_path,
                                                                 monkeypatch,
                                                                 evidence):
    import sigil.calibrate as cal
    from sigil.report import console, match_panel

    c = _measured()
    c.backend = evidence.probe.backend
    saved = tmp_path / "calibration.json"
    c.save(saved)
    monkeypatch.setattr(cal, "CALIBRATION_PATH", saved)

    with console.capture() as cap:
        match_panel(evidence)
    assert "measured error rate" in cap.get()


def test_the_match_panel_is_silent_when_no_calibration_has_been_run(tmp_path,
                                                                    monkeypatch,
                                                                    evidence):
    import sigil.calibrate as cal
    from sigil.report import console, match_panel

    monkeypatch.setattr(cal, "CALIBRATION_PATH", tmp_path / "absent.json")
    with console.capture() as cap:
        match_panel(evidence)
    assert "measured error rate" not in cap.get()


def test_a_calibration_from_a_different_backend_is_not_reported(tmp_path,
                                                                monkeypatch,
                                                                evidence):
    """Two recognisers put similarity on different scales; the rate would lie."""
    import sigil.calibrate as cal
    from sigil.report import console, match_panel

    c = _measured()
    c.backend = "some-other-model"
    saved = tmp_path / "calibration.json"
    c.save(saved)
    monkeypatch.setattr(cal, "CALIBRATION_PATH", saved)

    with console.capture() as cap:
        match_panel(evidence)
    assert "measured error rate" not in cap.get()


def test_a_corrupt_calibration_does_not_take_the_match_panel_down(tmp_path,
                                                                  monkeypatch,
                                                                  evidence):
    import sigil.calibrate as cal
    from sigil.report import console, match_panel

    saved = tmp_path / "calibration.json"
    saved.write_text("{ not json")
    monkeypatch.setattr(cal, "CALIBRATION_PATH", saved)

    with console.capture() as cap:
        match_panel(evidence)
    assert "Match found" in cap.get()


# --------------------------------------------- naming a face, not a pair


def test_a_clean_index_never_names_the_wrong_face():
    from sigil.calibrate import false_name_rate

    index = _index([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rate, clean, examples = false_name_rate(index, threshold=0.45)
    assert rate == 0.0 and clean == 0.0 and examples == []


def test_a_face_is_never_its_own_impostor():
    """Without excluding the diagonal every query trivially matches itself."""
    from sigil.calibrate import false_name_rate

    index = _index([[1, 0], [0, 1]])
    assert false_name_rate(index, threshold=0.99)[0] == 0.0


def test_two_faces_that_look_alike_make_both_queries_wrong():
    from sigil.calibrate import false_name_rate

    index = _index([[1, 0], [1, 0], [0, 1]], names=["A", "B", "C"])
    rate, _clean, examples = false_name_rate(index, threshold=0.45)
    assert rate == pytest.approx(2 / 3)
    assert {e["queried"] for e in examples} == {"A", "B"}


def test_one_person_indexed_twice_is_separated_from_a_real_misnaming():
    """Naming that face with the other entry's label answers a bad index."""
    from sigil.calibrate import false_name_rate

    index = _index([[1, 0], [1, 0], [0.9, 0.436], [0, 1]],
                   names=["Dupe A", "Dupe B", "Lookalike", "Other"])
    rate, clean, examples = false_name_rate(index, threshold=0.45)
    assert rate > clean, "duplicates should be excluded from the clean rate"
    dupes = {e["queried"] for e in examples if e["duplicate_entry"]}
    assert dupes == {"Dupe A", "Dupe B"}


def test_the_rate_falls_as_the_threshold_rises():
    from sigil.calibrate import false_name_rate

    index = _index([[1, 0], [0.9, 0.436], [0.6, 0.8], [0, 1]])
    loose = false_name_rate(index, threshold=0.3)[0]
    strict = false_name_rate(index, threshold=0.95)[0]
    assert loose > strict


def test_the_examples_are_ordered_worst_first():
    from sigil.calibrate import false_name_rate

    index = _index([[1, 0], [0.99, 0.141], [0.8, 0.6], [0, 1]])
    examples = false_name_rate(index, threshold=0.4)[2]
    sims = [e["similarity"] for e in examples]
    assert sims == sorted(sims, reverse=True)


def test_measure_records_the_identity_rate_alongside_the_pair_rate():
    by_qid, index = _population([0.9, 0.5], [0.1])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.38,
                identify_threshold=0.45)
    assert c.identify_threshold == 0.45
    assert c.false_name_rate is not None
    assert c.false_name_rate_excluding_artefacts is not None


def test_a_calibration_written_before_this_existed_still_loads(tmp_path):
    """The fields are defaulted so an older file parses rather than failing."""
    by_qid, index = _population([0.9, 0.5], [0.1])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.38)
    path = tmp_path / "old.json"
    data = c.to_dict()
    for key in ("identify_threshold", "false_name_rate",
                "false_name_rate_excluding_artefacts", "wrongly_named"):
        data.pop(key)
    path.write_text(json.dumps(data))

    back = Calibration.load(path)
    assert back.false_name_rate is None
    assert back.wrongly_named == []


def test_the_identity_table_quotes_the_rate_only_at_the_measured_threshold(
    tmp_path, monkeypatch
):
    import sigil.calibrate as cal
    from sigil.report import console, identity_table

    by_qid, index = _population([0.9, 0.5], [0.1])
    c = measure(FakeEncoder(), index, by_qid, threshold=0.38,
                identify_threshold=0.45)
    saved = tmp_path / "calibration.json"
    c.save(saved)
    monkeypatch.setattr(cal, "CALIBRATION_PATH", saved)

    event = {"index_size": 2, "threshold": 0.45, "hits": []}
    with console.capture() as cap:
        identity_table(event, echo=True)
    assert "named anyway" in cap.get()

    # A different bar must not borrow that number; the rate moves steeply.
    with console.capture() as cap:
        identity_table({**event, "threshold": 0.60}, echo=True)
    assert "named anyway" not in cap.get()
