"""Identity-index behaviour, with a synthetic index so no network is needed."""

import json

import numpy as np
import pytest

from sigil import identify as idmod
from sigil.identify import Identity, IdentityIndex


def unit(*v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def index():
    people = [
        Identity("Ada Lovelace", "Q7259", "https://x/ada.jpg", "en.wikipedia"),
        Identity("Alan Turing", "Q7251", "https://x/alan.jpg", "en.wikipedia"),
        Identity("Grace Hopper", "Q11641", "https://x/grace.jpg", "en.wikipedia"),
    ]
    vecs = np.vstack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
    return IdentityIndex(vecs, people, "insightface")


def test_query_ranks_the_nearest_face_first(index):
    hits = index.query(unit(0.9, 0.1, 0.0), top=3)
    assert hits[0].identity.name == "Ada Lovelace"
    assert hits[0].similarity > hits[1].similarity > hits[2].similarity


def test_query_similarity_is_a_real_cosine(index):
    hit = index.query(unit(1, 0, 0), top=1)[0]
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)


def test_query_normalises_an_unnormalised_probe(index):
    """Embeddings arrive normalised, but a caller must not be able to skew ranking
    by passing a longer vector."""
    a = index.query(np.array([5.0, 0.0, 0.0], dtype=np.float32), top=1)[0]
    b = index.query(unit(1, 0, 0), top=1)[0]
    assert a.similarity == pytest.approx(b.similarity, abs=1e-5)


def test_query_respects_top_n(index):
    assert len(index.query(unit(1, 1, 1), top=2)) == 2


def test_an_unrelated_face_scores_below_the_naming_threshold(index):
    from sigil.pipeline import IDENTITY_THRESHOLD

    # Pointing away from every indexed face: the honest answer is "no idea".
    # (Note the diagonal unit(1,1,1) would NOT work here - it sits at cos 0.577
    # from each basis vector, which is above the bar.)
    hits = index.query(unit(-1, -1, -1), top=3)
    assert all(h.similarity < IDENTITY_THRESHOLD["insightface"] for h in hits)


def test_missing_index_says_how_to_build_one(tmp_path, monkeypatch):
    monkeypatch.setattr(idmod, "INDEX_VECTORS", tmp_path / "none.npz")
    monkeypatch.setattr(idmod, "INDEX_META", tmp_path / "none.json")
    with pytest.raises(FileNotFoundError, match="sigil index build"):
        IdentityIndex.load()


def test_backend_mismatch_is_refused_not_silently_wrong(tmp_path, monkeypatch, index):
    """Embeddings from different models are not comparable; using one index with
    the other backend would return confident nonsense."""
    vec_path, meta_path = tmp_path / "v.npz", tmp_path / "m.json"
    np.savez_compressed(vec_path, vectors=index.vectors)
    meta_path.write_text(json.dumps({
        "backend": "insightface", "model": "x", "count": 3,
        "identities": [i.to_dict() for i in index.identities],
    }))
    monkeypatch.setattr(idmod, "INDEX_VECTORS", vec_path)
    monkeypatch.setattr(idmod, "INDEX_META", meta_path)

    class FakeEncoder:
        name = "opencv"
        model = "yunet+sface"

    with pytest.raises(RuntimeError, match="rebuild it"):
        IdentityIndex.load(FakeEncoder())

    class SameEncoder:
        name = "insightface"
        model = "x"

    assert len(IdentityIndex.load(SameEncoder())) == 3


def test_non_person_titles_are_filtered_before_lookup():
    from sigil.identify import SKIP_PREFIXES

    for junk in ("Special:Search", "Wikipedia:Featured_pictures", "Portal:Current_events"):
        assert junk.startswith(SKIP_PREFIXES)
    assert not "Aamir_Khan".startswith(SKIP_PREFIXES)


# ------------------------------------------------------------------- harvesting


class FakeSession:
    """Stands in for requests.Session, dispatching on the URL like the real APIs."""

    def __init__(self, pageviews=None, pages=None, entities=None, status=200):
        self.pageviews = pageviews or {}
        self.pages = pages or {}
        self.entities = entities or {}
        self.status = status
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        session = self

        class R:
            status_code = session.status

            @staticmethod
            def json():
                if "pageviews" in url:
                    ym = "/".join(url.split("/")[-3:-1])
                    arts = session.pageviews.get(ym, [])
                    return {"items": [{"articles": [{"article": a} for a in arts]}]}
                if "wikidata.org" in url:
                    ids = params["ids"].split("|")
                    return {"entities": {i: session.entities[i]
                                         for i in ids if i in session.entities}}
                titles = params["titles"].split("|")
                return {"query": {"pages": [session.pages[t]
                                            for t in titles if t in session.pages]}}

        return R()


def _entity(qid, name, instance_of="Q5"):
    return {
        "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": instance_of}}}}]},
        "labels": {"en": {"value": name}},
    }


def _page(title, qid, img):
    return {"title": title, "pageprops": {"wikibase_item": qid},
            "original": {"source": img}}


def test_popular_titles_merges_every_month_and_drops_non_articles(monkeypatch):
    monkeypatch.setattr(idmod, "_months", lambda n: ["2026/07", "2026/08"])
    session = FakeSession(pageviews={
        "2026/07": ["Ada_Lovelace", "Special:Search"],
        "2026/08": ["Alan_Turing", "Ada_Lovelace"],
    })

    found = idmod.popular_titles(session, ["en"], months=2)

    assert found == {"en": {"Ada_Lovelace", "Alan_Turing"}}
    assert len(session.calls) == 2, "one request per (wiki, month)"


def test_popular_titles_survives_a_wiki_that_fails(monkeypatch):
    """A missing month or an unreachable wiki must not sink the whole build."""
    monkeypatch.setattr(idmod, "_months", lambda n: ["2026/08"])
    session = FakeSession(pageviews={"2026/08": ["Ada_Lovelace"]}, status=404)

    assert idmod.popular_titles(session, ["en"], months=1) == {}


def test_resolve_people_keeps_humans_and_rejects_everything_else():
    """Without the P31=Q5 check the index fills with film posters and album
    covers, which do contain faces and would be matched confidently."""
    session = FakeSession(
        pages={
            "Ada Lovelace": _page("Ada Lovelace", "Q7259", "https://x/ada.jpg"),
            "Inception": _page("Inception", "Q25188", "https://x/poster.jpg"),
        },
        entities={
            "Q7259": _entity("Q7259", "Ada Lovelace"),
            "Q25188": _entity("Q25188", "Inception", instance_of="Q11424"),  # film
        },
    )

    people = idmod.resolve_people(session, "en", ["Ada Lovelace", "Inception"])

    assert [p.name for p in people] == ["Ada Lovelace"]
    assert people[0].image_url == "https://x/ada.jpg"
    assert people[0].source == "en.wikipedia"


def test_resolve_people_skips_pages_with_no_portrait():
    """A person with no image cannot be encoded, so they are not an identity."""
    session = FakeSession(
        pages={"Nobody": {"title": "Nobody", "pageprops": {"wikibase_item": "Q1"}}},
        entities={"Q1": _entity("Q1", "Nobody")},
    )

    assert idmod.resolve_people(session, "en", ["Nobody"]) == []


def test_resolve_people_batches_are_split_and_all_merged():
    """Batching is an API limit, not a cap on results - nothing may be lost."""
    n = 95
    titles = [f"P{i}" for i in range(n)]
    session = FakeSession(
        pages={t: _page(t, f"Q{i}", f"https://x/{i}.jpg") for i, t in enumerate(titles)},
        entities={f"Q{i}": _entity(f"Q{i}", f"Name {i}") for i in range(n)},
    )

    people = idmod.resolve_people(session, "en", titles)

    assert len(people) == n
    assert [p.name for p in people] == [f"Name {i}" for i in range(n)]


# ----------------------------------------------------------------- build_index


def test_build_index_encodes_persists_and_reloads(tmp_path, monkeypatch):
    """A built index must round-trip: what was encoded is what loads back, in
    the same order, with the vectors still lined up with the names."""
    import numpy as np

    from sigil.face import Face

    people = [Identity(f"Person {i}", f"Q{i}", f"https://x/{i}.jpg", "en.wikipedia")
              for i in range(5)]

    monkeypatch.setattr(idmod, "popular_titles", lambda s, langs, months: {"en": {"t"}})
    monkeypatch.setattr(idmod, "resolve_people", lambda s, lang, titles: people)
    monkeypatch.setattr(idmod, "fetch_image", lambda s, url, t: url.encode())
    monkeypatch.setattr(idmod, "decode_image", lambda blob: blob)
    monkeypatch.setattr(idmod, "INDEX_VECTORS", tmp_path / "v.npz")
    monkeypatch.setattr(idmod, "INDEX_META", tmp_path / "m.json")
    monkeypatch.setattr(idmod, "MODELS_DIR", tmp_path)

    class FakeEncoder:
        name = "insightface"
        model = "buffalo_l/w600k_r50"

        def detect_and_encode(self, img):
            # A distinct unit vector per portrait, keyed off the URL's index.
            i = int(img.decode().rsplit("/", 1)[-1].split(".")[0])
            v = np.zeros(5, dtype=np.float32)
            v[i] = 1.0
            return [Face(embedding=v, bbox=[0, 0, 10, 10], det_score=0.9)]

    encoder = FakeEncoder()
    count = idmod.build_index(encoder, langs=["en"], months=1)

    assert count == 5
    index = IdentityIndex.load(encoder)
    assert len(index) == 5
    assert [i.name for i in index.identities] == [f"Person {i}" for i in range(5)]
    # Vector i must still belong to person i - a reordering here would attach
    # every name to the wrong face.
    for i in range(5):
        hit = index.query(np.eye(5, dtype=np.float32)[i], top=1)[0]
        assert hit.identity.name == f"Person {i}"


def test_build_index_skips_portraits_with_no_usable_face(tmp_path, monkeypatch):
    """Vectors and identities must stay in lockstep when some portraits drop."""
    import numpy as np

    from sigil.face import Face

    people = [Identity(f"P{i}", f"Q{i}", f"https://x/{i}.jpg", "en.wikipedia")
              for i in range(4)]

    monkeypatch.setattr(idmod, "popular_titles", lambda s, langs, months: {"en": {"t"}})
    monkeypatch.setattr(idmod, "resolve_people", lambda s, lang, titles: people)
    # Portrait 1 will not download; portrait 2 downloads but has no face.
    monkeypatch.setattr(idmod, "fetch_image",
                        lambda s, url, t: None if "/1." in url else url.encode())
    monkeypatch.setattr(idmod, "decode_image", lambda blob: blob)
    monkeypatch.setattr(idmod, "INDEX_VECTORS", tmp_path / "v.npz")
    monkeypatch.setattr(idmod, "INDEX_META", tmp_path / "m.json")
    monkeypatch.setattr(idmod, "MODELS_DIR", tmp_path)

    class FakeEncoder:
        name = "insightface"
        model = "m"

        def detect_and_encode(self, img):
            if "/2." in img.decode():
                return []
            v = np.zeros(4, dtype=np.float32)
            v[int(img.decode().rsplit("/", 1)[-1].split(".")[0])] = 1.0
            return [Face(embedding=v, bbox=[0, 0, 10, 10], det_score=0.9)]

    encoder = FakeEncoder()
    assert idmod.build_index(encoder, langs=["en"], months=1) == 2

    index = IdentityIndex.load(encoder)
    assert [i.name for i in index.identities] == ["P0", "P3"]
    assert index.query(np.eye(4, dtype=np.float32)[3], top=1)[0].identity.name == "P3"


def test_build_index_refuses_to_write_an_empty_index(tmp_path, monkeypatch):
    """An index of nothing would load fine and silently name no one."""
    monkeypatch.setattr(idmod, "popular_titles", lambda s, langs, months: {"en": {"t"}})
    monkeypatch.setattr(
        idmod, "resolve_people",
        lambda s, lang, titles: [Identity("P", "Q", "https://x/0.jpg", "en.wikipedia")],
    )
    monkeypatch.setattr(idmod, "fetch_image", lambda s, url, t: None)
    monkeypatch.setattr(idmod, "INDEX_VECTORS", tmp_path / "v.npz")
    monkeypatch.setattr(idmod, "MODELS_DIR", tmp_path)

    class FakeEncoder:
        name = "insightface"
        model = "m"

        def detect_and_encode(self, img):
            return []

    with pytest.raises(RuntimeError, match="network"):
        idmod.build_index(FakeEncoder(), langs=["en"], months=1)


def test_build_index_respects_the_limit(tmp_path, monkeypatch):
    import numpy as np

    from sigil.face import Face

    people = [Identity(f"P{i}", f"Q{i}", f"https://x/{i}.jpg", "en.wikipedia")
              for i in range(10)]
    monkeypatch.setattr(idmod, "popular_titles", lambda s, langs, months: {"en": {"t"}})
    monkeypatch.setattr(idmod, "resolve_people", lambda s, lang, titles: people)
    monkeypatch.setattr(idmod, "fetch_image", lambda s, url, t: url.encode())
    monkeypatch.setattr(idmod, "decode_image", lambda blob: blob)
    monkeypatch.setattr(idmod, "INDEX_VECTORS", tmp_path / "v.npz")
    monkeypatch.setattr(idmod, "INDEX_META", tmp_path / "m.json")
    monkeypatch.setattr(idmod, "MODELS_DIR", tmp_path)

    class FakeEncoder:
        name = "insightface"
        model = "m"

        def detect_and_encode(self, img):
            return [Face(embedding=np.ones(3, dtype=np.float32),
                         bbox=[0, 0, 10, 10], det_score=0.9)]

    assert idmod.build_index(FakeEncoder(), langs=["en"], months=1, limit=3) == 3


def test_an_interrupted_build_keeps_what_it_already_encoded(tmp_path, monkeypatch):
    """Encoding a full index is the better part of an hour. Throwing that away
    on a Ctrl-C is the wrong answer to "this is taking too long"."""
    import numpy as np

    from sigil.face import Face

    people = [Identity(f"P{i}", f"Q{i}", f"https://x/{i}.jpg", "en.wikipedia")
              for i in range(10)]
    monkeypatch.setattr(idmod, "popular_titles", lambda s, langs, months: {"en": {"t"}})
    monkeypatch.setattr(idmod, "resolve_people", lambda s, lang, titles: people)
    monkeypatch.setattr(idmod, "fetch_image", lambda s, url, t: url.encode())
    monkeypatch.setattr(idmod, "decode_image", lambda blob: blob)
    monkeypatch.setattr(idmod, "INDEX_VECTORS", tmp_path / "v.npz")
    monkeypatch.setattr(idmod, "INDEX_META", tmp_path / "m.json")
    monkeypatch.setattr(idmod, "MODELS_DIR", tmp_path)

    class InterruptingEncoder:
        name = "insightface"
        model = "m"

        def detect_and_encode(self, img):
            i = int(img.decode().rsplit("/", 1)[-1].split(".")[0])
            if i == 3:
                raise KeyboardInterrupt
            v = np.zeros(10, dtype=np.float32)
            v[i] = 1.0
            return [Face(embedding=v, bbox=[0, 0, 10, 10], det_score=0.9)]

    encoder = InterruptingEncoder()
    assert idmod.build_index(encoder, langs=["en"], months=1) == 3

    index = IdentityIndex.load(encoder)
    assert [i.name for i in index.identities] == ["P0", "P1", "P2"]
    assert json.loads((tmp_path / "m.json").read_text())["partial"] is True


def test_an_interrupt_before_any_face_still_refuses_to_write(tmp_path, monkeypatch):
    """A partial index of nothing is just an empty index by another name."""
    monkeypatch.setattr(idmod, "popular_titles", lambda s, langs, months: {"en": {"t"}})
    monkeypatch.setattr(
        idmod, "resolve_people",
        lambda s, lang, titles: [Identity("P", "Q", "https://x/0.jpg", "en.wikipedia")],
    )
    monkeypatch.setattr(idmod, "fetch_image", lambda s, url, t: url.encode())
    monkeypatch.setattr(idmod, "decode_image", lambda blob: blob)
    monkeypatch.setattr(idmod, "INDEX_VECTORS", tmp_path / "v.npz")
    monkeypatch.setattr(idmod, "MODELS_DIR", tmp_path)

    class Interrupting:
        name = "insightface"
        model = "m"

        def detect_and_encode(self, img):
            raise KeyboardInterrupt

    with pytest.raises(RuntimeError, match="network"):
        idmod.build_index(Interrupting(), langs=["en"], months=1)


def test_a_partial_index_says_so_when_loaded(tmp_path, monkeypatch, index):
    """"No match" against a partial index may only mean "not harvested yet"."""
    vec_path, meta_path = tmp_path / "v.npz", tmp_path / "m.json"
    np.savez_compressed(vec_path, vectors=index.vectors)
    meta_path.write_text(json.dumps({
        "backend": "insightface", "model": "x", "count": 3, "partial": True,
        "identities": [i.to_dict() for i in index.identities],
    }))
    monkeypatch.setattr(idmod, "INDEX_VECTORS", vec_path)
    monkeypatch.setattr(idmod, "INDEX_META", meta_path)

    assert IdentityIndex.load().partial is True


def test_an_index_without_the_flag_is_not_partial(tmp_path, monkeypatch, index):
    """Indexes built before the flag existed must not read as interrupted."""
    vec_path, meta_path = tmp_path / "v.npz", tmp_path / "m.json"
    np.savez_compressed(vec_path, vectors=index.vectors)
    meta_path.write_text(json.dumps({
        "backend": "insightface", "model": "x", "count": 3,
        "identities": [i.to_dict() for i in index.identities],
    }))
    monkeypatch.setattr(idmod, "INDEX_VECTORS", vec_path)
    monkeypatch.setattr(idmod, "INDEX_META", meta_path)

    assert IdentityIndex.load().partial is False


def test_a_wiki_that_raises_does_not_sink_the_other_wikis(monkeypatch):
    """The harvest fans out across ten language editions. One of them throwing
    must cost that wiki's titles, not the build."""
    monkeypatch.setattr(idmod, "_months", lambda n: ["2026/08"])

    class FlakySession(FakeSession):
        def get(self, url, params=None, timeout=None):
            if "hi.wikipedia" in url:
                raise ConnectionError("hi.wikipedia is having a day")
            return super().get(url, params, timeout)

    session = FlakySession(pageviews={"2026/08": ["Ada_Lovelace"]})

    found = idmod.popular_titles(session, ["en", "hi", "ta"], months=1)

    assert set(found) == {"en", "ta"}
    assert found["en"] == {"Ada_Lovelace"}


def test_a_failed_title_batch_costs_only_that_batch(monkeypatch):
    """Batching is an API limit. A batch that throws must not take the rest of
    the language's people with it."""
    titles = [f"P{i}" for i in range(95)]

    class FlakySession(FakeSession):
        # Keyed on the batch's contents, not on call order: the batches are
        # fetched concurrently, so "the second call" is not "the second batch".
        def get(self, url, params=None, timeout=None):
            if "wikidata.org" not in url and "pageviews" not in url:
                if "P40" in (params or {}).get("titles", "").split("|"):
                    raise TimeoutError("that batch timed out")
            return super().get(url, params, timeout)

    session = FlakySession(
        pages={t: _page(t, f"Q{i}", f"https://x/{i}.jpg") for i, t in enumerate(titles)},
        entities={f"Q{i}": _entity(f"Q{i}", f"Name {i}") for i in range(95)},
    )

    people = idmod.resolve_people(session, "en", titles)

    # 95 titles in batches of 40; losing the second batch loses 40 of them.
    assert len(people) == 55
    assert "Name 0" in {p.name for p in people}
    assert "Name 94" in {p.name for p in people}


def test_a_failed_wikidata_batch_costs_only_that_batch(monkeypatch):
    titles = [f"P{i}" for i in range(95)]

    class FlakySession(FakeSession):
        def get(self, url, params=None, timeout=None):
            if "wikidata.org" in url and "Q40" in (params or {}).get("ids", "").split("|"):
                raise ConnectionError("wikidata refused")
            return super().get(url, params, timeout)

    session = FlakySession(
        pages={t: _page(t, f"Q{i}", f"https://x/{i}.jpg") for i, t in enumerate(titles)},
        entities={f"Q{i}": _entity(f"Q{i}", f"Name {i}") for i in range(95)},
    )

    people = idmod.resolve_people(session, "en", titles)

    assert len(people) == 55


# ---------------------------------------------- which months get harvested


def test_months_returns_the_last_complete_months_newest_first(monkeypatch):
    """Date arithmetic driving which pageview data a build asks for.

    If this drifted the build would query months that do not exist and harvest
    nothing, which looks like a network problem rather than a bug. The current
    month is deliberately excluded: its pageview data is still accruing.
    """
    import datetime as _dt

    import sigil.identify as ident

    class Sept(_dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 9, 15)

    monkeypatch.setattr(_dt, "date", Sept)
    assert ident._months(3) == ["2026/08", "2026/07", "2026/06"]


def test_months_crosses_a_year_boundary(monkeypatch):
    """January is where naive month arithmetic gives 2026/00 or 2026/12."""
    import datetime as _dt

    import sigil.identify as ident

    class Jan(_dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 1, 3)

    monkeypatch.setattr(_dt, "date", Jan)
    assert ident._months(4) == ["2025/12", "2025/11", "2025/10", "2025/09"]


def test_months_starting_from_the_first_of_a_month_still_looks_back(monkeypatch):
    """The boundary the implementation walks off: day 1 of the current month."""
    import datetime as _dt

    import sigil.identify as ident

    class FirstOfMarch(_dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 3, 1)

    monkeypatch.setattr(_dt, "date", FirstOfMarch)
    assert ident._months(2) == ["2026/02", "2026/01"]


def test_months_of_none_is_empty_not_an_error():
    import sigil.identify as ident

    assert ident._months(0) == []


def test_months_are_zero_padded_so_the_api_accepts_them(monkeypatch):
    """Wikimedia's pageviews path wants 2026/08, not 2026/8."""
    import datetime as _dt

    import sigil.identify as ident

    class Oct(_dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 20)

    monkeypatch.setattr(_dt, "date", Oct)
    got = ident._months(3)
    assert got == ["2026/09", "2026/08", "2026/07"]
    assert all(len(m) == 7 for m in got)


# ------------------------------ the index is two files that must correspond


def _write_index(tmp_path, monkeypatch, vectors, names, *, hash_them=True):
    import json as _json

    import numpy as np

    import sigil.identify as ident

    vec_path = tmp_path / "identity-index.npz"
    meta_path = tmp_path / "identity-index.json"
    matrix = np.array(vectors, dtype=np.float32)
    np.savez_compressed(vec_path, vectors=matrix)
    meta = {
        "backend": "insightface", "model": "m", "count": len(names),
        "langs": ["en"], "months": 3,
        "identities": [{"name": n, "qid": f"Q{i}", "image_url": "https://x",
                        "source": "en.wikipedia"} for i, n in enumerate(names)],
    }
    if hash_them:
        meta["vectors_sha256"] = ident.vectors_digest(matrix)
    meta_path.write_text(_json.dumps(meta))
    monkeypatch.setattr(ident, "INDEX_VECTORS", vec_path)
    monkeypatch.setattr(ident, "INDEX_META", meta_path)
    return vec_path, meta_path


def test_a_matched_pair_of_index_files_loads(tmp_path, monkeypatch):
    import sigil.identify as ident

    _write_index(tmp_path, monkeypatch, [[1, 0], [0, 1]], ["A", "B"])
    index = ident.IdentityIndex.load()
    assert len(index) == 2
    assert [i.name for i in index.identities] == ["A", "B"]


def test_more_vectors_than_names_is_refused(tmp_path, monkeypatch):
    """A build interrupted between the two writes leaves exactly this."""
    import sigil.identify as ident

    _write_index(tmp_path, monkeypatch, [[1, 0], [0, 1], [1, 1]], ["A", "B"],
                 hash_them=False)
    with pytest.raises(RuntimeError, match="3 vectors against 2 names"):
        ident.IdentityIndex.load()


def test_the_same_number_of_names_from_a_different_build_is_refused(tmp_path,
                                                                    monkeypatch):
    """The case a count cannot catch, and the one that matters.

    Two builds of the same size pair entirely different people, so loading one
    file from each attaches the wrong person's name to every face - with full
    confidence, which is the worst output this tool can produce.
    """
    import json as _json

    import numpy as np

    import sigil.identify as ident

    vec_path, meta_path = _write_index(
        tmp_path, monkeypatch, [[1, 0], [0, 1]], ["A", "B"])

    # Same shape, different faces - as a rebuild would produce.
    np.savez_compressed(vec_path,
                        vectors=np.array([[0, 1], [1, 0]], dtype=np.float32))

    assert len(_json.loads(meta_path.read_text())["identities"]) == 2
    with pytest.raises(RuntimeError, match="do not match the hash"):
        ident.IdentityIndex.load()


def test_an_index_written_before_the_hash_existed_still_loads(tmp_path,
                                                              monkeypatch):
    """Only the length check applies to it; refusing outright would be worse."""
    import sigil.identify as ident

    _write_index(tmp_path, monkeypatch, [[1, 0], [0, 1]], ["A", "B"],
                 hash_them=False)
    assert len(ident.IdentityIndex.load()) == 2


def test_the_build_writes_both_files_through_a_temporary(tmp_path, monkeypatch):
    """So an interrupted build cannot leave a half-written file in place."""
    import pathlib

    seen = []
    real_replace = pathlib.Path.replace

    def spy(self, target):
        seen.append((self.name, pathlib.Path(target).name))
        return real_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", spy)
    _run_build(tmp_path, monkeypatch)

    assert len(seen) == 2, f"expected two atomic renames, saw {seen}"
    assert all(src.endswith(".tmp") or ".tmp." in src for src, _ in seen), seen


def test_the_built_index_records_the_hash_and_reloads(tmp_path, monkeypatch):
    import json as _json

    import sigil.identify as ident

    meta_path = _run_build(tmp_path, monkeypatch)
    meta = _json.loads(meta_path.read_text())
    assert len(meta["vectors_sha256"]) == 64
    # And the pair it wrote passes its own check.
    assert len(ident.IdentityIndex.load()) == meta["count"]


def _run_build(tmp_path, monkeypatch):
    """Drive build_index with the network and the encoder stubbed out."""
    import numpy as np

    import sigil.identify as ident

    vec_path = tmp_path / "identity-index.npz"
    meta_path = tmp_path / "identity-index.json"
    monkeypatch.setattr(ident, "INDEX_VECTORS", vec_path)
    monkeypatch.setattr(ident, "INDEX_META", meta_path)
    monkeypatch.setattr(ident, "MODELS_DIR", tmp_path)

    people = [ident.Identity(name=f"P{i}", qid=f"Q{i}",
                             image_url=f"https://x/{i}.jpg", source="en.wikipedia")
              for i in range(3)]
    monkeypatch.setattr(ident, "popular_titles", lambda s, langs, months: {"en": {"t"}})
    monkeypatch.setattr(ident, "resolve_people", lambda s, lang, titles: people)
    monkeypatch.setattr(ident, "fetch_image", lambda s, u, t: b"bytes")
    monkeypatch.setattr(ident, "decode_image",
                        lambda b: np.zeros((8, 8, 3), dtype=np.uint8))

    class Enc:
        name, model = "insightface", "m"

        def detect_and_encode(self, img):
            from sigil.face import Face

            return [Face(embedding=np.array([1.0, 0.0], dtype=np.float32),
                         bbox=[0, 0, 1, 1], det_score=0.9)]

    ident.build_index(Enc(), langs=("en",), months=1)
    return meta_path
