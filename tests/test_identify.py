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
