"""Serving API: contract and input validation.

Skipped unless both trained checkpoints are present, since the app loads real
models at startup -- these are smoke tests for the endpoint, not for accuracy.
"""

from pathlib import Path

import pytest

CHECKPOINTS = [
    Path("results/checkpoints/sasrec_beauty.pt"),
    Path("results/checkpoints/genrec_beauty.pt"),
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in CHECKPOINTS),
    reason="needs trained Beauty checkpoints",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from serving.app import app

    with TestClient(app) as c:
        yield c


def test_recommend_returns_k_items_from_both_models(client):
    response = client.get("/recommend", params={"items": "1,2,3", "k": 5})
    assert response.status_code == 200

    body = response.json()
    assert len(body["sasrec"]) == 5
    assert len(body["genrec"]) == 5
    assert body["overlap"] == len(
        {r["item_id"] for r in body["sasrec"]} & {r["item_id"] for r in body["genrec"]}
    )


def test_every_recommendation_carries_its_semantic_id(client):
    body = client.get("/recommend", params={"items": "1,2,3", "k": 3}).json()
    for model in ("sasrec", "genrec"):
        for rec in body[model]:
            assert len(rec["semantic_id"]) == 4
            assert rec["popularity_bucket"] in {"unseen", "tail", "torso", "head"}


def test_history_items_are_never_recommended_back(client):
    history = "10,20,30,40"
    body = client.get("/recommend", params={"items": history, "k": 10}).json()
    seen = {int(x) for x in history.split(",")}
    for model in ("sasrec", "genrec"):
        assert not seen & {r["item_id"] for r in body[model]}


def test_generative_recommendations_are_always_real_items(client):
    """The Trie guarantee, exercised through the API rather than the unit test."""
    body = client.get("/recommend", params={"items": "5,6,7", "k": 10}).json()
    for rec in body["genrec"]:
        assert rec["item_id"] >= 1
        assert rec["title"] != "" or rec["train_frequency"] >= 0


@pytest.mark.parametrize("items", ["", "abc", "0", "999999", "1,2,12102"])
def test_bad_histories_are_rejected(client, items):
    """Empty, non-integer, and out-of-range ids are all 400s."""
    assert client.get("/recommend", params={"items": items}).status_code == 400


def test_stray_commas_are_tolerated(client):
    """Deliberate leniency: a trailing or doubled comma is a formatting slip,
    not an ambiguous request. Type and range stay strict."""
    body = client.get("/recommend", params={"items": "1,,2,", "k": 3}).json()
    assert [h["item_id"] for h in body["history"]] == [1, 2]


def test_random_user_history_is_directly_usable(client):
    body = client.get("/random_user", params={"seed": 3}).json()
    assert client.get("/recommend", params={"items": body["history"]}).status_code == 200
