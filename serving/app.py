"""FastAPI demo: both recommenders side by side, with the semantic ID decoded.

    uv run uvicorn serving.app:app --reload
    open http://127.0.0.1:8000/

The point of showing both models is not a leaderboard -- Week 6 already
established that the generative one loses on Beauty. It is that the *shape* of
the disagreement is visible per request: the generative model's top-10 tends to
be more popular and less varied, which is the diversity collapse the tables
report, made concrete on a single sequence.

Every response also carries each recommendation's semantic ID, so the codes stop
being an abstraction: items sharing a prefix should look related.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.beam_search import batched_beam_search
from src.data.dataset import build_eval_input
from src.data.genrec_dataset import build_eval_input_tokens
from src.eval.cold_start import bucket_of, item_train_frequency
from src.semantic_ids.vocab import SemanticIdVocab
from src.train import make_score_fn, pick_device
from src.utils import load_processed
from scripts.compare_atomic_vs_semantic import load_genrec, load_sasrec

SASREC_CONFIG = "configs/sasrec_beauty.yaml"
GENREC_CONFIG = "configs/genrec_beauty.yaml"

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_everything()
    yield
    STATE.clear()


app = FastAPI(title="From SASRec to TIGER — Beauty recommender demo", lifespan=lifespan)


class Recommendation(BaseModel):
    item_id: int
    title: str
    semantic_id: list[int]
    train_frequency: int
    popularity_bucket: str
    score: float


class Response(BaseModel):
    history: list[dict]
    sasrec: list[Recommendation]
    genrec: list[Recommendation]
    overlap: int


def load_everything() -> None:
    device = pick_device()
    with open(SASREC_CONFIG) as f:
        sasrec_cfg = yaml.safe_load(f)
    with open(GENREC_CONFIG) as f:
        genrec_cfg = yaml.safe_load(f)

    data_dir = Path(sasrec_cfg["data_dir"])
    train, valid, test, meta = load_processed(data_dir)
    vocab = SemanticIdVocab.from_data_dir(data_dir)
    with open(data_dir / "semantic_ids" / "item_texts.json") as f:
        titles = {int(k): v for k, v in json.load(f).items()}

    STATE.update(
        device=device,
        sasrec_cfg=sasrec_cfg,
        genrec_cfg=genrec_cfg,
        train=train,
        valid=valid,
        test=test,
        n_items=meta["n_items"],
        vocab=vocab,
        titles=titles,
        frequency=item_train_frequency(train, meta["n_items"]),
        sasrec=load_sasrec(
            sasrec_cfg,
            meta["n_items"],
            device,
            Path("results/checkpoints") / f"{sasrec_cfg['mlflow']['run_name']}.pt",
        ),
        genrec=load_genrec(
            genrec_cfg,
            vocab,
            device,
            Path("results/checkpoints") / f"{genrec_cfg['mlflow']['run_name']}.pt",
        ),
    )


def _describe(item: int, score: float) -> Recommendation:
    frequency = int(STATE["frequency"][item])
    return Recommendation(
        item_id=item,
        title=STATE["titles"].get(item, "")[:120],
        semantic_id=[int(t) for t in STATE["vocab"].item_tokens[item]],
        train_frequency=frequency,
        popularity_bucket=bucket_of(frequency),
        score=round(float(score), 4),
    )


def recommend_sasrec(history: list[int], k: int) -> list[Recommendation]:
    score_fn = make_score_fn(STATE["sasrec"], STATE["device"], mode="full")
    inputs = build_eval_input(history, STATE["sasrec_cfg"]["model"]["maxlen"])[None, :]
    scores = score_fn(inputs)[0].copy()
    scores[0] = -np.inf
    for item in set(history):
        scores[item] = -np.inf
    top = np.argsort(-scores)[:k]
    return [_describe(int(i), scores[i]) for i in top]


def recommend_genrec(history: list[int], k: int) -> list[Recommendation]:
    tokens = build_eval_input_tokens(
        history, STATE["vocab"], STATE["genrec_cfg"]["model"]["maxlen"]
    )[None, :]
    items, scores = batched_beam_search(
        STATE["genrec"],
        STATE["vocab"],
        tokens,
        STATE["device"],
        beam_size=max(k + len(set(history)), 20),
        n_return=max(k + len(set(history)), 20),
    )
    seen = set(history)
    out = []
    for item, score in zip(items[0], scores[0]):
        if item == 0 or item in seen or not np.isfinite(score):
            continue
        out.append(_describe(int(item), score))
        if len(out) == k:
            break
    return out


@app.get("/recommend", response_model=Response)
def recommend(
    items: str = Query(..., description="comma-separated internal item ids, oldest first"),
    k: int = Query(10, ge=1, le=50),
) -> Response:
    try:
        history = [int(x) for x in items.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "items must be comma-separated integers")
    if not history:
        raise HTTPException(400, "history is empty")
    bad = [i for i in history if not 1 <= i <= STATE["n_items"]]
    if bad:
        raise HTTPException(400, f"item ids out of range 1..{STATE['n_items']}: {bad}")

    sasrec = recommend_sasrec(history, k)
    genrec = recommend_genrec(history, k)
    return Response(
        history=[
            {
                "item_id": i,
                "title": STATE["titles"].get(i, "")[:120],
                "semantic_id": [int(t) for t in STATE["vocab"].item_tokens[i]],
            }
            for i in history
        ],
        sasrec=sasrec,
        genrec=genrec,
        overlap=len({r.item_id for r in sasrec} & {r.item_id for r in genrec}),
    )


@app.get("/random_user")
def random_user(seed: int | None = None) -> dict:
    """A real test user's history, for poking at the demo without inventing ids."""
    rng = np.random.default_rng(seed)
    users = sorted(STATE["train"])
    user = int(users[rng.integers(len(users))])
    history = STATE["train"][user][-10:]
    return {
        "user": user,
        "history": ",".join(str(i) for i in history),
        "held_out_test_item": STATE["test"].get(user),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<title>From SASRec to TIGER — demo</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}
 input{font:inherit;padding:.4rem;width:26rem} button{font:inherit;padding:.4rem .9rem}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:1.5rem}
 li{margin:.35rem 0} code{background:#f4f4f5;padding:.1rem .3rem;border-radius:3px}
 .b{font-size:.8em;color:#666} h2{font-size:1rem;margin-bottom:.4rem}
</style>
<h1>Amazon Beauty: atomic vs. semantic IDs</h1>
<p>Enter a history of item ids (oldest first), or <button onclick="pick()">use a random real user</button></p>
<input id="items" value="1,2,3"> <button onclick="go()">Recommend</button>
<div id="out"></div>
<script>
async function pick(){
  const r = await (await fetch('/random_user')).json();
  document.getElementById('items').value = r.history; go();
}
function list(rs){
  return '<ul>' + rs.map(r =>
    `<li>${r.title || '(item ' + r.item_id + ')'}<br>
     <span class="b">id ${r.item_id} · semantic <code>[${r.semantic_id}]</code>
     · seen ${r.train_frequency}x (${r.popularity_bucket}) · score ${r.score}</span></li>`).join('') + '</ul>';
}
async function go(){
  const v = document.getElementById('items').value;
  const r = await (await fetch('/recommend?items=' + encodeURIComponent(v))).json();
  if(r.detail){ document.getElementById('out').innerHTML = '<p style="color:#b00">'+r.detail+'</p>'; return; }
  document.getElementById('out').innerHTML =
    '<h2>History</h2>' + list(r.history.map(h => ({...h, train_frequency:'-', popularity_bucket:'-', score:'-'}))) +
    `<p><b>${r.overlap}</b> of 10 recommendations are shared between the two models.</p>` +
    '<div class="cols"><div><h2>SASRec (atomic IDs)</h2>' + list(r.sasrec) +
    '</div><div><h2>GenRec (semantic IDs)</h2>' + list(r.genrec) + '</div></div>';
}
go();
</script>
"""
