"""How much of GenRec's full-ranking score is lost to the beam?

The generative model ranks by constrained beam search, so an item the beam never
reaches is a miss even if exhaustive scoring would have placed it top-10. That is
an approximation the dot-product baselines do not carry, and differencing the two
without bounding it would repeat Week 4's mistake in a new costume.

This widens the beam and reports where the metric stops moving. If HR@10 is flat
from beam 20 to beam 200, the beam is not what separates GenRec from SASRec.

    uv run python -m scripts.beam_sensitivity --config configs/genrec_beauty.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import yaml

from src.eval.generative import evaluate_generative_full_ranking, make_sampled_score_fn
from src.eval.sampled import evaluate_sampled, load_negatives
from src.models.genrec import GenRec
from src.semantic_ids.vocab import SemanticIdVocab
from src.train import pick_device
from src.utils import load_processed


def main(config_path: str, beams: list[int], checkpoint: str | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = pick_device()
    data_dir = Path(cfg["data_dir"])
    train, valid, test, _ = load_processed(data_dir)
    negatives = load_negatives(data_dir / "negatives.json")
    vocab = SemanticIdVocab.from_data_dir(data_dir)

    maxlen_items = cfg["model"]["maxlen"]
    model = GenRec(
        vocab_size=vocab.vocab_size,
        n_levels=vocab.n_levels,
        level_offsets=vocab.level_offsets,
        level_sizes=vocab.level_sizes,
        maxlen_items=maxlen_items,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_blocks=cfg["model"]["num_blocks"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        pos_emb_type=cfg["model"].get("pos_emb_type", "learnable"),
    ).to(device)

    ckpt = checkpoint or f"results/checkpoints/{cfg['mlflow']['run_name']}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"loaded {ckpt}")

    k = cfg["eval"]["k"]
    extra = {u: [valid[u]] for u in test}

    sampled = evaluate_sampled(
        make_sampled_score_fn(model, vocab, device),
        train,
        test,
        negatives,
        maxlen=maxlen_items,
        extra_history=extra,
        k=k,
    )
    print(f"\nsampled (beam-independent): HR@{k} {sampled[f'HR@{k}']:.4f} "
          f"NDCG@{k} {sampled[f'NDCG@{k}']:.4f}")

    print(f"\n| beam | full HR@{k} | full NDCG@{k} | sec |")
    print("|---|---|---|---|")
    for beam in beams:
        start = time.time()
        metrics = evaluate_generative_full_ranking(
            model,
            vocab,
            train,
            test,
            maxlen_items=maxlen_items,
            device=device,
            extra_history=extra,
            exclude_extra=extra,
            k=k,
            beam_size=beam,
        )
        print(
            f"| {beam} | {metrics[f'HR@{k}']:.4f} | {metrics[f'NDCG@{k}']:.4f} | "
            f"{time.time() - start:.0f} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/genrec_beauty.yaml")
    parser.add_argument("--beams", type=int, nargs="+", default=[10, 20, 50, 100, 200])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args.beams, args.checkpoint)
