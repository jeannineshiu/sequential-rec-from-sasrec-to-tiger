"""Train the semantic-ID generative recommender.

Mirrors `src/train.py` -- same config shape, same MLflow experiment, same early
stopping on sampled valid NDCG@10 -- so a GenRec run and a SASRec run differ in
the model, not in the harness around it.

Scoring 101 candidates generatively would mean 101 forward passes per user
where SASRec needs one dot product -- 10 minutes per validation on Beauty. The
history KV cache in `GenRec.build_cache` brings that to ~14 seconds, which is
why validation here runs on the full valid set every epoch, exactly as SASRec's
does, instead of on a subsample. `train.valid_subsample` still exists for slower
machines; final test metrics always use every user regardless.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.genrec_dataset import GenRecTrainDataset
from src.eval.generative import (
    decode_legality,
    evaluate_generative_full_ranking,
    make_sampled_score_fn,
)
from src.eval.sampled import evaluate_sampled, load_negatives
from src.models.genrec import GenRec
from src.semantic_ids.vocab import PAD, SemanticIdVocab
from src.train import pick_device
from src.utils import get_git_hash, load_processed

import mlflow


def subsample_targets(targets: dict[int, int], n: int | None, seed: int) -> dict[int, int]:
    if not n or n >= len(targets):
        return targets
    users = sorted(targets)
    picked = random.Random(seed).sample(users, n)
    return {u: targets[u] for u in picked}


def run(
    config_path: str,
    max_epochs_override: int | None = None,
    seed_override: int | None = None,
    run_name_override: str | None = None,
) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if seed_override is not None:
        cfg["train"]["seed"] = seed_override
    if run_name_override is not None:
        cfg["mlflow"]["run_name"] = run_name_override

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = pick_device()
    print(f"device: {device}")

    data_dir = Path(cfg["data_dir"])
    train, valid, test, meta = load_processed(data_dir)
    negatives = load_negatives(data_dir / "negatives.json")

    vocab = SemanticIdVocab.from_data_dir(data_dir)
    print(vocab.summary())

    maxlen_items = cfg["model"]["maxlen"]
    train_ds = GenRecTrainDataset(train, vocab, maxlen_items=maxlen_items)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False
    )

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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], betas=(0.9, cfg["train"]["adam_beta2"])
    )
    # ignore_index=PAD covers the padded tail; the level-masked logits make this
    # a softmax over one codebook rather than over the whole token vocabulary.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    max_epochs = max_epochs_override or cfg["train"]["max_epochs"]
    patience = cfg["train"]["early_stop_patience"]
    k = cfg["eval"]["k"]
    beam_size = cfg["eval"].get("beam_size", 20)
    valid_subsample = cfg["train"].get("valid_subsample")
    valid_users = subsample_targets(valid, valid_subsample, seed=seed)
    print(f"validating on {len(valid_users)}/{len(valid)} users each epoch")

    best_valid_ndcg = -1.0
    best_state = None
    epochs_without_improvement = 0
    epoch_times: list[float] = []

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
        mlflow.set_tag("git_hash", get_git_hash())
        mlflow.log_params(
            {
                "dataset": cfg["dataset"],
                "device": str(device),
                "model_type": "genrec_semantic_id",
                "n_params": n_params,
                "vocab_size": vocab.vocab_size,
                "n_levels": vocab.n_levels,
                **{f"model_{k_}": v for k_, v in cfg["model"].items()},
                **{f"train_{k_}": v for k_, v in cfg["train"].items()},
            }
        )

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_start = time.time()
            total_loss = 0.0
            n_batches = 0

            for input_tokens, target_tokens in train_loader:
                input_tokens = input_tokens.to(device)
                target_tokens = target_tokens.to(device)

                logits = model(input_tokens)
                # An all-padding context predicts from nothing; those positions
                # would train an unconditional prior, so they are dropped.
                targets = target_tokens.masked_fill(input_tokens == PAD, PAD)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            score_fn = make_sampled_score_fn(model, vocab, device)
            valid_metrics = evaluate_sampled(
                score_fn, train, valid_users, negatives, maxlen=maxlen_items, k=k
            )
            valid_ndcg = valid_metrics[f"NDCG@{k}"]

            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "epoch_time_sec": epoch_time,
                    f"valid_HR_at_{k}": valid_metrics[f"HR@{k}"],
                    f"valid_NDCG_at_{k}": valid_ndcg,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch:3d} | loss {avg_loss:.4f} | "
                f"valid HR@{k} {valid_metrics[f'HR@{k}']:.4f} NDCG@{k} {valid_ndcg:.4f} | "
                f"{epoch_time:.1f}s"
            )

            if valid_ndcg > best_valid_ndcg:
                best_valid_ndcg = valid_ndcg
                best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"early stopping at epoch {epoch} (patience {patience})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint_dir = Path("results") / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{cfg['mlflow']['run_name']}.pt"
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path))

        test_extra_history = {u: [valid[u]] for u in test}

        score_fn = make_sampled_score_fn(model, vocab, device)
        test_sampled = evaluate_sampled(
            score_fn,
            train,
            test,
            negatives,
            maxlen=maxlen_items,
            extra_history=test_extra_history,
            k=k,
        )

        test_full = evaluate_generative_full_ranking(
            model,
            vocab,
            train,
            test,
            maxlen_items=maxlen_items,
            device=device,
            extra_history=test_extra_history,
            exclude_extra=test_extra_history,
            k=k,
            beam_size=beam_size,
        )

        legality = decode_legality(
            model,
            vocab,
            train,
            test,
            maxlen_items=maxlen_items,
            device=device,
            extra_history=test_extra_history,
        )

        final_metrics = (
            {f"test_sampled_{k_}": v for k_, v in test_sampled.items()}
            | {f"test_full_{k_}": v for k_, v in test_full.items()}
            | legality
        )
        final_metrics["epochs_trained"] = float(len(epoch_times))
        final_metrics["avg_epoch_time_sec"] = sum(epoch_times) / len(epoch_times)
        safe_final = {k_.replace("@", "_at_"): v for k_, v in final_metrics.items()}
        mlflow.log_metrics(safe_final)

        print("=== Test metrics ===")
        for k_, v in final_metrics.items():
            print(f"  {k_}: {v:.4f}")

        return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/genrec_beauty.yaml")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="override train.seed")
    parser.add_argument("--run-name", type=str, default=None, help="override mlflow.run_name")
    args = parser.parse_args()

    run(
        args.config,
        max_epochs_override=args.max_epochs,
        seed_override=args.seed,
        run_name_override=args.run_name,
    )
