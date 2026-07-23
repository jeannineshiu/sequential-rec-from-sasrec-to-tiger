"""Train SASRec on a config, with early stopping on sampled valid NDCG@10 and
MLflow logging of every epoch + final test metrics. See EXECUTION_PLAN.md Week 2.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SASRecTrainDataset
from src.eval.full_ranking import evaluate_full_ranking
from src.eval.sampled import evaluate_sampled, load_negatives
from src.models.sasrec import SASRec
from src.utils import get_git_hash, load_processed

import mlflow


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_score_fn(model: SASRec, device: torch.device, mode: str):
    """mode: 'sampled' -> score_fn(input_batch, candidate_batch); 'full' -> score_fn(input_batch)."""
    model.eval()

    if mode == "sampled":

        def score_fn(input_batch: np.ndarray, candidate_batch: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                inputs = torch.from_numpy(input_batch).long().to(device)
                candidates = torch.from_numpy(candidate_batch).long().to(device)
                scores = model.score(inputs, candidates)
            return scores.cpu().numpy()

        return score_fn

    def score_fn(input_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            inputs = torch.from_numpy(input_batch).long().to(device)
            scores = model.score_full_catalog(inputs)
        return scores.cpu().numpy()

    return score_fn


def run(config_path: str, max_epochs_override: int | None = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = pick_device()
    print(f"device: {device}")

    data_dir = Path(cfg["data_dir"])
    train, valid, test, meta = load_processed(data_dir)
    n_items = meta["n_items"]
    negatives = load_negatives(data_dir / "negatives.json")

    maxlen = cfg["model"]["maxlen"]
    neg_sampling = cfg["train"].get("neg_sampling", "uniform")
    train_ds = SASRecTrainDataset(
        train, n_items=n_items, maxlen=maxlen, seed=seed, neg_sampling=neg_sampling
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False
    )

    model = SASRec(
        n_items=n_items,
        maxlen=maxlen,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_blocks=cfg["model"]["num_blocks"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        pos_emb_type=cfg["model"].get("pos_emb_type", "learnable"),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], betas=(0.9, cfg["train"]["adam_beta2"])
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    max_epochs = max_epochs_override or cfg["train"]["max_epochs"]
    patience = cfg["train"]["early_stop_patience"]
    k = cfg["eval"]["k"]

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
                **{f"model_{k_}": v for k_, v in cfg["model"].items()},
                **{f"train_{k_}": v for k_, v in cfg["train"].items()},
            }
        )

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_start = time.time()
            total_loss = 0.0
            n_batches = 0

            for input_seqs, pos_seqs, neg_seqs in train_loader:
                input_seqs = input_seqs.to(device)
                pos_seqs = pos_seqs.to(device)
                neg_seqs = neg_seqs.to(device)

                pos_logits, neg_logits = model(input_seqs, pos_seqs, neg_seqs)
                valid_mask = (pos_seqs != 0).float()

                pos_loss = bce(pos_logits, torch.ones_like(pos_logits))
                neg_loss = bce(neg_logits, torch.zeros_like(neg_logits))
                loss = ((pos_loss + neg_loss) * valid_mask).sum() / valid_mask.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            valid_score_fn = make_score_fn(model, device, mode="sampled")
            valid_metrics = evaluate_sampled(
                valid_score_fn, train, valid, negatives, maxlen=maxlen, k=k
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
        sampled_score_fn = make_score_fn(model, device, mode="sampled")
        test_sampled = evaluate_sampled(
            sampled_score_fn,
            train,
            test,
            negatives,
            maxlen=maxlen,
            extra_history=test_extra_history,
            k=k,
        )

        full_score_fn = make_score_fn(model, device, mode="full")
        test_full = evaluate_full_ranking(
            full_score_fn,
            train,
            test,
            n_items=n_items,
            maxlen=maxlen,
            extra_history=test_extra_history,
            exclude_extra=test_extra_history,
            k=k,
        )

        final_metrics = {f"test_sampled_{k_}": v for k_, v in test_sampled.items()} | {
            f"test_full_{k_}": v for k_, v in test_full.items()
        }
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
    parser.add_argument("--config", type=str, default="configs/sasrec_ml1m.yaml")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    run(args.config, max_epochs_override=args.max_epochs)
