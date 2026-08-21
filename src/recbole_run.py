"""Train SASRec or BERT4Rec via RecBole for the training-budget comparison
and the SASRec cross-validation check. Logs to the same MLflow experiment as our
own training runs (tagged framework=recbole) so src/export_results.py can pull
everything into one table.
"""

import argparse
import os
import shutil
import time


# Cap CPU thread pools to the cores actually available to this process before
# any numeric library is imported. In a cgroup-limited container (e.g. a 4-CPU
# Daytona sandbox) OpenMP/MKL/BLAS otherwise read the HOST's core count and
# each spawn that many threads: a real run wedged with ~228 threads fighting
# over its cores in a futex livelock -- all cores pegged at 100%, GPU idle, not
# a single epoch completed. Getting the right number matters and is subtle:
#   - os.cpu_count() and os.sched_getaffinity() both report the HOST count on a
#     Daytona GPU sandbox (its cpu=N is a CFS *quota*, not an affinity mask), so
#     neither reflects the real limit -- an earlier fix using sched_getaffinity
#     read 96 and set 96 threads, reproducing the very oversubscription it was
#     meant to stop.
# So: honor an explicit override from the launcher first (daytona_week4.py knows
# the quota it requested and exports it), then read the cgroup CFS quota, then
# fall back to affinity/cpu_count for a normal local machine.
def _effective_cpu_limit() -> int:
    override = os.environ.get("RECBOLE_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if override and override.isdigit() and int(override) > 0:
        return int(override)
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux (e.g. local macOS)
        return os.cpu_count() or 1


_AVAIL_CPUS = _effective_cpu_limit()
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_var] = str(_AVAIL_CPUS)

import numpy as np  # noqa: E402

# RecBole 1.2.1's Config.compatibility_settings() reads numpy's deprecated
# underscore-suffixed aliases (np.float_, np.complex_, np.object_, np.str_,
# np.unicode_) which numpy 2.x removed entirely. Patch them back before
# importing anything that touches recbole.config.
_NUMPY2_SHIMS = {
    "float_": "float64",
    "complex_": "complex128",
    "object_": "object_",  # still exists, but the value points into a builtin
    "str_": "str_",
    "unicode_": "str_",
}
for _name, _replacement in _NUMPY2_SHIMS.items():
    if not hasattr(np, _name):
        setattr(np, _name, getattr(np, _replacement))

import torch  # noqa: E402

# PyTorch 2.6 flipped torch.load's default to weights_only=True, which cannot
# unpickle RecBole's checkpoints (they store the full config/optimizer state, not
# just tensors). RecBole's trainer.evaluate() calls torch.load internally with the
# default and dies with "Weights only load failed ... Unsupported operand". This
# is why every run that actually reached post-training evaluation crashed (our
# earlier runs were all stopped mid-training, so we never hit it until now). We
# only ever load checkpoints this same process just wrote, so restoring the old
# weights_only=False behavior is safe. Must be patched before RecBole loads any
# checkpoint (trainer.evaluate / resume).
_orig_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_full

from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import get_model, get_trainer, init_logger, init_seed  # noqa: E402

from src.utils import log_run  # noqa: E402

# Belt-and-suspenders alongside the env vars above: cap torch's own intra-op
# thread pool too (torch reads this at runtime, not just from the env).
torch.set_num_threads(_AVAIL_CPUS)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def export_scores(trainer, test_data, dataset, ckpt_path: str, out_path: str) -> None:
    """Dump the full item-score matrix for the test set, plus the token maps needed
    to align it with this repo's own data.

    Why: RecBole's evaluator draws its own 1+100 uniform negatives (`uni100`) instead
    of consuming `data/processed/ml-1m/negatives.json`, and it was run uni100-only, so
    the RecBole rows in the master table carry a negative-draw caveat and have no
    full-ranking numbers at all. Both are fixable *off* the GPU: with raw scores in
    hand, `src/eval/{sampled,full_ranking}.py` can rescore these predictions on this
    repo's fixed negatives and against the full catalog, on a laptop, for free.

    Exporting the dumb thing (a score matrix) rather than computing metrics here is
    deliberate: the id-mapping between RecBole's internal indices and this repo's is
    the only fiddly part, and doing it locally means it can be iterated on without
    burning GPU hours or risking a mid-run crash.

    Called inside a try/except by the caller -- a failure here must never cost a
    completed training run, since every metric is already logged by this point.

    Stored as float16: the full matrix is ~6040 users x ~3417 items, which is ~83MB
    in float32 and lands in git via the sandbox's push (GitHub warns over 50MB and
    hard-rejects over 100MB -- and a rejected push is the exact failure that
    destroyed a run on 2026-08-08). float16 halves it to a safe ~40MB at the cost of
    ~1e-3 relative precision, which can only change a rank when two items' logits
    differ by less than that. Worth recording as a caveat on any metric computed
    from this file, though it is orders of magnitude below the 1-6% margins under
    discussion.
    """
    import numpy as _np

    model = trainer.model
    model.eval()
    # field2id_token maps RecBole's internal contiguous index -> original dataset
    # token (the raw ML-1M user/item id as a string). Index 0 is RecBole's [PAD].
    item_tokens = dataset.field2id_token[dataset.iid_field]
    user_tokens = dataset.field2id_token[dataset.uid_field]

    rows, users = [], []
    uid_field = dataset.uid_field
    # Under eval mode uni100 the test loader is a NegSampleEvalDataLoader: it emits
    # 101 rows per user (the positive plus its 100 sampled negatives), all sharing
    # the SAME input sequence. full_sort_predict scores the whole catalog from that
    # sequence, so those 101 rows come back bit-identical -- verified on a smoke run.
    # Scoring them all would waste 101x the compute and, more importantly, write a
    # 101x-redundant matrix (a measured 68MB even after compression, uncomfortably
    # close to GitHub's 100MB hard limit on the sandbox's push). Keep the first row
    # per user.
    seen: set[int] = set()
    with torch.no_grad():
        for batched_data in test_data:
            interaction = batched_data[0].to(model.device)
            uids = interaction[uid_field].cpu().numpy()
            keep = _np.zeros(len(uids), dtype=bool)
            for i, uid in enumerate(uids):
                if uid not in seen:
                    seen.add(uid)
                    keep[i] = True
            if not keep.any():
                continue
            # full_sort_predict returns scores over every item, which is exactly what
            # a full-ranking evaluator needs; uni100 candidate lists are irrelevant here.
            scores = model.full_sort_predict(interaction).view(len(interaction), -1)
            rows.append(scores.cpu().numpy()[keep].astype(_np.float16))
            users.append(uids[keep])

    _np.savez_compressed(
        out_path,
        scores=_np.concatenate(rows, axis=0),
        user_index=_np.concatenate(users, axis=0),
        item_tokens=_np.asarray(item_tokens, dtype=object).astype(str),
        user_tokens=_np.asarray(user_tokens, dtype=object).astype(str),
        checkpoint=_np.asarray(ckpt_path),
    )
    print(f"  [export] wrote test score matrix to {out_path}", flush=True)


def run(
    model_name: str,
    budgets: list[tuple[int, str]],
    config_paths: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Train ``model_name`` ONCE to the largest budget and record a result for
    every budget from that single trajectory.

    ``budgets`` is a list of ``(epochs, run_name)`` pairs, e.g.
    ``[(200, "sasrec_recbole_1x"), (800, "..._4x"), (2000, "..._10x")]``. Because
    early stopping is disabled and validation runs every ``eval_step`` epochs, the
    best-valid model within the first N epochs is exactly what a standalone
    N-epoch run would have produced -- so one 2000-epoch run yields the 200/800/
    2000 budget points too, at ~1/3 the epochs of three separate runs and with no
    loss of fidelity (identical seed and data, same trajectory).

    Implementation note: RecBole writes the best-valid checkpoint to
    ``trainer.saved_model_file`` whenever validation improves and then calls our
    callback. At each budget milestone we only COPY that file aside; we never call
    ``evaluate()`` inside the callback, since ``evaluate(load_best_model=True)``
    reloads weights into the live model and would corrupt training for the larger
    budgets. Test evaluation for every milestone happens after training finishes,
    each from its own frozen checkpoint.
    """
    budgets = sorted(budgets, key=lambda b: b[0])
    milestone_names = {epochs: name for epochs, name in budgets}
    max_epochs = budgets[-1][0]

    # Multiple config files layer left-to-right (RecBole applies them in order), so
    # a variant is an overlay on the shared base rather than a forked copy of it --
    # e.g. ml1m_base.yaml + ml1m_sasrec_dropout02.yaml. Keeps the protocol settings
    # (split, negatives, budget, metrics) defined in exactly one place.
    config_paths = config_paths or ["configs/recbole/ml1m_base.yaml"]
    config = Config(
        model=model_name,
        dataset="ml-1m",
        config_file_list=list(config_paths),
        config_dict={"epochs": max_epochs, "seed": seed},
    )

    device = pick_device()
    config["device"] = device

    # Set up RecBole's logger so per-epoch training/validation lines are
    # actually emitted. Without this the low-level API path leaves the logger
    # handler-less, INFO logs get dropped, and a working run prints NOTHING for
    # its whole duration -- indistinguishable from a hang when watching a log.
    init_logger(config)
    print(
        f"model={model_name} budgets={[e for e, _ in budgets]} max_epochs={max_epochs} "
        f"eval_step={config['eval_step']} worker={config['worker']} "
        f"train_batch={config['train_batch_size']} device={device} threads={_AVAIL_CPUS}",
        flush=True,
    )

    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data.dataset).to(device)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # budget_epochs -> (frozen_checkpoint_path, best_valid_result_at_that_budget)
    snapshots: dict[int, tuple[str, dict]] = {}

    def snapshot_milestone(epoch_idx: int, valid_score: float) -> None:
        completed = epoch_idx + 1  # epoch_idx is 0-based
        if completed in milestone_names and trainer.best_valid_result is not None:
            frozen = os.path.join(config["checkpoint_dir"], f"{model_name}-budget-{completed}.pth")
            shutil.copyfile(trainer.saved_model_file, frozen)
            snapshots[completed] = (frozen, dict(trainer.best_valid_result))
            print(
                f"  [milestone] budget={completed}: froze best-valid checkpoint "
                f"(valid ndcg@10={trainer.best_valid_result['ndcg@10']:.4f})",
                flush=True,
            )

    start = time.time()
    trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
        callback_fn=snapshot_milestone,
    )
    total_train_time = time.time() - start

    results = []
    for epochs_budget, run_name in budgets:
        if epochs_budget not in snapshots:
            # Only happens if eval_step does not divide this budget (no validation
            # landed exactly on it). Surface loudly rather than silently drop a run.
            print(
                f"  [warn] no checkpoint captured at budget={epochs_budget}: eval_step "
                f"({config['eval_step']}) must divide every budget. Skipping this run.",
                flush=True,
            )
            continue
        ckpt_path, best_valid_result = snapshots[epochs_budget]
        test_result = trainer.evaluate(
            test_data,
            load_best_model=True,
            model_file=ckpt_path,
            show_progress=config["show_progress"],
        )
        print(f"budget={epochs_budget} valid={best_valid_result} test={test_result}", flush=True)

        metrics = {
            "valid_NDCG_at_10": float(best_valid_result["ndcg@10"]),
            "valid_HR_at_10": float(best_valid_result["hit@10"]),
            "test_NDCG_at_10": float(test_result["ndcg@10"]),
            "test_HR_at_10": float(test_result["hit@10"]),
            # Estimated wall time to reach this budget: total time scaled by the
            # budget's share of the single max-budget run (epochs are ~uniform).
            "train_time_sec": total_train_time * epochs_budget / max_epochs,
            "epochs_budget": float(epochs_budget),
        }
        log_run(
            experiment="sequential-rec",
            run_name=run_name,
            params={
                "model": model_name,
                "dataset": "ml-1m",
                "epochs": epochs_budget,
                "framework": "recbole",
                "device": str(device),
                # Record that all budgets came from one shared trajectory.
                "trained_to_epochs": max_epochs,
                # RecBole's per-model DEFAULTS are asymmetric --
                # SASRec ships dropout 0.5, BERT4Rec 0.2, with every other
                # architectural default identical -- and that this is the likely
                # driver of the flipped BERT4Rec-vs-SASRec headline. Runs that
                # differ only by an unlogged default are indistinguishable in the
                # master table, which is precisely how the asymmetry went unnoticed
                # in the first place. Log the knobs that vary.
                "hidden_dropout_prob": config["hidden_dropout_prob"],
                "attn_dropout_prob": config["attn_dropout_prob"],
                "hidden_size": config["hidden_size"],
                "n_heads": config["n_heads"],
                "n_layers": config["n_layers"],
                "loss_type": config["loss_type"],
                "train_batch_size": config["train_batch_size"],
                "configs": "+".join(os.path.basename(p) for p in config_paths),
            },
            metrics=metrics,
        )
        results.append(metrics)

        # Raw test scores for offline rescoring through this repo's own evaluator
        # (fixed negatives + full ranking -- see export_scores). Strictly a bonus
        # artifact: metrics for this budget are already logged above, so any
        # failure here is reported and swallowed rather than allowed to take down
        # a multi-hour run at its very last step.
        try:
            os.makedirs("results/scores", exist_ok=True)
            export_scores(trainer, test_data, dataset, ckpt_path, f"results/scores/{run_name}.npz")
        except Exception as exc:  # noqa: BLE001 - never fail a completed run over an extra
            print(f"  [warn] score export failed for {run_name}: {exc!r}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["SASRec", "BERT4Rec"])
    # New milestone path: one run yields every budget. Format: "epochs:run_name"
    # pairs, comma-separated, e.g. "200:sasrec_recbole_1x,800:...,2000:...".
    parser.add_argument("--budgets", type=str, default=None)
    # Back-compat single-budget path (used by scripts/daytona_remote_runner.sh).
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    # Comma-separated, applied left-to-right so later files override earlier ones:
    # "configs/recbole/ml1m_base.yaml,configs/recbole/ml1m_sasrec_dropout02.yaml".
    parser.add_argument("--config", type=str, default="configs/recbole/ml1m_base.yaml")
    args = parser.parse_args()

    if args.budgets:
        budgets = []
        for pair in args.budgets.split(","):
            epochs_str, name = pair.split(":")
            budgets.append((int(epochs_str), name))
    elif args.epochs is not None and args.run_name is not None:
        budgets = [(args.epochs, args.run_name)]
    else:
        parser.error("provide either --budgets or both --epochs and --run-name")
    run(args.model, budgets, config_paths=[p for p in args.config.split(",") if p])
