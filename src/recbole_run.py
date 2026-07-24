"""Train SASRec or BERT4Rec via RecBole for the Week 4 training-budget comparison
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


def run(
    model_name: str,
    budgets: list[tuple[int, str]],
    config_path: str = "configs/recbole/ml1m_base.yaml",
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

    config = Config(
        model=model_name,
        dataset="ml-1m",
        config_file_list=[config_path],
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
            },
            metrics=metrics,
        )
        results.append(metrics)
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
    run(args.model, budgets, config_path=args.config)
