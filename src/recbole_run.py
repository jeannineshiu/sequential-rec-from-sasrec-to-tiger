"""Train SASRec or BERT4Rec via RecBole for the Week 4 training-budget comparison
and the SASRec cross-validation check. Logs to the same MLflow experiment as our
own training runs (tagged framework=recbole) so src/export_results.py can pull
everything into one table.
"""

import argparse
import os
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
    epochs: int,
    run_name: str,
    config_path: str = "configs/recbole/ml1m_base.yaml",
    seed: int = 42,
) -> dict:
    config = Config(
        model=model_name,
        dataset="ml-1m",
        config_file_list=[config_path],
        config_dict={"epochs": epochs, "seed": seed},
    )

    device = pick_device()
    config["device"] = device

    # Set up RecBole's logger so per-epoch training/validation lines are
    # actually emitted. Without this the low-level API path leaves the logger
    # handler-less, INFO logs get dropped, and a working run prints NOTHING for
    # its whole duration -- indistinguishable from a hang when watching a log.
    init_logger(config)
    print(f"model={model_name} epochs={epochs} device={device} threads={_AVAIL_CPUS}", flush=True)

    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data.dataset).to(device)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    start = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    train_time = time.time() - start

    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    print("best_valid_result:", best_valid_result)
    print("test_result:", test_result)

    metrics = {
        "valid_NDCG_at_10": float(best_valid_result["ndcg@10"]),
        "valid_HR_at_10": float(best_valid_result["hit@10"]),
        "test_NDCG_at_10": float(test_result["ndcg@10"]),
        "test_HR_at_10": float(test_result["hit@10"]),
        "train_time_sec": train_time,
        "epochs_budget": float(epochs),
    }
    log_run(
        experiment="sequential-rec",
        run_name=run_name,
        params={
            "model": model_name,
            "dataset": "ml-1m",
            "epochs": epochs,
            "framework": "recbole",
            "device": str(device),
        },
        metrics=metrics,
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["SASRec", "BERT4Rec"])
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/recbole/ml1m_base.yaml")
    args = parser.parse_args()
    run(args.model, args.epochs, args.run_name, config_path=args.config)
