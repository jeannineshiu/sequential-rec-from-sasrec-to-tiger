"""Train SASRec or BERT4Rec via RecBole for the Week 4 training-budget comparison
and the SASRec cross-validation check. Logs to the same MLflow experiment as our
own training runs (tagged framework=recbole) so src/export_results.py can pull
everything into one table.
"""

import argparse
import time

import numpy as np

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
from recbole.utils import get_model, get_trainer, init_seed  # noqa: E402

from src.utils import log_run  # noqa: E402


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
    print(f"model={model_name} epochs={epochs} device={device}")

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
