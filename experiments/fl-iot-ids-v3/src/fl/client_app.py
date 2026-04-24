from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import pickle
from time import perf_counter

import flwr as fl
import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context
try:
    from flwr.common.constant import PARTITION_ID_KEY
except ImportError:  # Flower <= 1.18 does not export the constant.
    PARTITION_ID_KEY = "partition-id"

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR, DATA_DIR, get_processed_path
from src.common.utils import get_expected_node_ids, resolve_node_id_from_partition
from src.data.dataloader import create_dataloaders_for_node
from src.model.evaluate import evaluate_model
from src.model.network import MLPClassifier
from src.model.train import train_one_epoch
from src.model.losses import build_loss, load_class_weights
from src.model.validation import resolve_num_classes, validate_model_output_dim


logger = get_logger("fl_client")

WEIGHTED_IMBALANCE_STRATEGIES = {"class_weights", "focal_loss_weighted"}


def get_model_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def resolve_node_id(context: Context, node_ids: list[str]) -> str:
    """
    Map Flower simulation partition IDs deterministically to node1/node2/node3.

    Important:
    `context.node_id` is a Flower runtime node identifier, not a stable dataset
    partition index. Using modulo on that value can duplicate one node and skip
    another. We must use `PARTITION_ID_KEY` from `context.node_config`.
    """
    raw_partition_id = None

    if hasattr(context, "node_config") and context.node_config is not None:
        raw_partition_id = context.node_config.get(PARTITION_ID_KEY)

    if raw_partition_id is None:
        raise ValueError(
            "Flower partition id is missing from context.node_config. "
            "Refusing to fall back to context.node_id because it can duplicate clients."
        )

    partition_id = int(raw_partition_id)
    node_id = resolve_node_id_from_partition(partition_id, node_ids)

    logger.info(
        "Resolved Flower partition_id=%s to dataset client=%s",
        partition_id,
        node_id,
    )
    return node_id


def resolve_node_dir(node_id: str, scenario: str):
    scenario_npz = get_processed_path(scenario, node_id)
    if scenario_npz.exists():
        return scenario_npz.parent

    legacy_dir = DATA_DIR / "processed" / node_id
    if legacy_dir.exists():
        logger.warning(
            "Falling back to legacy processed path for %s: %s",
            node_id,
            legacy_dir,
        )
        return legacy_dir

    raise FileNotFoundError(
        f"Processed data not found for scenario='{scenario}', node='{node_id}'. "
        f"Tried '{scenario_npz}' and '{legacy_dir}'."
    )


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        node_id: str,
        scenario: str,
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
        imbalance_strategy: str = "class_weights",
        focal_gamma: float = 2.0,
        fl_strategy: str = "fedavg",
        proximal_mu: float = 0.0,
        num_classes: int = 34,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
        benign_class_id: int = 1,
        rare_class_ids: tuple[int, ...] = (0, 3, 30, 31, 33),
    ):
        self.node_id = node_id
        self.scenario = scenario
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = 1e-4
        self.imbalance_strategy = imbalance_strategy
        self.focal_gamma = focal_gamma
        self.fl_strategy = fl_strategy.lower()
        self.proximal_mu = proximal_mu
        self.num_classes = int(num_classes)
        self.benign_class_id = benign_class_id
        self.rare_class_ids = rare_class_ids

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        node_dir = resolve_node_dir(node_id=node_id, scenario=scenario)
        self.train_loader, self.eval_loader = create_dataloaders_for_node(
            node_dir=node_dir,
            batch_size=batch_size,
            num_workers=0,
        )

        sample_batch = next(iter(self.train_loader))
        X_sample, _ = sample_batch
        input_dim = X_sample.shape[1]

        base_dataset = getattr(self.train_loader.dataset, "dataset", self.train_loader.dataset)
        max_label = int(base_dataset.y.max().item())
        if max_label >= self.num_classes:
            raise ValueError(
                f"[{self.node_id}] Local labels contain class id {max_label}, "
                f"but dataset.num_classes={self.num_classes}."
            )

        self.model = MLPClassifier(
            input_dim=input_dim,
            num_classes=self.num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)
        validate_model_output_dim(self.model, self.num_classes)

        class_weights = None
        if self.imbalance_strategy in WEIGHTED_IMBALANCE_STRATEGIES:
            class_weights_path = ARTIFACTS_DIR / f"class_weights_{self.scenario}.pkl"
            class_weights = load_class_weights(class_weights_path, device=self.device)
            if class_weights is None:
                raise FileNotFoundError(
                    f"Missing scenario-specific class weights: {class_weights_path}. "
                    f"Run: python -m src.scripts.generate_weights --scenario {self.scenario}"
                )
            if int(class_weights.numel()) != self.num_classes:
                raise ValueError(
                    f"Class weights at {class_weights_path} have {class_weights.numel()} "
                    f"entries but dataset.num_classes={self.num_classes}."
                )
        self.criterion = build_loss(
            class_weights=class_weights,
            imbalance_strategy=self.imbalance_strategy,
            focal_gamma=self.focal_gamma,
        )

        self._reset_optimizer()

        logger.info(
            "[%s] ready | device=%s | samples=%s | batches=%s | num_classes=%s",
            self.node_id,
            self.device,
            len(self.train_loader.dataset),
            len(self.train_loader),
            self.num_classes,
        )

    def _train_fedprox_epoch(self, global_params):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            ce_loss = self.criterion(logits, y)

            prox_term = sum(
                torch.sum((param - global_param) ** 2)
                for param, global_param in zip(self.model.parameters(), global_params)
            )
            loss = ce_loss + (self.proximal_mu / 2.0) * prox_term
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return {
            "loss": running_loss / max(len(self.train_loader), 1),
            "accuracy": correct / total if total > 0 else 0.0,
            "num_samples": total,
        }

    def _get_c_local(self) -> list[np.ndarray]:
        if not hasattr(self, "c_local") or self.c_local is None:
            self.c_local = [np.zeros_like(param) for param in get_model_parameters(self.model)]
        return self.c_local

    def _train_scaffold_epoch(
        self,
        c_local: list[np.ndarray],
        c_global: list[np.ndarray],
    ):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        c_diff = [
            torch.tensor(cg - cl, dtype=torch.float32, device=self.device)
            for cl, cg in zip(c_local, c_global)
        ]

        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            for param, diff in zip(self.model.parameters(), c_diff):
                if param.grad is not None:
                    param.grad.data.add_(diff)
            self.optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return {
            "loss": running_loss / max(len(self.train_loader), 1),
            "accuracy": correct / total if total > 0 else 0.0,
            "num_samples": total,
        }

    def _reset_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def get_parameters(self, config):
        logger.info("[%s] get_parameters()", self.node_id)
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        logger.info("[%s] fit() started", self.node_id)
        set_model_parameters(self.model, parameters)
        self._reset_optimizer()

        bytes_before = sum(param.nbytes for param in parameters)
        start = perf_counter()
        last = None

        if self.fl_strategy == "fedprox" and self.proximal_mu > 0.0:
            global_params = [
                param.detach().clone().to(self.device)
                for param in self.model.parameters()
            ]
            for _ in range(self.local_epochs):
                last = self._train_fedprox_epoch(global_params)
        elif self.fl_strategy == "scaffold":
            raw_c_global = config.get("scaffold_c_global")
            if raw_c_global is None:
                raise ValueError(
                    "SCAFFOLD requires 'scaffold_c_global' in fit config. "
                    "Use the v3 ServerApp scaffold strategy."
                )
            c_global = pickle.loads(raw_c_global)
            c_local = self._get_c_local()
            if len(c_global) != len(c_local):
                raise ValueError("SCAFFOLD control variate length mismatch.")

            w_before = [np.array(param, copy=True) for param in parameters]
            for _ in range(self.local_epochs):
                last = self._train_scaffold_epoch(c_local, c_global)

            w_after = get_model_parameters(self.model)
            steps = max(self.local_epochs * len(self.train_loader), 1)
            delta_c = [
                (wb - wa) / (self.learning_rate * steps) - cg
                for wb, wa, cg in zip(w_before, w_after, c_global)
            ]
            self.c_local = [cl + dc for cl, dc in zip(c_local, delta_c)]
        else:
            for _ in range(self.local_epochs):
                last = train_one_epoch(
                    model=self.model,
                    loader=self.train_loader,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    device=self.device,
                )

        updated_parameters = get_model_parameters(self.model)
        train_time_sec = perf_counter() - start
        update_size_bytes = sum(param.nbytes for param in updated_parameters)

        logger.info(
            "[%s] fit() done | loss=%.4f | acc=%.4f",
            self.node_id,
            float(last["loss"]),
            float(last["accuracy"]),
        )

        metrics = {
            "node_id": self.node_id,
            "train_loss_last": float(last["loss"]),
            "train_accuracy_last": float(last["accuracy"]),
            "train_time_sec": float(train_time_sec),
            "update_size_bytes": int(update_size_bytes),
            "bytes_received": float(bytes_before),
        }
        if self.fl_strategy == "scaffold":
            metrics["scaffold_delta_c"] = pickle.dumps(
                [dc.astype(np.float32) for dc in delta_c]
            )

        return (
            updated_parameters,
            len(self.train_loader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        logger.info("[%s] evaluate() started", self.node_id)
        set_model_parameters(self.model, parameters)

        metrics = evaluate_model(
            model=self.model,
            loader=self.eval_loader,
            criterion=self.criterion,
            device=self.device,
            benign_class_id=self.benign_class_id,
            rare_class_ids=self.rare_class_ids,
            num_classes=self.num_classes,
        )

        logger.info(
            "[%s] evaluate() done | loss=%.4f | acc=%.4f | macro_f1=%.4f",
            self.node_id,
            float(metrics["loss"]),
            float(metrics["accuracy"]),
            float(metrics["macro_f1"]),
        )

        metric_payload = {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "recall_macro": float(metrics["recall_macro"]),
            "benign_recall": float(metrics["benign_recall"]),
            "false_positive_rate": float(metrics["false_positive_rate"]),
            "rare_class_recall": float(metrics["rare_class_recall"]),
        }
        metric_payload.update(
            {
                key: float(value)
                for key, value in metrics.items()
                if key.startswith(("tp_class_", "fp_class_", "fn_class_"))
            }
        )

        return (
            float(metrics["loss"]),
            len(self.eval_loader.dataset),
            metric_payload,
        )


def make_client_fn(config: Mapping[str, object]):
    scenario_cfg = dict(config.get("scenario", {}))
    train_cfg = dict(config.get("train", {}))
    model_cfg = dict(config.get("model", {}))
    dataset_cfg = dict(config.get("dataset", {}))
    imbalance_cfg = dict(config.get("imbalance", {}))
    experiment_cfg = dict(config.get("experiment", {}))

    scenario = str(scenario_cfg.get("name", "normal_noniid"))
    batch_size = int(train_cfg.get("batch_size", 256))
    local_epochs = int(train_cfg.get("local_epochs", 1))
    learning_rate = float(train_cfg.get("learning_rate", 0.001))
    fl_strategy = str(
        experiment_cfg.get("fl_strategy", config.get("strategy", {}).get("name", "fedavg"))
    )
    proximal_mu = float(train_cfg.get("proximal_mu", 0.0))
    imbalance_strategy = str(imbalance_cfg.get("name", "class_weights"))
    focal_gamma = float(imbalance_cfg.get("focal_gamma", 2.0))
    hidden_dims = tuple(model_cfg.get("hidden_dims", [128, 64]))
    dropout = float(model_cfg.get("dropout", 0.2))
    num_classes = resolve_num_classes(dataset_cfg, model_cfg)
    benign_class_id = int(dataset_cfg.get("benign_class_id", 1))
    rare_class_ids = tuple(dataset_cfg.get("rare_class_ids", [0, 3, 30, 31, 33]))
    num_clients = int(scenario_cfg.get("num_clients", 3))
    node_ids = list(scenario_cfg.get("node_ids", get_expected_node_ids(num_clients)))

    assert len(node_ids) == num_clients, (
        f"node_ids length mismatch: node_ids={node_ids}, num_clients={num_clients}"
    )
    assert len(set(node_ids)) == len(node_ids), (
        f"Duplicate node_ids configured: {node_ids}"
    )

    logger.info("Selected clients: %s", ", ".join(node_ids))

    def configured_client_fn(context: Context):
        node_id = resolve_node_id(context, node_ids)
        logger.info(
            "ClientApp started | resolved node_id=%s | scenario=%s | available_clients=%s",
            node_id,
            scenario,
            ", ".join(node_ids),
        )
        return FlowerClient(
            node_id=node_id,
            scenario=scenario,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            imbalance_strategy=imbalance_strategy,
            focal_gamma=focal_gamma,
            fl_strategy=fl_strategy,
            proximal_mu=proximal_mu,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            benign_class_id=benign_class_id,
            rare_class_ids=rare_class_ids,
        ).to_client()

    return configured_client_fn


def create_client_app(config: Mapping[str, object]) -> ClientApp:
    return ClientApp(client_fn=make_client_fn(config))


def client_fn(context: Context):
    run_config = context.run_config
    config = {
        "experiment": {"fl_strategy": str(run_config.get("strategy", "fedavg"))},
        "scenario": {"name": str(run_config.get("scenario", "normal_noniid"))},
        "train": {
            "batch_size": int(run_config.get("batch-size", 256)),
            "local_epochs": int(run_config.get("local-epochs", 1)),
            "learning_rate": float(run_config.get("learning-rate", 0.001)),
            "proximal_mu": float(run_config.get("proximal-mu", 0.0)),
        },
        "imbalance": {
            "name": str(run_config.get("imbalance-strategy", "class_weights")),
            "focal_gamma": float(run_config.get("focal-gamma", 2.0)),
        },
        "model": {
            "hidden_dims": [128, 64],
            "dropout": 0.2,
        },
        "dataset": {
            "num_classes": int(run_config.get("num-classes", 34)),
            "benign_class_id": 1,
            "rare_class_ids": [0, 3, 30, 31, 33],
        },
    }
    return make_client_fn(config)(context)


app = ClientApp(client_fn=client_fn)
