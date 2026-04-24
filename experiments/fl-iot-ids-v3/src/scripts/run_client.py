from __future__ import annotations

import argparse
import logging
import pickle
from collections import OrderedDict
from typing import Any, List

import flwr as fl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.common.config import load_yaml_config
from src.common.paths import ARTIFACTS_DIR, PROCESSED_DIR, get_processed_path
from src.data.dataloader import create_dataloaders_for_node
from src.model.network import MLPClassifier
from src.model.train import train_one_epoch
from src.model.losses import build_loss, load_class_weights
from src.model.validation import validate_model_output_dim

BENIGN_CLASS = 1  # BenignTraffic = label_id 1 in the v3 dataset
WEIGHTED_IMBALANCE_STRATEGIES = {"class_weights", "focal_loss_weighted"}

logger = logging.getLogger("run_client")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flower client for fl-iot-ids-v3")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--node-id", type=str, choices=["node1", "node2", "node3"], required=True)
    parser.add_argument("--scenario", type=str, default="normal_noniid")
    parser.add_argument("--server-address", type=str, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    # FedProx proximal term coefficient (0.0 = standard FedAvg)
    parser.add_argument("--mu", type=float, default=None,
                        help="FedProx proximal term coefficient (default: 0.0 = FedAvg).")
    # FL algorithm the client should run
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["fedavg", "fedprox", "scaffold"],
                        help="FL training strategy (default: fedavg).")
    parser.add_argument(
        "--imbalance-strategy",
        type=str,
        default=None,
        choices=["none", "class_weights", "focal_loss", "focal_loss_weighted"],
    )
    parser.add_argument("--focal-gamma", type=float, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------

class IoTFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        node_id: str,
        model: torch.nn.Module,
        train_loader: Any,
        eval_loader: Any,
        device: torch.device,
        local_epochs: int,
        learning_rate: float,
        class_weights: torch.Tensor | None = None,
        imbalance_strategy: str = "class_weights",
        focal_gamma: float = 2.0,
        mu: float = 0.0,
        strategy: str = "fedavg",
    ) -> None:
        self.node_id = node_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = 1e-4
        self.imbalance_strategy = imbalance_strategy
        self.focal_gamma = focal_gamma
        self.mu = mu
        self.strategy = strategy

        self.criterion = build_loss(
            class_weights=class_weights.to(device) if class_weights is not None else None,
            imbalance_strategy=imbalance_strategy,
            focal_gamma=focal_gamma,
        )
        self._reset_optimizer_and_scheduler()

    def _reset_optimizer_and_scheduler(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=3,
            factor=0.5,
        )

    # ------------------------------------------------------------------
    # SCAFFOLD control-variate helpers (disk-based, localhost setup)
    # ------------------------------------------------------------------

    def _c_path(self, name: str):
        return ARTIFACTS_DIR / f"scaffold_c_{name}.pkl"

    def _load_c(self, name: str) -> List[np.ndarray] | None:
        p = self._c_path(name)
        if p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
        return None

    def _save_c(self, name: str, c: List[np.ndarray]) -> None:
        p = self._c_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(c, f)

    def _get_c(self, name: str) -> List[np.ndarray]:
        """Return saved control variate, or zeros matching model shape on first call."""
        c = self._load_c(name)
        if c is None:
            c = [np.zeros_like(p) for p in get_parameters(self.model)]
        # Shape guard: reset to zeros if model was changed since last save
        model_shapes = [p.shape for p in get_parameters(self.model)]
        if any(ci.shape != s for ci, s in zip(c, model_shapes)):
            logger.warning("SCAFFOLD: shape mismatch in %s — reinitialising to zeros", name)
            c = [np.zeros(s, dtype=np.float32) for s in model_shapes]
        return c

    # ------------------------------------------------------------------
    # Custom training loops
    # ------------------------------------------------------------------

    def _train_fedprox_epoch(
        self, global_params: List[torch.Tensor]
    ) -> dict:
        """One epoch of FedProx: CE loss + (mu/2) * ||w - w_global||^2."""
        self.model.train()
        total_ce = 0.0
        correct = 0
        total = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(x)
            ce_loss = self.criterion(outputs, y)

            # Proximal term — penalises drift from global model
            prox = sum(
                torch.sum((p - gp) ** 2)
                for p, gp in zip(self.model.parameters(), global_params)
            )
            loss = ce_loss + (self.mu / 2.0) * prox
            loss.backward()
            self.optimizer.step()

            total_ce += ce_loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        n_batches = max(len(self.train_loader), 1)
        return {"loss": total_ce / n_batches, "accuracy": correct / max(total, 1)}

    def _train_scaffold_epoch(
        self,
        c_local: List[np.ndarray],
        c_global: List[np.ndarray],
    ) -> dict:
        """One epoch of SCAFFOLD: gradient += (c_global - c_local) per parameter."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Pre-compute correction tensors once per epoch (c_global - c_local)
        c_diff = [
            torch.tensor(cg - cl, dtype=torch.float32, device=self.device)
            for cl, cg in zip(c_local, c_global)
        ]

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()

            # SCAFFOLD correction: corrected_grad = grad + (c_global - c_local)
            for p, diff in zip(self.model.parameters(), c_diff):
                if p.grad is not None:
                    p.grad.data.add_(diff)

            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        n_batches = max(len(self.train_loader), 1)
        return {"loss": total_loss / n_batches, "accuracy": correct / max(total, 1)}

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        logging.getLogger("fl_client").info("[%s] get_parameters()", self.node_id)
        return get_parameters(self.model)

    def fit(self, parameters, config):
        import time

        flog = logging.getLogger("fl_client")
        flog.info("[%s] fit() | strategy=%s | mu=%.4f", self.node_id, self.strategy, self.mu)

        bytes_received = sum(p.nbytes for p in parameters)
        set_parameters(self.model, parameters)
        self._reset_optimizer_and_scheduler()

        last_loss = 0.0
        last_acc = 0.0
        fit_start = time.time()
        extra_metrics: dict = {}

        # ── FedAvg ──────────────────────────────────────────────────────
        if self.strategy == "fedavg":
            for _ in range(self.local_epochs):
                m = train_one_epoch(
                    model=self.model,
                    loader=self.train_loader,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    device=self.device,
                )
                last_loss = float(m["loss"])
                last_acc = float(m["accuracy"])

        # ── FedProx ─────────────────────────────────────────────────────
        elif self.strategy == "fedprox":
            # Snapshot global weights before any local update
            global_params = [p.clone().detach() for p in self.model.parameters()]
            for _ in range(self.local_epochs):
                m = self._train_fedprox_epoch(global_params)
                last_loss = float(m["loss"])
                last_acc = float(m["accuracy"])

        # ── SCAFFOLD ────────────────────────────────────────────────────
        elif self.strategy == "scaffold":
            c_local = self._get_c(f"local_{self.node_id}")
            c_global = self._get_c("global")

            # Save w_before (= global params received from server)
            w_before = [p.copy() for p in parameters]

            for _ in range(self.local_epochs):
                m = self._train_scaffold_epoch(c_local, c_global)
                last_loss = float(m["loss"])
                last_acc = float(m["accuracy"])

            w_after = get_parameters(self.model)
            K = max(self.local_epochs * len(self.train_loader), 1)
            lr = self.learning_rate

            # delta_c = (w_before - w_after) / (lr * K) - c_global
            delta_c = [
                (wb - wa) / (lr * K) - cg
                for wb, wa, cg in zip(w_before, w_after, c_global)
            ]

            # Update and persist local control variate
            new_c_local = [cl + dc for cl, dc in zip(c_local, delta_c)]
            self._save_c(f"local_{self.node_id}", new_c_local)

            # Pack delta_c as bytes for the server to aggregate
            extra_metrics["scaffold_delta_c"] = pickle.dumps(
                [dc.astype(np.float32) for dc in delta_c]
            )
            flog.info(
                "[%s] SCAFFOLD: delta_c norm=%.6f | c_local updated",
                self.node_id,
                float(np.mean([np.linalg.norm(dc) for dc in delta_c])),
            )

        fit_time = time.time() - fit_start
        updated_params = get_parameters(self.model)
        bytes_sent = sum(p.nbytes for p in updated_params)

        flog.info(
            "[%s] fit() done | loss=%.4f | acc=%.4f | bytes_sent=%d | time=%.2fs",
            self.node_id, last_loss, last_acc, bytes_sent, fit_time,
        )

        metrics = {
            "node_id": self.node_id,          # required by server for expert weighting
            "loss": last_loss,
            "accuracy": last_acc,
            "bytes_sent": float(bytes_sent),
            "bytes_received": float(bytes_received),
            "fit_time_sec": fit_time,
            **extra_metrics,
        }
        return updated_params, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        flog = logging.getLogger("fl_client")
        flog.info("[%s] evaluate() started", self.node_id)

        set_parameters(self.model, parameters)
        self.model.eval()

        all_preds, all_labels = [], []
        loss = 0.0

        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss += self.criterion(outputs, y).item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        loss /= max(len(self.eval_loader), 1)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        benign_mask = y_true == BENIGN_CLASS
        benign_recall = (
            float((y_pred[benign_mask] == BENIGN_CLASS).mean())
            if benign_mask.sum() > 0 else 0.0
        )
        fpr = 1.0 - benign_recall

        self.scheduler.step(loss)
        current_lr = self.optimizer.param_groups[0]["lr"]

        flog.info(
            "[%s] evaluate() done | loss=%.4f | acc=%.4f | f1=%.4f | fpr=%.4f | lr=%.6f",
            self.node_id, loss, acc, f1_macro, fpr, current_lr,
        )

        return float(loss), len(self.eval_loader.dataset), {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "benign_recall": benign_recall,
            "false_positive_rate": fpr,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    cfg = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)

    client_cfg = cfg.get("client", {})
    model_cfg  = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    imbalance_cfg = cfg.get("imbalance", {})
    runtime_cfg = cfg.get("runtime", {})

    node_id = args.node_id
    scenario = args.scenario

    server_address = (
        args.server_address or client_cfg.get("server_address", "127.0.0.1:8080")
    )
    local_epochs = (
        args.local_epochs if args.local_epochs is not None
        else client_cfg.get("local_epochs", 1)
    )
    batch_size = (
        args.batch_size if args.batch_size is not None
        else client_cfg.get("batch_size", 256)
    )
    learning_rate = (
        args.learning_rate if args.learning_rate is not None
        else client_cfg.get("learning_rate", 0.0005)
    )
    mu = (
        args.mu if args.mu is not None
        else float(client_cfg.get("mu", 0.0))
    )
    strategy = (
        args.strategy if args.strategy is not None
        else client_cfg.get("strategy", "fedavg")
    )
    imbalance_strategy = (
        args.imbalance_strategy if args.imbalance_strategy is not None
        else str(imbalance_cfg.get("name", "class_weights"))
    )
    focal_gamma = (
        args.focal_gamma if args.focal_gamma is not None
        else float(imbalance_cfg.get("focal_gamma", 2.0))
    )

    seed = int(runtime_cfg.get("seed", 42))
    device_name = runtime_cfg.get("device", "cpu")
    num_workers = int(runtime_cfg.get("num_workers", 0))

    set_seed(seed)

    # Resolve data directory (scenario-aware → legacy fallback)
    scenario_npz = get_processed_path(scenario, node_id)
    legacy_npz = PROCESSED_DIR / node_id / "train_preprocessed.npz"

    if scenario_npz.exists():
        node_dir = scenario_npz.parent
    elif scenario == "normal_noniid" and legacy_npz.exists():
        logger.warning("Falling back to legacy path %s", legacy_npz.parent)
        node_dir = legacy_npz.parent
    else:
        raise FileNotFoundError(
            f"Processed data not found for scenario='{scenario}', node='{node_id}'.\n"
            f"  Tried: {scenario_npz}\n"
            f"  Run:   python -m src.scripts.generate_scenarios --scenario {scenario}"
        )

    device = torch.device(device_name)

    train_loader, eval_loader = create_dataloaders_for_node(
        node_dir=node_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = MLPClassifier(
        input_dim=int(model_cfg.get("input_dim", 28)),
        hidden_dims=model_cfg.get("hidden_dims", [128, 64]),
        num_classes=int(dataset_cfg.get("num_classes", model_cfg.get("num_classes", 34))),
    )
    num_classes = int(dataset_cfg.get("num_classes", model_cfg.get("num_classes", 34)))
    validate_model_output_dim(model, num_classes)

    class_weights: torch.Tensor | None = None
    if imbalance_strategy in WEIGHTED_IMBALANCE_STRATEGIES:
        cw_path = ARTIFACTS_DIR / f"class_weights_{scenario}.pkl"
        class_weights = load_class_weights(cw_path, device=device)
        if class_weights is None:
            raise FileNotFoundError(
                f"Missing scenario-specific class weights: {cw_path}. "
                f"Run: python -m src.scripts.generate_weights --scenario {scenario}"
            )
        if int(class_weights.numel()) != num_classes:
            raise ValueError(
                f"Class weights at {cw_path} have {class_weights.numel()} entries "
                f"but num_classes={num_classes}."
            )
        logger.info("Loaded class weights from %s", cw_path)

    logger.info(
        "Starting client | node=%s | scenario=%s | strategy=%s | mu=%.4f | "
        "server=%s | epochs=%d | batch=%d | lr=%.5f | imbalance=%s",
        node_id, scenario, strategy, mu,
        server_address, local_epochs, batch_size, learning_rate, imbalance_strategy,
    )

    client = IoTFLClient(
        node_id=node_id,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        class_weights=class_weights,
        imbalance_strategy=imbalance_strategy,
        focal_gamma=focal_gamma,
        mu=mu,
        strategy=strategy,
    )

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
