from __future__ import annotations

import torch

from src.common.paths import DATA_DIR, ARTIFACTS_DIR
from src.data.dataloader import create_dataloaders_for_node
from src.model.evaluate import evaluate_model
from src.model.losses import build_loss, load_class_weights
from src.model.network import MLPClassifier
from src.model.train import train_one_epoch


def main():
    node_id = "node1"
    node_dir = DATA_DIR / "processed" / node_id

    batch_size = 256
    epochs = 3
    learning_rate = 1e-3

    train_loader, eval_loader = create_dataloaders_for_node(
        node_dir=node_dir,
        batch_size=batch_size,
        num_workers=0,
    )

    sample_batch = next(iter(train_loader))
    X_sample, _ = sample_batch

    input_dim = X_sample.shape[1]
    base_dataset = train_loader.dataset.dataset
    num_classes = int(base_dataset.y.max().item()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=(128, 64),
        dropout=0.2,
    ).to(device)

    class_weights_path = ARTIFACTS_DIR / "class_weights_34.pkl"
    class_weights = load_class_weights(class_weights_path, device=device)
    criterion = build_loss(class_weights=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    print("\n=== Local training summary ===")
    print(f"Node ID       : {node_id}")
    print(f"Input dim     : {input_dim}")
    print(f"Num classes   : {num_classes}")
    print(f"Device        : {device}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Eval batches  : {len(eval_loader)}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        eval_metrics = evaluate_model(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"eval_loss={eval_metrics['loss']:.4f} | "
            f"eval_acc={eval_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
