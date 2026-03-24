from __future__ import annotations

import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / max(len(loader), 1)
    acc = correct / total if total > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "num_samples": total,
    }


def train_local(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epochs: int = 1,
):
    history = []

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        metrics["epoch"] = epoch
        history.append(metrics)

    return history