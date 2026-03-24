from __future__ import annotations

import torch


def evaluate_model(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

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