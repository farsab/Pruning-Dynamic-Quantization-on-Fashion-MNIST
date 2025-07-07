from __future__ import annotations
import os, time, math, tempfile, copy, pathlib
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as tq
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Util functions
# --------------------------------------------------------------------------- #
def get_dataloaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    tfms = transforms.Compose([transforms.ToTensor()])
    root = pathlib.Path(".").expanduser() / "data"
    train_set = datasets.FashionMNIST(root, train=True, download=True, transform=tfms)
    test_set = datasets.FashionMNIST(root, train=False, download=True, transform=tfms)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    for xb, yb in tqdm(loader, desc="Train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optim.step()


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


def model_size_mb(state_dict_path: str) -> float:
    return os.path.getsize(state_dict_path) / 2 ** 20  # MiB


def save_model(model, name: str) -> str:
    path = f"{name}.pt"
    torch.save(model.state_dict(), path)
    return path


# --------------------------------------------------------------------------- #
# Compression steps
# --------------------------------------------------------------------------- #
def apply_global_pruning(model: nn.Module, amount: float = 0.5) -> nn.Module:
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    # Remove re-param hooks to make pruning permanent
    for m, _ in parameters_to_prune:
        prune.remove(m, "weight")
    return model


def quantize_model(model: nn.Module) -> nn.Module:
    return tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, test_dl = get_dataloaders()

    baseline = SmallCNN().to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("⏳ Training baseline (1 epoch)…")
    train_one_epoch(baseline, train_dl, optimizer, criterion, device)
    base_acc = evaluate(baseline, test_dl, device)
    base_path = save_model(baseline.cpu(), "baseline")

    # --- Pruning + Quantization ------------------------------------------- #
    print("✂️  Applying 50 % global unstructured pruning…")
    pruned = copy.deepcopy(baseline)
    pruned = apply_global_pruning(pruned, amount=0.5)

    print("🔧 Dynamic quantization of linear layers…")
    pruned_quant = quantize_model(pruned)

    pq_acc = evaluate(pruned_quant.to(device), test_dl, device)
    pq_path = save_model(pruned_quant.cpu(), "pruned_quant")

    # ---------------------------------------------------------------------- #
    results = pd.concat(
        [
            pd.DataFrame(
                [
                    dict(
                        Model="Baseline",
                        Accuracy=base_acc,
                        Size_MB=model_size_mb(base_path),
                    )
                ]
            ),
            pd.DataFrame(
                [
                    dict(
                        Model="Pruned+Quantized",
                        Accuracy=pq_acc,
                        Size_MB=model_size_mb(pq_path),
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    print("\n=== Compression Results ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nFinished in {time.time() - t0:.1f} s")
