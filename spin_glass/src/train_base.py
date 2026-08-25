"""Train the 20-epoch baseline spin-glass diffusion model used by the paper.

The training archive is distributed separately because of its size.  This
script implements the baseline stage that precedes ``train_low_noise.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MiniAF3ScoreModel, sample_spherical_noise_and_target
from train_low_noise import SpinGlassDataset, collate_fn


SPIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = SPIN_ROOT / "data" / "train_40000"
DEFAULT_OUTPUT = SPIN_ROOT / "checkpoints" / "base_training"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dir = args.train_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = SpinGlassDataset(train_dir)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    device = torch.device(args.device)
    model = MiniAF3ScoreModel(c_s=64, c_z=32, num_blocks=4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        started = time.time()
        for sequence, clean, mask in loader:
            sequence = sequence.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            batch_size, sites = sequence.shape
            diffusion_time = 1.0e-4 + (1.0 - 1.0e-4) * torch.rand(
                (batch_size, 1), device=device
            )
            expanded_time = diffusion_time.unsqueeze(2).expand(batch_size, sites, 1)
            noisy, target = sample_spherical_noise_and_target(clean, expanded_time)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence, noisy, diffusion_time, mask)
            elementwise = expanded_time * (prediction - target).square()
            loss = elementwise[mask.unsqueeze(-1).expand_as(elementwise)].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        record = {
            "epoch": epoch,
            "mean_weighted_dsm_loss": total_loss / len(loader),
            "seconds": time.time() - started,
        }
        history.append(record)
        torch.save(model.state_dict(), output_dir / f"epoch_{epoch:02d}.pt")
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
