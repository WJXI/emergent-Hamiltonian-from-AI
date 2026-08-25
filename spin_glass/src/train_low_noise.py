"""Controlled continuation and low-noise fine-tuning from the epoch-20 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SPIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MiniAF3ScoreModel, sample_spherical_noise_and_target


DEFAULT_TRAIN_DIR = SPIN_ROOT / "data" / "train_40000"
DEFAULT_START_CHECKPOINT = SPIN_ROOT / "checkpoints" / "epoch_20.pt"
DEFAULT_OUTPUT_DIR = SPIN_ROOT / "checkpoints" / "low_noise_mixture"
SEED = 20260801


class SpinGlassDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.sequences: list[str] = []
        self.spins: list[np.ndarray] = []
        for path in sorted(data_dir.glob("chunk_*.npz")):
            data = np.load(path, allow_pickle=True)
            self.sequences.extend(str(x) for x in data["sequences"])
            self.spins.extend(data["spins"])
        if not self.sequences:
            raise FileNotFoundError(f"No training chunks found in {data_dir}")
        print(f"loaded {len(self.sequences)} training sequences", flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        seq = torch.tensor([0 if token == "A" else 1 for token in sequence], dtype=torch.long)
        snapshot_index = np.random.randint(len(self.spins[index]))
        spins = torch.as_tensor(self.spins[index][snapshot_index], dtype=torch.float32)
        return seq, spins


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    max_len = 50
    batch_size = len(batch)
    sequences = torch.zeros((batch_size, max_len), dtype=torch.long)
    spins = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for row, (sequence, spin_array) in enumerate(batch):
        length = len(sequence)
        sequences[row, :length] = sequence
        spins[row, :length] = spin_array
        mask[row, :length] = True
    return sequences, spins, mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_time(branch: str, batch_size: int, device: torch.device) -> torch.Tensor:
    original = torch.rand((batch_size, 1), device=device) * 0.9999 + 1.0e-4
    log_min = math.log(1.0e-4)
    log_max = math.log(1.0e-2)
    low_noise = torch.exp(
        torch.rand((batch_size, 1), device=device) * (log_max - log_min) + log_min
    )
    choose_low_noise = torch.rand((batch_size, 1), device=device) < 0.5
    # Generate all three random tensors in both branches so that the subsequent
    # spherical noise uses the same random-number stream in the paired runs.
    if branch == "uniform_continue":
        return original
    if branch != "low_noise_mixture":
        raise ValueError(f"Unknown branch: {branch}")
    return torch.where(choose_low_noise, low_noise, original)


def save_checkpoint(
    path: Path,
    model: MiniAF3ScoreModel,
    optimizer: torch.optim.Optimizer,
    branch: str,
    completed_epoch: int,
    history: list[dict],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "branch": branch,
            "completed_epoch": completed_epoch,
            "history": history,
            "seed": SEED,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        required=True,
        choices=["uniform_continue", "low_noise_mixture"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--start-checkpoint", type=Path, default=DEFAULT_START_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    branch_dir = args.output_dir.resolve()
    result_dir = SPIN_ROOT / "results"
    branch_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    latest_path = branch_dir / "latest.pt"

    train_dir = args.train_dir.resolve()
    start_checkpoint = args.start_checkpoint.resolve()
    dataset = SpinGlassDataset(train_dir)
    generator = torch.Generator().manual_seed(SEED)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    model = MiniAF3ScoreModel(c_s=64, c_z=32, num_blocks=4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict] = []
    first_epoch = 1

    if args.resume and latest_path.exists():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload.get("history", []))
        first_epoch = int(payload["completed_epoch"]) + 1
        print(f"resuming {args.branch} at epoch {first_epoch}", flush=True)
    else:
        model.load_state_dict(torch.load(start_checkpoint, map_location=device, weights_only=True))
        print(f"starting {args.branch} from original epoch 20", flush=True)

    config = {
        "branch": args.branch,
        "start_checkpoint": str(start_checkpoint),
        "start_checkpoint_sha256": sha256(start_checkpoint),
        "train_dir": str(train_dir),
        "seed": SEED,
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "training_sequences": len(dataset),
        "device": str(device),
        "torch_version": torch.__version__,
        "time_sampling": (
            "Uniform(1e-4,1)"
            if args.branch == "uniform_continue"
            else "0.5*Uniform(1e-4,1) + 0.5*LogUniform(1e-4,1e-2)"
        ),
    }
    (result_dir / f"{args.branch}_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config), flush=True)

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        low_t_examples = 0
        start_time = time.time()

        for batch_index, (sequence, clean_spins, mask) in enumerate(dataloader, start=1):
            sequence = sequence.to(device, non_blocking=True)
            clean_spins = clean_spins.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            batch_size, num_sites = sequence.shape

            time_batch = sample_time(args.branch, batch_size, device)
            expanded_time = time_batch.unsqueeze(2).expand(batch_size, num_sites, 1)
            noisy_spins, target_score = sample_spherical_noise_and_target(
                clean_spins, expanded_time
            )

            optimizer.zero_grad(set_to_none=True)
            predicted_score = model(sequence, noisy_spins, time_batch, mask)
            loss_all = expanded_time * (predicted_score - target_score) ** 2
            loss = loss_all[mask.unsqueeze(-1).expand_as(loss_all)].mean()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_examples += batch_size
            low_t_examples += int((time_batch < 1.0e-3).sum().item())
            if batch_index % 50 == 0:
                print(
                    f"branch={args.branch} epoch={epoch} batch={batch_index}/{len(dataloader)} "
                    f"loss={loss.item():.6f} grad_norm={float(gradient_norm):.4f}",
                    flush=True,
                )

        row = {
            "epoch_after_20": epoch,
            "effective_original_epoch": 20 + epoch,
            "average_batch_loss": total_loss / len(dataloader),
            "low_t_fraction_below_1e-3": low_t_examples / total_examples,
            "seconds": time.time() - start_time,
        }
        history.append(row)
        epoch_path = branch_dir / f"epoch_{20 + epoch:02d}.pt"
        save_checkpoint(epoch_path, model, optimizer, args.branch, epoch, history)
        save_checkpoint(latest_path, model, optimizer, args.branch, epoch, history)
        (result_dir / f"{args.branch}_training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)

    print(f"completed branch={args.branch}; checkpoint={latest_path}", flush=True)


if __name__ == "__main__":
    main()
