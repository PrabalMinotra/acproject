"""
Bonus 4: Distinguisher-Based Setup
=====================================
A *distinguisher* is a classifier that answers:

    "Is this 32-bit string a SIMON ciphertext, or just random bits?"

Dataset: 50% real ciphertexts (label=1) | 50% uniform random bits (label=0)
Input:   32-bit string (as a bit vector)
Output:  binary label

Metric: Distinguisher Advantage = 2 * (accuracy − 0.5)
  • 0 → pure guessing (cipher is indistinguishable from random)
  • 1 → perfect distinguisher

Expectation: advantage is high for very low rounds and drops toward 0
             as rounds increase (cipher approaches a pseudorandom permutation).
"""

import os
import sys
import json
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from src.dataset import CipherDataset, int_to_bit_array


# ---------------------------------------------------------------------------
# Distinguisher dataset
# ---------------------------------------------------------------------------

class DistinguisherDataset(Dataset):
    """
    50% real SIMON ciphertexts (label 1) and
    50% uniformly random 32-bit strings (label 0).
    """

    def __init__(self, num_samples, rounds, key, cipher_name='simon'):
        assert num_samples % 2 == 0, "num_samples must be even"
        half = num_samples // 2
        block_size = 32  # SIMON-32/64

        # Real ciphertexts ─────────────────────────────────────────────────
        real_ds = CipherDataset(half, rounds, key, cipher_name)
        self.block_size = real_ds.block_size

        self.samples = []
        self.labels  = []

        for i in range(half):
            _, ct_bits = real_ds[i]
            self.samples.append(ct_bits)
            self.labels.append(1.0)

        # Random bit strings (look like "random oracle" output) ────────────
        rng = np.random.default_rng(seed=99 + rounds)
        for _ in range(half):
            rand_val = int(rng.integers(0, 2**self.block_size))
            rand_bits = torch.tensor(int_to_bit_array(rand_val, self.block_size))
            self.samples.append(rand_bits)
            self.labels.append(0.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], torch.tensor([self.labels[idx]])


# ---------------------------------------------------------------------------
# Small binary classifier (reused for all 3 model types)
# ---------------------------------------------------------------------------

class BinaryLR(nn.Module):
    def __init__(self, input_size=32):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class BinaryMLP(nn.Module):
    def __init__(self, input_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256),        nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


class BinaryCNN(nn.Module):
    def __init__(self, input_size=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * input_size, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.input_size = input_size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_distinguisher(model, train_loader, val_loader, epochs, device, save_path):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for ct, label in train_loader:
            ct, label = ct.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(ct)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

        # Validation accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for ct, label in val_loader:
                ct, label = ct.to(device), label.to(device)
                preds = (model(ct) > 0).float()
                correct += (preds == label).sum().item()
                total   += label.numel()
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return best_val_acc


def eval_distinguisher(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ct, label in test_loader:
            ct, label = ct.to(device), label.to(device)
            preds = (model(ct) > 0).float()
            correct += (preds == label).sum().item()
            total   += label.numel()
    acc = correct / total
    advantage = 2 * (acc - 0.5)
    return acc, advantage


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

KEY       = 0x1918111009080100
CIPHER    = 'simon'
MAX_ROUNDS = 5


def run_distinguisher(num_samples=10000, epochs=15, device='cpu'):
    os.makedirs('results/bonus', exist_ok=True)
    os.makedirs('results/bonus/models', exist_ok=True)

    all_results = {}

    for r in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}\nDistinguisher | SIMON | rounds={r}\n{'='*50}")

        ds = DistinguisherDataset(num_samples, r, KEY, CIPHER)
        block_size = ds.block_size

        t_size = int(0.8 * num_samples)
        v_size = int(0.1 * num_samples)
        te_size = num_samples - t_size - v_size

        train_ds, val_ds, test_ds = random_split(
            ds, [t_size, v_size, te_size],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

        models = {
            'LR':  BinaryLR(block_size),
            'MLP': BinaryMLP(block_size),
            'CNN': BinaryCNN(block_size),
        }

        round_results = {}

        for name, model in models.items():
            save_path = f"results/bonus/models/distinguisher_{CIPHER}_{name}_r{r}.pt"

            train_distinguisher(model, train_loader, val_loader,
                                epochs=epochs, device=device, save_path=save_path)

            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path, weights_only=True))

            acc, adv = eval_distinguisher(model, test_loader, device)
            print(f"  [{name}] accuracy={acc:.4f}  advantage={adv:+.4f}")

            round_results[name] = {
                'accuracy':   acc,
                'advantage':  adv,   # 0 = random; 1 = perfect distinguisher
            }

        all_results[r] = round_results

    out_path = 'results/bonus/distinguisher.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved distinguisher results → {out_path}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs',  type=int, default=15)
    parser.add_argument('--fast',    action='store_true',
                        help='Quick smoke-test: 2000 samples, 5 epochs')
    args = parser.parse_args()

    if args.fast:
        args.samples, args.epochs = 2000, 5

    device = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')
    print(f"Device: {device}")
    run_distinguisher(num_samples=args.samples, epochs=args.epochs, device=device)
