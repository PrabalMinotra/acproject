import argparse
import json
import os
import random
import time

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import CIPHER_MAP, int_to_bit_array


class DistinguisherMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_distinguisher_loaders(cipher_name, rounds, key, num_samples, batch_size, seed):
    cipher_class = CIPHER_MAP[cipher_name]
    cipher = cipher_class(key)
    block_size = cipher.block_size

    rng = random.Random(seed)
    num_real = num_samples // 2
    num_rand = num_samples - num_real

    pts = []
    cts = []
    labels = []

    for _ in range(num_real):
        pt = rng.getrandbits(block_size)
        ct = cipher.encrypt(pt, rounds=rounds)
        pts.append(pt)
        cts.append(ct)
        labels.append(1)

    for _ in range(num_rand):
        pt = rng.getrandbits(block_size)
        ct = rng.getrandbits(block_size)
        pts.append(pt)
        cts.append(ct)
        labels.append(0)

    indices = list(range(num_samples))
    rng.shuffle(indices)

    features = np.zeros((num_samples, block_size * 2), dtype=np.float32)
    targets = np.zeros((num_samples, 1), dtype=np.float32)

    for row, idx in enumerate(indices):
        pt_bits = int_to_bit_array(pts[idx], bits=block_size)
        ct_bits = int_to_bit_array(cts[idx], bits=block_size)
        features[row, :block_size] = pt_bits
        features[row, block_size:] = ct_bits
        targets[row, 0] = labels[idx]

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features),
        torch.tensor(targets),
    )

    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, block_size


def train_distinguisher(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    return history, best_state


def evaluate_distinguisher(model, test_loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


def parse_hex_key(value):
    if value is None:
        return None
    text = value.strip().lower()
    if text.startswith('0x'):
        text = text[2:]
    if not text:
        raise ValueError('Empty hex key')
    return int(text, 16)


def parse_rounds(text):
    text = text.strip()
    if '-' in text:
        start, end = text.split('-', 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in text.split(',') if x.strip()]


def run_distinguisher(
    cipher_name,
    rounds,
    key,
    num_samples,
    epochs,
    batch_size,
    lr,
    seed,
    results_path,
):
    device = detect_device()
    print(f"Using device: {device}")

    results_dir = os.path.dirname(results_path)
    os.makedirs(results_dir, exist_ok=True)

    results = {cipher_name: {}}

    for r in rounds:
        train_loader, val_loader, test_loader, block_size = build_distinguisher_loaders(
            cipher_name=cipher_name,
            rounds=r,
            key=key,
            num_samples=num_samples,
            batch_size=batch_size,
            seed=seed,
        )

        model = DistinguisherMLP(input_size=block_size * 2)

        start_time = time.time()
        history, best_state = train_distinguisher(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
        )
        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc = evaluate_distinguisher(model, test_loader, device=device)
        elapsed = time.time() - start_time

        print(f"[r={r}] Test Acc: {test_acc:.4f} | Time: {elapsed:.1f}s")

        results[cipher_name][str(r)] = {
            'test_acc': test_acc,
            'history': history,
            'time_s': elapsed,
            'block_size': block_size,
            'input_bits': block_size * 2,
            'key': int(key),
        }

        with open(results_path, 'w', encoding='ascii') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distinguisher-based setup (PT || CT).')
    parser.add_argument('--cipher', type=str, default='simon')
    parser.add_argument('--rounds', type=str, default='1,2,3')
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results', type=str, default='bonus/results/distinguisher.json')
    args = parser.parse_args()

    run_distinguisher(
        cipher_name=args.cipher.lower(),
        rounds=parse_rounds(args.rounds),
        key=parse_hex_key(args.key) if args.key else 0x1918111009080100,
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        results_path=args.results,
    )
