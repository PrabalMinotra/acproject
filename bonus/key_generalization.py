import argparse
import hashlib
import json
import os
import time

import torch

from src.dataset import CIPHER_MAP, get_dataloaders
from src.models import LogisticRegressionModel, MLPModel, CNNModel
from src.train import train_model
from src.eval import evaluate_model

DEFAULT_CIPHERS = ['simon', 'katan']

DEFAULT_KEYS = {
    'simon': {
        'train': 0x1918111009080100,
        'test': 0x0f0e0d0c0b0a0908,
    },
    'katan': {
        'train': 0x00010203040506070809,
        'test': 0x0f1e2d3c4b5a69788796,
    },
}

MODELS = {
    'LogisticRegression': LogisticRegressionModel,
    'MLP': MLPModel,
    'CNN': CNNModel,
}


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


def parse_ciphers(text):
    if not text:
        return DEFAULT_CIPHERS
    return [c.strip().lower() for c in text.split(',') if c.strip()]


def cipher_key_size(cipher_name):
    if cipher_name not in CIPHER_MAP:
        raise ValueError(f"Unknown cipher: {cipher_name}")
    cipher = CIPHER_MAP[cipher_name](0)
    return int(getattr(cipher, 'key_size', 0))


def derive_default_key(cipher_name, label, key_bits):
    if key_bits <= 0:
        raise ValueError(f"Invalid key size for {cipher_name}: {key_bits}")
    needed_bytes = (key_bits + 7) // 8
    material = bytearray()
    counter = 0
    while len(material) < needed_bytes:
        seed = f"{cipher_name}:{label}:{counter}".encode('ascii')
        material.extend(hashlib.sha256(seed).digest())
        counter += 1
    key_int = int.from_bytes(material[:needed_bytes], 'big')
    if key_bits % 8:
        key_int &= (1 << key_bits) - 1
    return key_int


def default_key_pair(cipher_name):
    if cipher_name in DEFAULT_KEYS:
        train_key = DEFAULT_KEYS[cipher_name]['train']
        test_key = DEFAULT_KEYS[cipher_name]['test']
        return train_key, test_key
    key_bits = cipher_key_size(cipher_name)
    return (
        derive_default_key(cipher_name, 'train', key_bits),
        derive_default_key(cipher_name, 'test', key_bits),
    )


def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def cache_dir_for_key(cache_root, cipher_name, key_value, label):
    key_hex = f"{key_value:x}"
    return os.path.join(cache_root, cipher_name, f"{label}_{key_hex}")


def resolve_key(cipher_name, override_key, label, defaults):
    if override_key is not None:
        return override_key
    return defaults[0] if label == 'train' else defaults[1]


def run_key_generalization(
    ciphers,
    rounds,
    train_key_override,
    test_key_override,
    num_samples,
    epochs,
    batch_size,
    cache_root,
    use_cache,
    save_cache,
    results_path,
):
    device = detect_device()
    print(f"Using device: {device}")

    results_dir = os.path.dirname(results_path)
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    all_results = {}

    for cipher_name in ciphers:
        train_default, test_default = default_key_pair(cipher_name)

        train_key = resolve_key(cipher_name, train_key_override, 'train', (train_default, test_default))
        test_key = resolve_key(cipher_name, test_key_override, 'test', (train_default, test_default))

        cipher_results = all_results.setdefault(cipher_name, {})

        for r in rounds:
            print(f"\n{'=' * 60}\n{cipher_name.upper()} r={r}\n{'=' * 60}")

            train_cache = cache_dir_for_key(cache_root, cipher_name, train_key, 'train')
            test_cache = cache_dir_for_key(cache_root, cipher_name, test_key, 'test')

            train_loader, val_loader, _, block_size, output_bits = get_dataloaders(
                num_samples=num_samples,
                rounds=r,
                key=train_key,
                cipher_name=cipher_name,
                batch_size=batch_size,
                cache_dir=train_cache,
                use_cache=use_cache,
                save_cache=save_cache,
            )

            _, _, test_loader, test_block_size, test_output_bits = get_dataloaders(
                num_samples=num_samples,
                rounds=r,
                key=test_key,
                cipher_name=cipher_name,
                batch_size=batch_size,
                cache_dir=test_cache,
                use_cache=use_cache,
                save_cache=save_cache,
            )

            if block_size != test_block_size or output_bits != test_output_bits:
                raise ValueError("Train/test loaders have mismatched dimensions")

            round_results = cipher_results.setdefault(str(r), {})

            key_tag = f"train{train_key:x}_test{test_key:x}"
            for model_name, model_cls in MODELS.items():
                model = model_cls(input_size=block_size, output_size=block_size)
                save_path = os.path.join(
                    models_dir,
                    f"{cipher_name}_{model_name}_r{r}_{key_tag}.pt",
                )

                start_time = time.time()
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    device=device,
                    save_path=save_path,
                )

                if os.path.exists(save_path):
                    model.load_state_dict(torch.load(save_path, weights_only=True))

                test_acc, avg_hamming = evaluate_model(model, test_loader, device=device)
                elapsed = time.time() - start_time

                print(
                    f"[{model_name}] Test Acc: {test_acc:.4f} | "
                    f"Avg Hamming: {avg_hamming:.2f} | Time: {elapsed:.1f}s"
                )

                round_results[model_name] = {
                    'train_key': int(train_key),
                    'test_key': int(test_key),
                    'test_acc': test_acc,
                    'avg_hamming': avg_hamming,
                    'history': history,
                    'time_s': elapsed,
                    'block_size': block_size,
                    'output_bits': output_bits,
                }

                with open(results_path, 'w', encoding='ascii') as f:
                    json.dump(all_results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on one key, test on a different key.')
    parser.add_argument('--ciphers', type=str, default=','.join(DEFAULT_CIPHERS))
    parser.add_argument('--rounds', type=str, default='5')
    parser.add_argument('--train-key', type=str, default=None)
    parser.add_argument('--test-key', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--cache-root', type=str, default='bonus/data')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--no-save-cache', action='store_true')
    parser.add_argument('--results', type=str, default='bonus/results/key_generalization.json')
    args = parser.parse_args()

    run_key_generalization(
        ciphers=parse_ciphers(args.ciphers),
        rounds=parse_rounds(args.rounds),
        train_key_override=parse_hex_key(args.train_key),
        test_key_override=parse_hex_key(args.test_key),
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cache_root=args.cache_root,
        use_cache=not args.no_cache,
        save_cache=not args.no_save_cache,
        results_path=args.results,
    )
