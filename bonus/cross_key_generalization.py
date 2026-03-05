"""
Bonus 1: Cross-Key Generalization
===================================
Train a model on SIMON ciphertexts encrypted with key A,
then evaluate on ciphertexts encrypted with a *different* key B.

The experiment tests whether an ML model learns the cipher's structure
(key-independent) or merely memorises key-specific correlations.
"""

import os
import sys
import json
import argparse

# Allow running from project root as: python -m bonus.cross_key_generalization
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import get_dataloaders_cross_key, get_dataloaders
from src.models import LogisticRegressionModel, MLPModel, CNNModel
from src.train import train_model
from src.eval import evaluate_model

TRAIN_KEY = 0x1918111009080100   # Official SIMON-32/64 test-vector key
TEST_KEY  = 0xDEADBEEFCAFEBABE   # Unseen key during training

CIPHER    = 'simon'
MAX_ROUNDS = 5


def run_cross_key(num_samples=10000, epochs=15, device='cpu'):
    os.makedirs('results/bonus', exist_ok=True)
    os.makedirs('results/bonus/models', exist_ok=True)

    all_results = {}

    for r in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}\nCross-Key | SIMON | rounds={r}\n{'='*50}")

        (train_loader, val_loader,
         same_key_loader, cross_key_loader,
         block_size) = get_dataloaders_cross_key(
            num_samples=num_samples,
            rounds=r,
            train_key=TRAIN_KEY,
            test_key=TEST_KEY,
            cipher_name=CIPHER,
        )

        models = {
            'LR':  LogisticRegressionModel(input_size=block_size, output_size=block_size),
            'MLP': MLPModel(input_size=block_size, output_size=block_size),
            'CNN': CNNModel(input_size=block_size, output_size=block_size),
        }

        round_results = {}

        for name, model in models.items():
            save_path = f"results/bonus/models/crosskey_{CIPHER}_{name}_r{r}.pt"

            train_model(model, train_loader, val_loader,
                        epochs=epochs, device=device, save_path=save_path)

            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path, weights_only=True))

            same_acc,  same_ham  = evaluate_model(model, same_key_loader,  device)
            cross_acc, cross_ham = evaluate_model(model, cross_key_loader, device)

            print(f"  [{name}] same-key acc={same_acc:.4f}  cross-key acc={cross_acc:.4f}"
                  f"  Δacc={same_acc - cross_acc:+.4f}")

            round_results[name] = {
                'same_key':  {'acc': same_acc,  'hamming': same_ham},
                'cross_key': {'acc': cross_acc, 'hamming': cross_ham},
                'acc_drop':  same_acc - cross_acc
            }

        all_results[r] = round_results

    out_path = 'results/bonus/cross_key.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved cross-key results → {out_path}")
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
    run_cross_key(num_samples=args.samples, epochs=args.epochs, device=device)
