"""
Bonus 3: SIMON vs SPECK Comparison
=====================================
Both SIMON-32/64 and SPECK-32/64 are NSA-designed lightweight block ciphers
from the same family (ARX-based), same block/key size, but different structures.
This experiment pits them head-to-head using the same ML pipeline.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import get_dataloaders
from src.models import LogisticRegressionModel, MLPModel, CNNModel
from src.train import train_model
from src.eval import evaluate_model

KEY       = 0x1918111009080100
MAX_ROUNDS = 5
CIPHERS   = ['simon', 'speck']


def run_simon_vs_speck(num_samples=10000, epochs=15, device='cpu'):
    os.makedirs('results/bonus', exist_ok=True)
    os.makedirs('results/bonus/models', exist_ok=True)

    all_results = {}

    for cipher_name in CIPHERS:
        print(f"\n{'#'*55}\nCipher: {cipher_name.upper()}\n{'#'*55}")
        all_results[cipher_name] = {}

        for r in range(1, MAX_ROUNDS + 1):
            print(f"\n{'='*45}\n{cipher_name.upper()} | rounds={r}\n{'='*45}")

            train_loader, val_loader, test_loader, block_size = get_dataloaders(
                num_samples=num_samples, rounds=r, key=KEY,
                cipher_name=cipher_name, batch_size=1024
            )

            models = {
                'LR':  LogisticRegressionModel(input_size=block_size, output_size=block_size),
                'MLP': MLPModel(input_size=block_size, output_size=block_size),
                'CNN': CNNModel(input_size=block_size, output_size=block_size),
            }

            round_results = {}

            for name, model in models.items():
                save_path = f"results/bonus/models/simonspeck_{cipher_name}_{name}_r{r}.pt"

                train_model(model, train_loader, val_loader,
                            epochs=epochs, device=device, save_path=save_path)

                if os.path.exists(save_path):
                    model.load_state_dict(torch.load(save_path, weights_only=True))

                acc, hamming = evaluate_model(model, test_loader, device)
                print(f"  [{name}] acc={acc:.4f}  hamming={hamming:.2f}")

                round_results[name] = {'acc': acc, 'hamming': hamming}

            all_results[cipher_name][r] = round_results

    out_path = 'results/bonus/simon_vs_speck.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved SIMON vs SPECK results → {out_path}")
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
    run_simon_vs_speck(num_samples=args.samples, epochs=args.epochs, device=device)
