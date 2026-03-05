"""
Bonus 2: Partial Output Learning
==================================
Train models to predict only *part* of SIMON's ciphertext:
  - "full"  → all 32 bits
  - "left"  → top 16 bits  (left word of SIMON-32/64)
  - "right" → bottom 16 bits (right word of SIMON-32/64)

Hypothesis: Predicting half the output should be noticeably easier,
especially at low round counts where individual words diffuse slowly.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import get_dataloaders_partial
from src.models import LogisticRegressionModel, MLPModel, CNNModel
from src.train import train_model
from src.eval import evaluate_model

KEY       = 0x1918111009080100
CIPHER    = 'simon'
MAX_ROUNDS = 5
OUTPUT_SLICES = ['full', 'left', 'right']


def run_partial(num_samples=10000, epochs=15, device='cpu'):
    os.makedirs('results/bonus', exist_ok=True)
    os.makedirs('results/bonus/models', exist_ok=True)

    all_results = {}

    for r in range(1, MAX_ROUNDS + 1):
        all_results[r] = {}

        for slc in OUTPUT_SLICES:
            print(f"\n{'='*50}\nPartial | SIMON | rounds={r} | slice={slc}\n{'='*50}")

            (train_loader, val_loader, test_loader,
             block_size, output_size) = get_dataloaders_partial(
                num_samples=num_samples,
                rounds=r,
                key=KEY,
                cipher_name=CIPHER,
                output_slice=slc,
            )

            # input_size always = block_size (full plaintext given as input)
            # output_size = slice size (16 or 32 bits)
            models = {
                'MLP': MLPModel(input_size=block_size, output_size=output_size),
                'CNN': CNNModel(input_size=block_size, output_size=output_size),
            }

            slice_results = {}

            for name, model in models.items():
                save_path = f"results/bonus/models/partial_{CIPHER}_{slc}_{name}_r{r}.pt"

                train_model(model, train_loader, val_loader,
                            epochs=epochs, device=device, save_path=save_path)

                if os.path.exists(save_path):
                    model.load_state_dict(torch.load(save_path, weights_only=True))

                acc, hamming = evaluate_model(model, test_loader, device)
                print(f"  [{name}] slice={slc:5s}  acc={acc:.4f}  hamming={hamming:.2f}")

                slice_results[name] = {
                    'acc': acc,
                    'hamming': hamming,
                    'output_size': output_size,
                    'block_size': block_size,
                }

            all_results[r][slc] = slice_results

    out_path = 'results/bonus/partial_output.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved partial-output results → {out_path}")
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
    run_partial(num_samples=args.samples, epochs=args.epochs, device=device)
