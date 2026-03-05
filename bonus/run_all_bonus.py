"""
Bonus Master Runner
======================
Runs all 4 bonus experiments sequentially, then generates all plots.

Usage:
    # Full run (10 000 samples, 15 epochs each)
    python -m bonus.run_all_bonus

    # Fast smoke-test (2 000 samples, 5 epochs each)
    python -m bonus.run_all_bonus --fast

    # Only generate plots from existing results
    python -m bonus.run_all_bonus --plots-only
"""

import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    parser = argparse.ArgumentParser(description='Run all 4 bonus experiments')
    parser.add_argument('--samples',    type=int, default=10000,
                        help='Samples per experiment (default: 10 000)')
    parser.add_argument('--epochs',     type=int, default=15,
                        help='Training epochs per model (default: 15)')
    parser.add_argument('--fast',       action='store_true',
                        help='Smoke-test mode: 2 000 samples, 5 epochs')
    parser.add_argument('--plots-only', action='store_true',
                        help='Skip experiments, only regenerate plots')
    args = parser.parse_args()

    if args.fast:
        args.samples, args.epochs = 2000, 5

    device = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')

    print("=" * 60)
    print(" Bonus Experiments – Master Runner")
    print("=" * 60)
    print(f"  Device  : {device}")
    print(f"  Samples : {args.samples}")
    print(f"  Epochs  : {args.epochs}")
    print("=" * 60)

    os.makedirs('results/bonus', exist_ok=True)
    summary = {}

    if not args.plots_only:

        # ── Bonus 1: Cross-Key Generalisation ──────────────────────────────
        print("\n\n[ BONUS 1 ] Cross-Key Generalisation")
        t0 = time.time()
        from bonus.cross_key_generalization import run_cross_key
        results1 = run_cross_key(num_samples=args.samples, epochs=args.epochs, device=device)
        summary['bonus1_cross_key'] = {
            'elapsed_s': round(time.time() - t0, 1),
            'rounds': list(results1.keys()),
        }

        # ── Bonus 2: Partial Output Learning ───────────────────────────────
        print("\n\n[ BONUS 2 ] Partial Output Learning")
        t0 = time.time()
        from bonus.partial_output_learning import run_partial
        results2 = run_partial(num_samples=args.samples, epochs=args.epochs, device=device)
        summary['bonus2_partial'] = {
            'elapsed_s': round(time.time() - t0, 1),
        }

        # ── Bonus 3: SIMON vs SPECK ─────────────────────────────────────────
        print("\n\n[ BONUS 3 ] SIMON vs SPECK")
        t0 = time.time()
        from bonus.simon_vs_speck import run_simon_vs_speck
        results3 = run_simon_vs_speck(num_samples=args.samples, epochs=args.epochs, device=device)
        summary['bonus3_simon_speck'] = {
            'elapsed_s': round(time.time() - t0, 1),
        }

        # ── Bonus 4: Distinguisher ──────────────────────────────────────────
        print("\n\n[ BONUS 4 ] Distinguisher")
        t0 = time.time()
        from bonus.distinguisher import run_distinguisher
        results4 = run_distinguisher(num_samples=args.samples, epochs=args.epochs, device=device)
        summary['bonus4_distinguisher'] = {
            'elapsed_s': round(time.time() - t0, 1),
        }

        with open('results/bonus/run_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        print("\n\nAll experiments complete. Summary → results/bonus/run_summary.json")

    # ── Generate Plots ──────────────────────────────────────────────────────
    print("\n\n[ PLOTS ] Generating all bonus plots...")
    from bonus.plots import all_plots
    all_plots()
    print("\nDone! Plots are in results/bonus/plots/")

    # ── Print quick result table ────────────────────────────────────────────
    print_summary()


def print_summary():
    """Print a concise result table if all result files are present."""
    import os, json

    files = {
        'Bonus 1 – Cross-Key':      'results/bonus/cross_key.json',
        'Bonus 2 – Partial Output': 'results/bonus/partial_output.json',
        'Bonus 3 – SIMON vs SPECK': 'results/bonus/simon_vs_speck.json',
        'Bonus 4 – Distinguisher':  'results/bonus/distinguisher.json',
    }

    print("\n" + "=" * 60)
    print(" Quick Result Summary")
    print("=" * 60)

    for label, path in files.items():
        if not os.path.exists(path):
            print(f"\n  {label}: [not found]")
            continue

        with open(path) as f:
            data = json.load(f)

        print(f"\n  {label}")

        if 'cross_key' in path:
            # Show MLP drop for each round
            for r in sorted(data, key=int):
                entry = data[r].get('MLP', {})
                same  = entry.get('same_key',  {}).get('acc', float('nan'))
                cross = entry.get('cross_key', {}).get('acc', float('nan'))
                print(f"    r={r}: same={same:.3f}  cross={cross:.3f}  Δ={same-cross:+.3f}")

        elif 'partial' in path:
            for r in sorted(data, key=int):
                row = []
                for slc in ['full','left','right']:
                    acc = data[r].get(slc, {}).get('MLP', {}).get('acc', float('nan'))
                    row.append(f"{slc}={acc:.3f}")
                print(f"    r={r}: {' | '.join(row)}")

        elif 'simon_vs_speck' in path:
            rounds = sorted(data.get('simon', {}), key=int)
            for r in rounds:
                s_acc = data['simon'][r].get('MLP', {}).get('acc', float('nan'))
                sp_acc= data['speck'][r].get('MLP', {}).get('acc', float('nan'))
                print(f"    r={r}: SIMON_MLP={s_acc:.3f}  SPECK_MLP={sp_acc:.3f}")

        elif 'distinguisher' in path:
            for r in sorted(data, key=int):
                adv = data[r].get('MLP', {}).get('advantage', float('nan'))
                acc = data[r].get('MLP', {}).get('accuracy',  float('nan'))
                print(f"    r={r}: MLP acc={acc:.3f}  advantage={adv:+.3f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
