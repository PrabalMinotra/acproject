"""
Bonus Plots Module
===================
Shared matplotlib helpers for all 4 bonus experiments.
Call individual functions or all_plots() after running experiments.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = 'results/bonus/plots'
ROUNDS   = list(range(1, 6))

MODEL_COLORS = {'LR': '#4C72B0', 'MLP': '#DD8452', 'CNN': '#55A868'}


def _savefig(name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Bonus 1: Cross-Key Generalization
# ---------------------------------------------------------------------------

def plot_cross_key(json_path='results/bonus/cross_key.json'):
    with open(json_path) as f:
        data = json.load(f)

    rounds = sorted(int(r) for r in data)
    models = list(next(iter(data.values())).keys())

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    fig.suptitle('Bonus 1 – Cross-Key Generalisation (SIMON-32/64)', fontsize=14, fontweight='bold')

    for ax, mname in zip(axes, models):
        same  = [data[str(r)][mname]['same_key']['acc']  for r in rounds]
        cross = [data[str(r)][mname]['cross_key']['acc'] for r in rounds]
        x = np.arange(len(rounds))
        width = 0.35
        ax.bar(x - width/2, same,  width, label='Same key',  color='#4C72B0', alpha=0.85)
        ax.bar(x + width/2, cross, width, label='Cross key', color='#DD8452', alpha=0.85)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='Chance (0.5)')
        ax.set_title(mname)
        ax.set_xlabel('Rounds')
        ax.set_xticks(x)
        ax.set_xticklabels(rounds)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Bit Accuracy')
    _savefig('cross_key.png')


# ---------------------------------------------------------------------------
# Bonus 2: Partial Output Learning
# ---------------------------------------------------------------------------

def plot_partial_output(json_path='results/bonus/partial_output.json'):
    with open(json_path) as f:
        data = json.load(f)

    rounds = sorted(int(r) for r in data)
    slices = ['full', 'left', 'right']
    models = list(next(iter(next(iter(data.values())).values())).keys())

    slice_colors = {'full': '#4C72B0', 'left': '#DD8452', 'right': '#55A868'}
    slice_ls     = {'full': '-', 'left': '--', 'right': ':'}

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), sharey=True)
    fig.suptitle('Bonus 2 – Partial Output Learning (SIMON-32/64)', fontsize=14, fontweight='bold')

    for ax, mname in zip(axes, models):
        for slc in slices:
            accs = [data[str(r)][slc][mname]['acc'] for r in rounds]
            ax.plot(rounds, accs, marker='o', color=slice_colors[slc],
                    linestyle=slice_ls[slc], label=slc, linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
        ax.set_title(mname)
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Bit Accuracy')
        ax.set_xticks(rounds)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=8)

    _savefig('partial_output.png')


# ---------------------------------------------------------------------------
# Bonus 3: SIMON vs SPECK
# ---------------------------------------------------------------------------

def plot_simon_vs_speck(json_path='results/bonus/simon_vs_speck.json'):
    with open(json_path) as f:
        data = json.load(f)

    rounds  = sorted(int(r) for r in data['simon'])
    ciphers = ['simon', 'speck']
    models  = list(next(iter(data['simon'].values())).keys())

    cipher_colors = {'simon': '#4C72B0', 'speck': '#C44E52'}
    cipher_ls     = {'simon': '-', 'speck': '--'}

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    fig.suptitle('Bonus 3 – SIMON vs SPECK (32/64-bit, same rounds)', fontsize=14, fontweight='bold')

    for ax, mname in zip(axes, models):
        for cipher in ciphers:
            accs = [data[cipher][str(r)][mname]['acc'] for r in rounds]
            ax.plot(rounds, accs, marker='o', color=cipher_colors[cipher],
                    linestyle=cipher_ls[cipher], label=cipher.upper(), linewidth=2)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, label='Chance')
        ax.set_title(mname)
        ax.set_xlabel('Rounds')
        ax.set_xticks(rounds)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Bit Accuracy')
    _savefig('simon_vs_speck.png')


# ---------------------------------------------------------------------------
# Bonus 4: Distinguisher Advantage
# ---------------------------------------------------------------------------

def plot_distinguisher(json_path='results/bonus/distinguisher.json'):
    with open(json_path) as f:
        data = json.load(f)

    rounds = sorted(int(r) for r in data)
    models = list(next(iter(data.values())).keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Bonus 4 – Distinguisher (SIMON vs Random, 32-bit)', fontsize=14, fontweight='bold')

    for mname, color in MODEL_COLORS.items():
        if mname not in models:
            continue
        accs  = [data[str(r)][mname]['accuracy']  for r in rounds]
        advs  = [data[str(r)][mname]['advantage']  for r in rounds]
        ax1.plot(rounds, accs, marker='o', color=color, label=mname, linewidth=2)
        ax2.plot(rounds, advs, marker='s', color=color, label=mname, linewidth=2)

    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax1.set_title('Binary Accuracy')
    ax1.set_xlabel('Rounds')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(rounds)
    ax1.set_ylim(0.45, 1.05)
    ax1.legend()

    ax2.axhline(0.0, color='red', linestyle='--', linewidth=0.8, label='No advantage')
    ax2.set_title('Distinguisher Advantage  (2·(acc−0.5))')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('Advantage')
    ax2.set_xticks(rounds)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()

    _savefig('distinguisher.png')


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------

def all_plots():
    results = {
        'cross_key':    'results/bonus/cross_key.json',
        'partial':      'results/bonus/partial_output.json',
        'simon_speck':  'results/bonus/simon_vs_speck.json',
        'distinguisher':'results/bonus/distinguisher.json',
    }
    fns = {
        'cross_key':    plot_cross_key,
        'partial':      plot_partial_output,
        'simon_speck':  plot_simon_vs_speck,
        'distinguisher':plot_distinguisher,
    }
    for key, path in results.items():
        if os.path.exists(path):
            print(f"Plotting {key}...")
            fns[key](path)
        else:
            print(f"  [skip] {path} not found – run the experiment first.")


if __name__ == '__main__':
    all_plots()
