import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np

MODELS = ['LogisticRegression', 'MLP', 'CNN']

def generate_comparative_plots(metrics_path='results/metrics.json'):
    with open(metrics_path, 'r') as f:
        data = json.load(f)
        
    os.makedirs('results/plots', exist_ok=True)
    
    ciphers = list(data.keys())
    
    def build_ticks(rounds_list):
        if not rounds_list:
            return []
        min_r, max_r = min(rounds_list), max(rounds_list)
        if max_r - min_r <= 10:
            return list(range(min_r, max_r + 1))
        step = max(1, (max_r - min_r) // 10)
        return list(range(min_r, max_r + 1, step))

    
    plt.figure(figsize=(14, 8))
    for cipher in ciphers:
        rounds = sorted([int(r) for r in data[cipher].keys()])
        accs = []
        for r in rounds:
            
            best_acc = max([data[cipher][str(r)][m]['test_acc'] for m in MODELS])
            accs.append(best_acc)
            
        plt.plot(rounds, accs, marker='o', label=cipher.upper())
        
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing (50%)')
    plt.axhline(y=1.0, color='g', linestyle='--', label='Perfect Reconstruction (100%)')
    plt.title('Best Model Accuracy vs. Number of Rounds Across 16 Ciphers')
    plt.xlabel('Number of Rounds (r)')
    plt.ylabel('Bitwise Prediction Accuracy')
    ticks = build_ticks([int(r) for cipher in ciphers for r in data[cipher].keys()])
    if ticks:
        plt.xticks(ticks)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/plots/all_ciphers_accuracy.png', dpi=300)
    plt.close()
    
    
    plt.figure(figsize=(14, 8))
    for cipher in ciphers:
        rounds = sorted([int(r) for r in data[cipher].keys()])
        h_dists = []
        
        for r in rounds:
            block_size = data[cipher][str(r)][MODELS[0]]['block_size']
            output_bits = data[cipher][str(r)][MODELS[0]].get('output_bits', block_size)
            
            best_h_dist = min([data[cipher][str(r)][m]['avg_hamming'] for m in MODELS])
            h_dists.append(best_h_dist / output_bits) 
            
        plt.plot(rounds, h_dists, marker='^', label=cipher.upper())
        
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance (50% bits differ)')
    plt.title('Normalized Hamming Distance vs. Number of Rounds Across 16 Ciphers')
    plt.xlabel('Number of Rounds (r)')
    plt.ylabel('Normalized Hamming Distance (Distance / Output Bits)')
    if ticks:
        plt.xticks(ticks)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/plots/all_ciphers_hamming_distance.png', dpi=300)
    plt.close()

def generate_convergence_plots(metrics_path='results/metrics.json', output_dir='results/plots/convergence'):
    with open(metrics_path, 'r') as f:
        data = json.load(f)

    for cipher, rounds_dict in data.items():
        cipher_dir = os.path.join(output_dir, cipher)
        os.makedirs(cipher_dir, exist_ok=True)

        rounds = sorted([int(r) for r in rounds_dict.keys()])
        for r in rounds:
            round_key = str(r)
            for model in MODELS:
                history = rounds_dict[round_key][model].get('history')
                if not history:
                    continue

                epochs = list(range(1, len(history['train_loss']) + 1))
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

                axes[0].plot(epochs, history['train_loss'], label='train')
                axes[0].plot(epochs, history['val_loss'], label='val')
                axes[0].set_ylabel('Loss')
                axes[0].set_title(f'{cipher.upper()} {model} r={r}')
                axes[0].legend()
                axes[0].grid(True)

                axes[1].plot(epochs, history['val_acc'], color='green')
                axes[1].set_ylabel('Val Acc')
                axes[1].set_xlabel('Epoch')
                axes[1].grid(True)

                out_path = os.path.join(cipher_dir, f'{model}_r{r}.png')
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close()

def compute_max_meaningful_rounds(
    metrics_path='results/metrics.json',
    output_path='results/max_meaningful_rounds.json',
    acc_threshold=0.55,
    hamming_threshold=0.45
):
    with open(metrics_path, 'r') as f:
        data = json.load(f)

    summary = {
        'thresholds': {
            'acc': acc_threshold,
            'norm_hamming': hamming_threshold
        },
        'per_cipher': {}
    }

    for cipher, rounds_dict in data.items():
        rounds = sorted([int(r) for r in rounds_dict.keys()])
        cipher_entry = {
            'models': {},
            'best_model': None
        }

        for model in MODELS:
            max_round = None
            for r in rounds:
                metrics = rounds_dict[str(r)][model]
                acc = metrics['test_acc']
                output_bits = metrics.get('output_bits', metrics['block_size'])
                norm_hamming = metrics['avg_hamming'] / output_bits
                if acc >= acc_threshold and norm_hamming <= hamming_threshold:
                    max_round = r
            cipher_entry['models'][model] = max_round

        
        best_model = None
        best_round = None
        best_acc = -1.0
        for model, max_round in cipher_entry['models'].items():
            if max_round is None:
                continue
            if best_round is None or max_round > best_round:
                best_round = max_round
                best_model = model
                best_acc = rounds_dict[str(max_round)][model]['test_acc']
            elif max_round == best_round:
                acc = rounds_dict[str(max_round)][model]['test_acc']
                if acc > best_acc:
                    best_model = model
                    best_acc = acc

        cipher_entry['best_model'] = {
            'model': best_model,
            'max_round': best_round
        }

        summary['per_cipher'][cipher] = cipher_entry

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots and summaries from metrics.')
    parser.add_argument('--metrics', type=str, default='results/metrics.json')
    parser.add_argument('--acc-threshold', type=float, default=0.55)
    parser.add_argument('--hamming-threshold', type=float, default=0.45)
    parser.add_argument('--summary-out', type=str, default='results/max_meaningful_rounds.json')
    args = parser.parse_args()

    generate_comparative_plots(metrics_path=args.metrics)
    generate_convergence_plots(metrics_path=args.metrics)
    compute_max_meaningful_rounds(
        metrics_path=args.metrics,
        output_path=args.summary_out,
        acc_threshold=args.acc_threshold,
        hamming_threshold=args.hamming_threshold
    )
    print("Plots and summaries saved to results")
