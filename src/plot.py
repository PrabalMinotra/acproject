import json
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_comparative_plots(metrics_path='results/metrics.json'):
    with open(metrics_path, 'r') as f:
        data = json.load(f)
        
    os.makedirs('results/plots', exist_ok=True)
    
    models = ['LogisticRegression', 'MLP', 'CNN']
    ciphers = list(data.keys())
    
    # 1. Comparative Accuracy Plot across all 12 ciphers (Using MLP as best case representation)
    plt.figure(figsize=(14, 8))
    for cipher in ciphers:
        rounds = sorted([int(r) for r in data[cipher].keys()])
        accs = []
        for r in rounds:
            # Let's track the best performing model (usually CNN or MLP)
            best_acc = max([data[cipher][str(r)][m]['test_acc'] for m in models])
            accs.append(best_acc)
            
        plt.plot(rounds, accs, marker='o', label=cipher.upper())
        
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing (50%)')
    plt.axhline(y=1.0, color='g', linestyle='--', label='Perfect Reconstruction (100%)')
    plt.title('Best Model Accuracy vs. Number of Rounds Across 12 Ciphers')
    plt.xlabel('Number of Rounds (r)')
    plt.ylabel('Bitwise Prediction Accuracy')
    plt.xticks(range(1, 6))
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/plots/all_ciphers_accuracy.png', dpi=300)
    plt.close()
    
    # 2. Diffusion / Hamming Distance plot
    plt.figure(figsize=(14, 8))
    for cipher in ciphers:
        rounds = sorted([int(r) for r in data[cipher].keys()])
        h_dists = []
        # Normalizing Hamming distance by block size to make comparative plotting feasible
        for r in rounds:
            block_size = data[cipher][str(r)][models[0]]['block_size']
            # Best model (closest hamming distance to 0, though when random it approaches 0.5 * block_size)
            best_h_dist = min([data[cipher][str(r)][m]['avg_hamming'] for m in models])
            h_dists.append(best_h_dist / block_size) # Normalize to % of block
            
        plt.plot(rounds, h_dists, marker='^', label=cipher.upper())
        
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance (50% bits differ)')
    plt.title('Normalized Hamming Distance vs. Number of Rounds Across 12 Ciphers')
    plt.xlabel('Number of Rounds (r)')
    plt.ylabel('Normalized Hamming Distance (Distance / Block Size)')
    plt.xticks(range(1, 6))
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/plots/all_ciphers_hamming_distance.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_comparative_plots()
    print("Aggregate plots saved to results/plots/")
