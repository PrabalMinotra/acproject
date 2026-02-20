import json
import matplotlib.pyplot as plt
import os

def generate_plots(metrics_file='results/metrics.json'):
    with open(metrics_file, 'r') as f:
        results = json.load(f)
        
    rounds = sorted([int(r) for r in results.keys()])
    models = list(results[str(rounds[0])].keys())
    
    os.makedirs('results/plots', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Plot 1: Accuracy vs Rounds
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for model in models:
        accs = [results[str(r)][model]['test_acc'] for r in rounds]
        plt.plot(rounds, accs, marker='o', label=model)
    
    # 50% random guessing baseline
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing (50%)')
    
    plt.title('Prediction Accuracy vs. Number of Rounds for SIMON32/64')
    plt.xlabel('Number of Rounds (r)')
    plt.ylabel('Bitwise Accuracy')
    plt.xticks(rounds)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/plots/accuracy_vs_rounds.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 2: Hamming Distance vs Rounds
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for model in models:
        hams = [results[str(r)][model]['avg_hamming'] for r in rounds]
        plt.plot(rounds, hams, marker='v', label=model)
        
    # Random guessing expected Hamming Distance = 16 for 32-bit block
    plt.axhline(y=16.0, color='r', linestyle='--', label='Random Guessing (16.0)')
    
    plt.title('Average Hamming Distance vs. Number of Rounds for SIMON32/64')
    plt.xlabel('Number of Rounds (r)')
    plt.ylabel('Hamming Distance')
    plt.xticks(rounds)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/plots/hamming_dist_vs_rounds.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 3: Training Loss vs Epochs
    # -------------------------------------------------------------------------
    for model in models:
        plt.figure(figsize=(10, 6))
        for r in rounds:
            history = results[str(r)][model]['history']['train_loss']
            epochs = list(range(1, len(history) + 1))
            plt.plot(epochs, history, label=f'r={r}')
            
        plt.title(f'{model} Training Loss vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/plots/loss_vs_epochs_{model}.png', dpi=300)
        plt.close()
        
    print("Plots successfully generated in results/plots/!")

if __name__ == '__main__':
    generate_plots()
