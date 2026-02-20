import os
import torch
import json
import time
from src.dataset import get_dataloaders
from src.models import LogisticRegressionModel, MLPModel, CNNModel
from src.train import train_model
from src.eval import evaluate_model

# 12 Ciphers
CIPHERS = [
    'simon', 'speck', 'present', 'prince', 'tea', 'xtea', 
    'rc5', 'katan', 'rectangle', 'chacha20', 'salsa20', 'trivium'
]

def run_experiments(max_rounds=5, num_samples=100000, epochs=20):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    # Store results per cipher, then per round, then per model
    all_results = {}
    
    for cipher_name in CIPHERS:
        print(f"\n{'#'*60}\nEvaluating Cipher: {cipher_name.upper()}\n{'#'*60}")
        cipher_results = {}
        all_results[cipher_name] = cipher_results
        
        for r in range(1, max_rounds + 1):
            print(f"\n{'='*40}\nData & Train for {cipher_name.upper()} r={r}\n{'='*40}")
            
            train_loader, val_loader, test_loader, block_size = get_dataloaders(
                num_samples=num_samples, 
                rounds=r, 
                cipher_name=cipher_name, 
                batch_size=1024
            )
            
            # Dynamically set input/output neuron sizes to the block size of the cipher
            models = {
                'LogisticRegression': LogisticRegressionModel(input_size=block_size, output_size=block_size),
                'MLP': MLPModel(input_size=block_size, output_size=block_size),
                'CNN': CNNModel(input_size=block_size, output_size=block_size)
            }
            
            cipher_results[r] = {}
            
            for model_name, model in models.items():
                save_path = f"results/models/{cipher_name}_{model_name}_r{r}.pt"
                
                start_time = time.time()
                history = train_model(
                    model=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    epochs=epochs, 
                    device=device,
                    save_path=save_path
                )
                
                if os.path.exists(save_path):
                    model.load_state_dict(torch.load(save_path, weights_only=True))
                    
                test_acc, avg_hamming = evaluate_model(model, test_loader, device=device)
                elapsed = time.time() - start_time
                print(f"[{model_name} r={r}] Test Acc: {test_acc:.4f} | Avg Hamming: {avg_hamming:.2f} | Time: {elapsed:.1f}s")
                
                cipher_results[r][model_name] = {
                    'test_acc': test_acc,
                    'avg_hamming': avg_hamming,
                    'history': history,
                    'time_s': elapsed,
                    'block_size': block_size
                }
                
                # Save continually to avoid data loss on crash
                with open('results/metrics.json', 'w') as f:
                    json.dump(all_results, f, indent=4)
                    
    print("\nAll 12 Cipher experiments completed and metrics saved! Results are in results/metrics.json")

if __name__ == '__main__':
    # Using 10,000 samples for the 12-cipher loop so it finishes in a reasonable time.
    run_experiments(max_rounds=5, num_samples=10000, epochs=15)
