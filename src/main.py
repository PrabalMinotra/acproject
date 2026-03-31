import argparse
import os
import torch
import json
import time
from src.dataset import get_dataloaders
from src.models import LogisticRegressionModel, MLPModel, CNNModel
from src.train import train_model
from src.eval import evaluate_model


CIPHERS = [
    'simon', 'speck', 'present', 'prince', 'tea', 'xtea', 
    'rc5', 'katan', 'rectangle', 'chacha20', 'salsa20', 'trivium'
]

def run_experiments(
    max_rounds=20,
    num_samples=10000,
    epochs=15,
    acc_threshold=0.55,
    hamming_threshold=0.45,
    early_stop_rounds=2
):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    
    all_results = {}
    
    for cipher_name in CIPHERS:
        print(f"\n{'#'*60}\nEvaluating Cipher: {cipher_name.upper()}\n{'#'*60}")
        cipher_results = {}
        all_results[cipher_name] = cipher_results
        
        fail_streak = 0
        for r in range(1, max_rounds + 1):
            print(f"\n{'='*40}\nData & Train for {cipher_name.upper()} r={r}\n{'='*40}")
            
            train_loader, val_loader, test_loader, block_size, output_bits = get_dataloaders(
                num_samples=num_samples, 
                rounds=r, 
                cipher_name=cipher_name, 
                batch_size=1024
            )
            
            
            models = {
                'LogisticRegression': LogisticRegressionModel(input_size=block_size, output_size=block_size),
                'MLP': MLPModel(input_size=block_size, output_size=block_size),
                'CNN': CNNModel(input_size=block_size, output_size=block_size)
            }
            
            cipher_results[r] = {}
            
            for model_name, model in models.items():
                save_path = f"results/models/{cipher_name}_{model_name}_r{r}.pt"
                log_path = f"results/logs/{cipher_name}_{model_name}_r{r}.csv"
                
                start_time = time.time()
                history = train_model(
                    model=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    epochs=epochs, 
                    device=device,
                    save_path=save_path,
                    log_path=log_path
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
                    'block_size': block_size,
                    'output_bits': output_bits
                }
                
                
                with open('results/metrics.json', 'w') as f:
                    json.dump(all_results, f, indent=4)

            meaningful = False
            for model_name in models.keys():
                metrics = cipher_results[r][model_name]
                norm_hamming = metrics['avg_hamming'] / output_bits
                if metrics['test_acc'] >= acc_threshold and norm_hamming <= hamming_threshold:
                    meaningful = True
                    break

            if meaningful:
                fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak >= early_stop_rounds:
                    print(f"Stopping {cipher_name.upper()} after r={r} (no meaningful learning).")
                    break
                    
    print("\nAll 12 Cipher experiments completed and metrics saved! Results are in results/metrics.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run reduced-round cipher experiments.')
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--acc-threshold', type=float, default=0.55)
    parser.add_argument('--hamming-threshold', type=float, default=0.45)
    parser.add_argument('--early-stop-rounds', type=int, default=2)
    args = parser.parse_args()

    run_experiments(
        max_rounds=args.max_rounds,
        num_samples=args.num_samples,
        epochs=args.epochs,
        acc_threshold=args.acc_threshold,
        hamming_threshold=args.hamming_threshold,
        early_stop_rounds=args.early_stop_rounds
    )