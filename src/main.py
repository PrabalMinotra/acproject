import os
import torch
import json
import time
from src.dataset import get_dataloaders
from src.models import LogisticRegressionModel, MLPModel, CNNModel
from src.train import train_model
from src.eval import evaluate_model

def run_experiments(max_rounds=5, num_samples=100000, epochs=20):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    results = {}
    
    for r in range(1, max_rounds + 1):
        print(f"\n{'='*40}\nGenerating data and training for r={r}\n{'='*40}")
        # Phase I & II: Generate dataset for round r
        train_loader, val_loader, test_loader = get_dataloaders(num_samples=num_samples, rounds=r, batch_size=1024)
        
        models = {
            'LogisticRegression': LogisticRegressionModel(),
            'MLP': MLPModel(),
            'CNN': CNNModel()
        }
        
        results[r] = {}
        
        for name, model in models.items():
            save_path = f"results/models/{name}_r{r}.pt"
            
            start_time = time.time()
            # Train the model
            history = train_model(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                epochs=epochs, 
                device=device,
                save_path=save_path
            )
            
            # Load best model for evaluation
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path, weights_only=True))
                
            # Evaluate the model
            test_acc, avg_hamming = evaluate_model(model, test_loader, device=device)
            elapsed = time.time() - start_time
            print(f"[{name} r={r}] Test Acc: {test_acc:.4f} | Avg Hamming Distance: {avg_hamming:.2f} | Time: {elapsed:.1f}s")
            
            results[r][name] = {
                'test_acc': test_acc,
                'avg_hamming': avg_hamming,
                'history': history,
                'time_s': elapsed
            }
            
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nAll experiments completed and metrics saved! Results are in results/metrics.json")

if __name__ == '__main__':
    # Using 100,000 samples and up to 5 rounds for reasonable timeframe. 
    # Can adjust epochs to trade off speed for potential minor accuracy bumps.
    run_experiments(max_rounds=5, num_samples=100000, epochs=20)
