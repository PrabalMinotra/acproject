import os
import torch
from src.dataset import get_dataloaders
from tqdm import tqdm

def save_datasets(max_rounds=5, num_samples=100000):
    os.makedirs('data', exist_ok=True)
    
    for r in range(1, max_rounds + 1):
        print(f"Generating and saving data for r={r}...")
        
        train_loader, val_loader, test_loader = get_dataloaders(num_samples=num_samples, rounds=r, batch_size=1024)
        
        def loader_to_tensor(loader):
            pts, cts = [], []
            for pt_batch, ct_batch in loader:
                pts.append(pt_batch)
                cts.append(ct_batch)
            return torch.cat(pts), torch.cat(cts)
        
        train_pt, train_ct = loader_to_tensor(train_loader)
        val_pt, val_ct = loader_to_tensor(val_loader)
        test_pt, test_ct = loader_to_tensor(test_loader)
        
        dataset_dict = {
            'train': {'pt': train_pt, 'ct': train_ct},
            'val': {'pt': val_pt, 'ct': val_ct},
            'test': {'pt': test_pt, 'ct': test_ct}
        }
        
        torch.save(dataset_dict, f'data/dataset_r{r}.pt')
        print(f"Saved data/dataset_r{r}.pt")

if __name__ == '__main__':
    save_datasets()
