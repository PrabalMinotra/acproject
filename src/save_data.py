import argparse
import os
from dataset import get_dataloaders

def save_datasets(max_rounds=20, num_samples=10000, cipher_name='simon', force_refresh=False):
    os.makedirs('data', exist_ok=True)
    
    for r in range(1, max_rounds + 1):
        print(f"Generating and saving data for {cipher_name} r={r}...")
        
        get_dataloaders(
            num_samples=num_samples,
            rounds=r,
            cipher_name=cipher_name,
            batch_size=1024,
            use_cache=not force_refresh,
            save_cache=True
        )
        save_name = f'data/dataset_r{r}.pt' if cipher_name == 'simon' else f'data/dataset_{cipher_name}_r{r}.pt'
        print(f"Saved {save_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save cached datasets for reduced-round ciphers.')
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--cipher', type=str, default='simon')
    parser.add_argument('--force-refresh', action='store_true')
    args = parser.parse_args()

    save_datasets(
        max_rounds=args.max_rounds,
        num_samples=args.num_samples,
        cipher_name=args.cipher,
        force_refresh=args.force_refresh
    )