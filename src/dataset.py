import torch
from torch.utils.data import Dataset
import numpy as np

def int_to_bit_array(val, bits=32):
    """Convert an integer to a numpy array of bits (0s and 1s)."""
    return np.array([int(x) for x in format(val, f'0{bits}b')], dtype=np.float32)

class SimonDataset(Dataset):
    def __init__(self, num_samples, rounds, key):
        from src.simon import SimonCipher
        self.num_samples = num_samples
        self.rounds = rounds
        self.key = key
        
        self.cipher = SimonCipher(self.key)
        
        # Use a fixed seed for reproducibility but make it dependent on rounds 
        # so each round count gets a fresh dataset as required.
        np.random.seed(42 + rounds)
        self.pts = np.random.randint(0, 2**32, size=num_samples, dtype=np.uint32)
        
        print(f"Generating CTs for {num_samples} samples with {rounds} rounds...")
        self.cts = np.array([self.cipher.encrypt(int(pt), rounds=self.rounds) for pt in self.pts], dtype=np.uint32)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        pt_bits = int_to_bit_array(self.pts[idx], bits=32)
        ct_bits = int_to_bit_array(self.cts[idx], bits=32)
        return torch.tensor(pt_bits), torch.tensor(ct_bits)

def get_dataloaders(num_samples=100000, rounds=1, key=0x1918111009080100, batch_size=1024):
    full_dataset = SimonDataset(num_samples, rounds, key)
    
    # Train / Val / Test split: 80% / 10% / 10%
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
