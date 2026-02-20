import torch
from torch.utils.data import Dataset
import numpy as np

# A mapping of all supported ciphers for easy instantiation
from src.ciphers.simon import SimonCipher
from src.ciphers.speck import SpeckCipher
from src.ciphers.present import PresentCipher
from src.ciphers.prince import PrinceCipher
from src.ciphers.tea import TEACipher
from src.ciphers.xtea import XTEACipher
from src.ciphers.rc5 import RC5Cipher
from src.ciphers.katan import KatanCipher
from src.ciphers.rectangle import RectangleCipher
from src.ciphers.chacha20 import ChaCha20Cipher
from src.ciphers.salsa20 import Salsa20Cipher
from src.ciphers.trivium import TriviumCipher

CIPHER_MAP = {
    'simon': SimonCipher,
    'speck': SpeckCipher,
    'present': PresentCipher,
    'prince': PrinceCipher,
    'tea': TEACipher,
    'xtea': XTEACipher,
    'rc5': RC5Cipher,
    'katan': KatanCipher,
    'rectangle': RectangleCipher,
    'chacha20': ChaCha20Cipher,
    'salsa20': Salsa20Cipher,
    'trivium': TriviumCipher
}

def int_to_bit_array(val, bits):
    """Convert an integer to a numpy array of bits (0s and 1s)."""
    # handle extremely large integers using python format
    bin_str = format(val, f'0{bits}b')
    return np.array([int(x) for x in bin_str], dtype=np.float32)

class CipherDataset(Dataset):
    def __init__(self, num_samples, rounds, key, cipher_name):
        self.num_samples = num_samples
        self.rounds = rounds
        self.key = key
        
        self.cipher_class = CIPHER_MAP[cipher_name.lower()]
        self.cipher = self.cipher_class(self.key)
        self.block_size = self.cipher.block_size
        
        # Use a fixed seed for reproducibility dependent on bounds and cipher
        np.random.seed(42 + rounds + sum(ord(c) for c in cipher_name))
        
        print(f"Generating CTs for {num_samples} samples with {rounds} rounds for {cipher_name} ({self.block_size}-bit)...")
        self.pts = []
        self.cts = []
        
        # NumPy random ints are limited to 64 bit natively, so for 512 bit (ChaCha/Salsa) we use Python's random or generate chunks
        if self.block_size <= 64:
            pts_raw = np.random.randint(0, 2**min(63, self.block_size), size=num_samples, dtype=np.uint64)
            for pt in pts_raw:
                val = int(pt)
                ct = self.cipher.encrypt(val, rounds=self.rounds)
                self.pts.append(val)
                self.cts.append(ct)
        else:
            # Huge blocks (like Stream Cipher XOR sizes of 512 bits)
            import random
            random.seed(42 + rounds)
            for _ in range(num_samples):
                val = random.getrandbits(self.block_size)
                ct = self.cipher.encrypt(val, rounds=self.rounds)
                self.pts.append(val)
                self.cts.append(ct)
                
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        pt_bits = int_to_bit_array(self.pts[idx], bits=self.block_size)
        ct_bits = int_to_bit_array(self.cts[idx], bits=self.block_size)
        return torch.tensor(pt_bits), torch.tensor(ct_bits)

def get_dataloaders(num_samples=100000, rounds=1, key=0x1918111009080100, cipher_name='simon', batch_size=1024):
    full_dataset = CipherDataset(num_samples, rounds, key, cipher_name)
    
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
    
    return train_loader, val_loader, test_loader, full_dataset.block_size
