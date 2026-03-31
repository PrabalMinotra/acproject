import torch
import os
from torch.utils.data import Dataset
import numpy as np
import random


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
from src.ciphers.cham import ChamCipher
from src.ciphers.hight import HightCipher
from src.ciphers.lea import LeaCipher
from src.ciphers.simeck import SimeckCipher

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
    'trivium': TriviumCipher,
    'cham': ChamCipher,
    'hight': HightCipher,
    'lea': LeaCipher,
    'simeck': SimeckCipher
}

def int_to_bit_array(val, bits):
    
    
    bin_str = format(val, f'0{bits}b')
    return np.array([int(x) for x in bin_str], dtype=np.float32)

def _cache_path(cache_dir, cipher_name, rounds):
    if cipher_name == 'simon':
        filename = f'dataset_r{rounds}.pt'
    else:
        filename = f'dataset_{cipher_name}_r{rounds}.pt'
    return os.path.join(cache_dir, filename)

def _cache_matches(cached, num_samples, rounds, key, cipher_name):
    meta = cached.get('meta') if isinstance(cached, dict) else None
    if meta:
        return (
            meta.get('num_samples') == num_samples and
            meta.get('rounds') == rounds and
            meta.get('cipher_name') == cipher_name and
            meta.get('key') == int(key)
        )
    return False

class CachedCipherDataset(Dataset):
    def __init__(self, pts, cts, label_mask_tensor):
        self.pts = pts
        self.cts = cts
        self.label_mask_tensor = label_mask_tensor

    def __len__(self):
        return self.pts.size(0)

    def __getitem__(self, idx):
        return self.pts[idx], self.cts[idx], self.label_mask_tensor

class CipherDataset(Dataset):
    def __init__(self, num_samples, rounds, key, cipher_name):
        self.num_samples = num_samples
        self.rounds = rounds
        self.key = key
        
        self.cipher_class = CIPHER_MAP[cipher_name.lower()]
        self.cipher = self.cipher_class(self.key)
        self.block_size = self.cipher.block_size

        self.header_bits = int(getattr(self.cipher, 'header_bits', 0))
        self.output_bits = self.block_size - self.header_bits
        if self.output_bits <= 0:
            raise ValueError(f"Invalid output size for {cipher_name}: {self.output_bits}")

        self.label_mask = np.ones(self.block_size, dtype=np.float32)
        if self.header_bits > 0:
            self.label_mask[-self.header_bits:] = 0.0
        self.label_mask_tensor = torch.tensor(self.label_mask)
        
        
        seed = 42 + rounds + sum(ord(c) for c in cipher_name)
        rng = random.Random(seed)
        
        print(f"Generating CTs for {num_samples} samples with {rounds} rounds for {cipher_name} ({self.block_size}-bit)...")
        self.pts = []
        self.cts = []

        
        for _ in range(num_samples):
            val = rng.getrandbits(self.block_size)
            ct = self.cipher.encrypt(val, rounds=self.rounds)
            self.pts.append(val)
            self.cts.append(ct)
                
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        pt_bits = int_to_bit_array(self.pts[idx], bits=self.block_size)
        ct_bits = int_to_bit_array(self.cts[idx], bits=self.block_size)
        return torch.tensor(pt_bits), torch.tensor(ct_bits), self.label_mask_tensor

def get_dataloaders(
    num_samples=10000,
    rounds=1,
    key=0x1918111009080100,
    cipher_name='simon',
    batch_size=1024,
    cache_dir='data',
    use_cache=True,
    save_cache=True
):
    cipher_class = CIPHER_MAP[cipher_name.lower()]
    cipher = cipher_class(key)
    block_size = cipher.block_size
    header_bits = int(getattr(cipher, 'header_bits', 0))
    output_bits = block_size - header_bits

    label_mask = torch.ones(block_size, dtype=torch.float32)
    if header_bits > 0:
        label_mask[-header_bits:] = 0.0

    cache_path = _cache_path(cache_dir, cipher_name, rounds)
    if use_cache and os.path.exists(cache_path):
        cached = torch.load(cache_path)
        if _cache_matches(cached, num_samples, rounds, key, cipher_name):
            train_ds = CachedCipherDataset(cached['train']['pt'], cached['train']['ct'], label_mask)
            val_ds = CachedCipherDataset(cached['val']['pt'], cached['val']['ct'], label_mask)
            test_ds = CachedCipherDataset(cached['test']['pt'], cached['test']['ct'], label_mask)

            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader, test_loader, block_size, output_bits

    full_dataset = CipherDataset(num_samples, rounds, key, cipher_name)
    
    
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

    if save_cache:
        os.makedirs(cache_dir, exist_ok=True)

        def loader_to_tensor(loader):
            pts, cts = [], []
            for batch in loader:
                pt_batch, ct_batch = batch[0], batch[1]
                pts.append(pt_batch)
                cts.append(ct_batch)
            return torch.cat(pts), torch.cat(cts)

        train_pt, train_ct = loader_to_tensor(train_loader)
        val_pt, val_ct = loader_to_tensor(val_loader)
        test_pt, test_ct = loader_to_tensor(test_loader)

        dataset_dict = {
            'meta': {
                'num_samples': num_samples,
                'rounds': rounds,
                'cipher_name': cipher_name,
                'key': int(key),
                'block_size': block_size,
                'output_bits': output_bits
            },
            'train': {'pt': train_pt, 'ct': train_ct},
            'val': {'pt': val_pt, 'ct': val_ct},
            'test': {'pt': test_pt, 'ct': test_ct}
        }
        torch.save(dataset_dict, cache_path)

    return train_loader, val_loader, test_loader, full_dataset.block_size, full_dataset.output_bits
