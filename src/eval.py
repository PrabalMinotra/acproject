import torch

def evaluate_model(model, test_loader, device='cpu'):
    model = model.to(device)
    model.eval()
    
    correct_bits = 0
    total_bits = 0
    total_hamming_dist = 0.0
    
    with torch.no_grad():
        for pt, ct in test_loader:
            pt, ct = pt.to(device), ct.to(device)
            outputs = model(pt)
            preds = (outputs > 0).float()
            
            # Bitwise prediction accuracy
            correct_bits += (preds == ct).sum().item()
            total_bits += ct.numel()
            
            # Hamming distance per sample: sum of differing bits
            hamming_dists = (preds != ct).sum(dim=1).float()
            total_hamming_dist += hamming_dists.sum().item()
            
    num_samples = len(test_loader.dataset)
    acc = correct_bits / total_bits
    avg_hamming = total_hamming_dist / num_samples
    
    return acc, avg_hamming
