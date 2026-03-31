import torch

def evaluate_model(model, test_loader, device='cpu'):
    model = model.to(device)
    model.eval()
    
    correct_bits = 0
    total_bits = 0
    total_hamming_dist = 0.0
    
    def unpack_batch(batch):
        if len(batch) == 3:
            return batch
        pt, ct = batch
        mask = torch.ones_like(ct)
        return pt, ct, mask

    with torch.no_grad():
        for batch in test_loader:
            pt, ct, mask = unpack_batch(batch)
            pt, ct, mask = pt.to(device), ct.to(device), mask.to(device)
            outputs = model(pt)
            preds = (outputs > 0).float()
            
            
            correct_bits += ((preds == ct).float() * mask).sum().item()
            total_bits += mask.sum().item()
            
            
            diff = (preds != ct).float() * mask
            hamming_dists = diff.sum(dim=1).float()
            total_hamming_dist += hamming_dists.sum().item()
            
    num_samples = len(test_loader.dataset)
    acc = correct_bits / total_bits
    avg_hamming = total_hamming_dist / num_samples
    
    return acc, avg_hamming
