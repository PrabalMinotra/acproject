import os
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=15,
    lr=0.001,
    device='cpu',
    save_path=None,
    log_path=None
):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"Training {model.__class__.__name__} for {epochs} epochs on {device}...")

    log_file = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, 'w', encoding='ascii')
        log_file.write('epoch,train_loss,val_loss,val_acc\n')
    
    def unpack_batch(batch):
        if len(batch) == 3:
            return batch
        pt, ct = batch
        mask = torch.ones_like(ct)
        return pt, ct, mask

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            pt, ct, mask = unpack_batch(batch)
            pt, ct, mask = pt.to(device), ct.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(pt)
            loss_raw = criterion(outputs, ct)
            mask_sum = mask.sum().clamp(min=1.0)
            loss = (loss_raw * mask).sum() / mask_sum
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pt.size(0)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        
        model.eval()
        val_loss = 0.0
        correct_bits = 0
        total_bits = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pt, ct, mask = unpack_batch(batch)
                pt, ct, mask = pt.to(device), ct.to(device), mask.to(device)
                outputs = model(pt)
                loss_raw = criterion(outputs, ct)
                mask_sum = mask.sum().clamp(min=1.0)
                loss = (loss_raw * mask).sum() / mask_sum
                val_loss += loss.item() * pt.size(0)
                
                preds = (outputs > 0).float()
                correct_bits += ((preds == ct).float() * mask).sum().item()
                total_bits += mask.sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct_bits / total_bits
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if log_file:
            log_file.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f},{val_acc:.6f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)

    if log_file:
        log_file.close()

    return history
