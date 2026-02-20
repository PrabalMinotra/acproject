import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu', save_path=None):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"Training {model.__class__.__name__} for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for pt, ct in train_loader:
            pt, ct = pt.to(device), ct.to(device)
            optimizer.zero_grad()
            outputs = model(pt)
            loss = criterion(outputs, ct)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pt.size(0)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_bits = 0
        total_bits = 0
        
        with torch.no_grad():
            for pt, ct in val_loader:
                pt, ct = pt.to(device), ct.to(device)
                outputs = model(pt)
                loss = criterion(outputs, ct)
                val_loss += loss.item() * pt.size(0)
                
                preds = (outputs > 0).float()
                correct_bits += (preds == ct).sum().item()
                total_bits += ct.numel()

        val_loss /= len(val_loader.dataset)
        val_acc = correct_bits / total_bits
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
                
    return history
