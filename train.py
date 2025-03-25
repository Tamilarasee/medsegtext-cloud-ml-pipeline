import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-8)

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / (union + 1e-8)

def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        dice, iou = 0, 0
        for batch_idx, (images, masks, texts) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images, texts)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            dice += dice_score(preds, masks).item()
            iou += iou_score(preds, masks).item()
            
            # # Print current batch number
            if batch_idx % 500 == 0:  # Print every 35 batches for clarity
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, masks, texts in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images, texts)
                loss = criterion(preds, masks)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Dice: {dice/len(train_loader):.4f}, IoU: {iou/len(train_loader):.4f}, \n"
              f"Val Loss: {val_loss/len(val_loader):.4f}, Dice: {dice/len(val_loader):.4f}, IoU: {iou/len(val_loader):.4f}")
    
    torch.save(model.state_dict(), "unet_model.pth")
    print("Model saved successfully!")


