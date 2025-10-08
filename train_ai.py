import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import time
import gc

# Increase PIL limit for large images
Image.MAX_IMAGE_PIXELS = 200000000

# ============================================
# 1. Lightweight Dataset (loads images one by one)
# ============================================
class LightweightDataset(Dataset):
    """
    Memory-efficient dataset
    Folder structure:
    data/
      train/
        high_quality/  # label 1
        low_quality/   # label 0
    """
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        print(f"Loading dataset from: {root_dir}")
        
        # Load high quality images
        high_q_dir = self.root_dir / 'high_quality'
        if high_q_dir.exists():
            for img_path in high_q_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(1)
            for img_path in high_q_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(1)
        
        # Load low quality images
        low_q_dir = self.root_dir / 'low_quality'
        if low_q_dir.exists():
            for img_path in low_q_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(0)
            for img_path in low_q_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(0)
        
        # Limit samples if specified (for testing)
        if max_samples and len(self.images) > max_samples:
            self.images = self.images[:max_samples]
            self.labels = self.labels[:max_samples]
        
        print(f"Found {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Resize if too large
            max_size = 800
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            label = self.labels[idx]
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data on error
            dummy_img = torch.zeros(3, 128, 128)
            return dummy_img, self.labels[idx]


# ============================================
# 2. Lightweight Model (MobileNetV2 - fast and small)
# ============================================
class LightweightClassifier(nn.Module):
    """Small and fast model for weak GPUs/CPUs"""
    def __init__(self, num_classes=2):
        super(LightweightClassifier, self).__init__()
        
        # MobileNetV2 - very light model (3.5M parameters vs ResNet50 25M)
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Replace final layer
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================
# 3. Simple transforms (less augmentation = faster)
# ============================================
def get_simple_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((128, 128)),  # Smaller size = faster
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# ============================================
# 4. Memory-efficient training with garbage collection
# ============================================
def train_lightweight(model, train_loader, val_loader, epochs=10, lr=0.001, save_every=2):
    # Use CPU if CUDA not available or causes issues
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable memory optimization for CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n{'='*50}")
        print(f"Epoch [{epoch+1}/{epochs}] - Training...")
        print(f"{'='*50}")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_acc = 100 * train_correct / train_total
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")
            
            # Clear memory
            del images, labels, outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ========== VALIDATION ==========
        print(f"\nValidating...")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Clear memory
                del images, labels, outputs, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # ========== EPOCH SUMMARY ==========
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*50}")
        print(f"Epoch [{epoch+1}/{epochs}] Summary:")
        print(f"{'='*50}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_lightweight.pth')
            print(f"*** Saved new best model! (Acc: {val_acc:.2f}%) ***")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Force garbage collection
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*50}")
    
    return model


# ============================================
# 5. Prediction function
# ============================================
def predict_image(model, image_path, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence


# ============================================
# 6. Main execution
# ============================================
if __name__ == '__main__':
    print("="*50)
    print("LIGHTWEIGHT IMAGE CLASSIFIER")
    print("Optimized for low-end PCs")
    print("="*50)
    
    # ========== CONFIGURATION ==========
    BATCH_SIZE = 8          # Small batch = less memory (try 4 if still crashes)
    EPOCHS = 10             # Number of training epochs
    LEARNING_RATE = 0.001   # Learning rate
    NUM_WORKERS = 0         # 0 = no multiprocessing (more stable)
    
    print(f"\nConfiguration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ========== LOAD DATA ==========
    print(f"\nLoading datasets...")
    
    train_dataset = LightweightDataset(
        root_dir='data/train',
        transform=get_simple_transforms(train=True)
    )
    
    val_dataset = LightweightDataset(
        root_dir='data/val',
        transform=get_simple_transforms(train=False)
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No training images found!")
        print("Please check your data/train/ folder structure")
        exit(1)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False  # Disable for stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    
    # ========== CREATE MODEL ==========
    print(f"\nCreating model...")
    model = LightweightClassifier(num_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # ========== TRAIN ==========
    print(f"\nStarting training...")
    print(f"This may take a while on CPU...")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        trained_model = train_lightweight(
            model, 
            train_loader, 
            val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            save_every=2
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        torch.save(model.state_dict(), 'interrupted_model.pth')
        print("Model saved as 'interrupted_model.pth'")
    
    # ========== TEST PREDICTION ==========
    print(f"\n{'='*50}")
    print("Testing prediction on sample image...")
    print(f"{'='*50}")
    
    # You can test with any image
    test_image = 'test_image.jpg'
    if Path(test_image).exists():
        predicted_class, confidence = predict_image(
            trained_model,
            test_image,
            get_simple_transforms(train=False)
        )
        
        class_names = ['Low Quality', 'High Quality']
        print(f"\nImage: {test_image}")
        print(f"Predicted: {class_names[predicted_class]}")
        print(f"Confidence: {confidence*100:.2f}%")
    else:
        print(f"\nNo test image found at '{test_image}'")
    
    print(f"\n{'='*50}")
    print("Done!")
    print(f"{'='*50}")