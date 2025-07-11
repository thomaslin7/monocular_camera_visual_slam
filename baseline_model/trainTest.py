#!/usr/bin/env python3
"""
Training script for depth estimation using local stairs dataset
Uses the baseline VGG16 model architecture with local RGB-D data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
import glob
from PIL import Image
import random

# Import model architecture and training function from baseline.py
from baseline import Decoder, fullBaseline, encoder, train_model

# ===============================
# 📊 DATASET CLASS
# ===============================

class StairsDepthDataset(Dataset):
    def __init__(self, sequences, base_path, transform=None, target_transform=None):
        """
        Args:
            sequences: List of sequence names (e.g., ['seq-01', 'seq-02'])
            base_path: Path to datasets/stairs directory
            transform: Transform to apply to RGB images
            target_transform: Transform to apply to depth maps
        """
        self.base_path = base_path
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        
        # Collect all RGB and depth file pairs
        for seq in sequences:
            seq_path = os.path.join(base_path, seq)
            if os.path.exists(seq_path):
                # Get all color files
                color_files = glob.glob(os.path.join(seq_path, "*.color.png"))
                
                for color_file in color_files:
                    # Get corresponding depth file
                    base_name = os.path.basename(color_file).replace('.color.png', '')
                    depth_file = os.path.join(seq_path, f"{base_name}.depth.png")
                    
                    if os.path.exists(depth_file):
                        self.samples.append((color_file, depth_file))
        
        print(f"Found {len(self.samples)} RGB-D pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        color_path, depth_path = self.samples[idx]
        
        # Load RGB image
        image = Image.open(color_path).convert('RGB')
        
        # Load depth image (single channel)
        depth = Image.open(depth_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            depth = self.target_transform(depth)
        
        return image, depth

# ===============================
# 🔧 DATA PREPARATION
# ===============================

def create_datasets(base_path="../datasets/stairs"):
    """Create train, validation, and test datasets based on predefined splits"""
    
    # Read predefined splits
    train_split_file = os.path.join(base_path, "TrainSplit.txt")
    test_split_file = os.path.join(base_path, "TestSplit.txt")
    
    # Read train sequences
    with open(train_split_file, 'r') as f:
        train_sequences = [line.strip() for line in f if line.strip()]
    
    # Read test sequences  
    with open(test_split_file, 'r') as f:
        test_sequences = [line.strip() for line in f if line.strip()]
    
    # Convert to seq-XX format
    train_seqs = [f"seq-0{seq[-1]}" for seq in train_sequences]  # sequence2 -> seq-02
    test_seqs = [f"seq-0{seq[-1]}" for seq in test_sequences]    # sequence1 -> seq-01
    
    # Create validation split from training sequences (take one sequence for val)
    val_seqs = [train_seqs.pop()]  # Remove last training sequence for validation
    
    print(f"Training sequences: {train_seqs}")
    print(f"Validation sequences: {val_seqs}")
    print(f"Test sequences: {test_seqs}")
    
    # Define transforms
    # RGB image transform (VGG16 normalization)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Depth transform (convert to float and normalize)
    depth_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),  # Convert to float32
        transforms.Lambda(lambda x: x / 1000.0),  # Normalize depth values (assuming mm to meters)
    ])
    
    # Create datasets
    train_dataset = StairsDepthDataset(train_seqs, base_path, image_transform, depth_transform)
    val_dataset = StairsDepthDataset(val_seqs, base_path, image_transform, depth_transform)
    test_dataset = StairsDepthDataset(test_seqs, base_path, image_transform, depth_transform)
    
    return train_dataset, val_dataset, test_dataset

# Training function is imported from baseline.py

# ===============================
# 🧪 TESTING FUNCTION
# ===============================

def test_model(model, test_loader, device):
    """Test the trained model and calculate metrics"""
    
    model.eval()
    test_results = []
    
    print("Testing model...")
    
    with torch.no_grad():
        for batch_idx, (images, true_depths) in enumerate(test_loader):
            images = images.to(device)
            true_depths = true_depths.to(device)
            
            # Get predictions
            predicted_depths = model(images)
            
            # Move to CPU for metrics calculation
            true_depths_cpu = true_depths.cpu()
            predicted_depths_cpu = predicted_depths.cpu()
            
            # Calculate metrics for each image in the batch
            for i in range(images.shape[0]):
                true_flat = true_depths_cpu[i].flatten().numpy()
                pred_flat = predicted_depths_cpu[i].flatten().numpy()
                
                mse = mean_squared_error(true_flat, pred_flat)
                mae = mean_absolute_error(true_flat, pred_flat)
                rmse = np.sqrt(mse)
                
                # Calculate relative error
                rel_error = np.mean(np.abs(true_flat - pred_flat) / (true_flat + 1e-6))
                
                test_results.append({
                    'batch': batch_idx,
                    'sample': i,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'relative_error': rel_error
                })
                
                print(f"Sample {len(test_results)}: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, RelErr={rel_error:.4f}")
    
    # Calculate averages
    avg_mse = np.mean([r['mse'] for r in test_results])
    avg_mae = np.mean([r['mae'] for r in test_results])
    avg_rmse = np.mean([r['rmse'] for r in test_results])
    avg_rel_error = np.mean([r['relative_error'] for r in test_results])
    
    print("\n" + "="*30)
    print("📊 AVERAGE TEST METRICS")
    print("="*30)
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average Relative Error: {avg_rel_error:.4f}")
    
    return test_results, {
        'mse': avg_mse,
        'mae': avg_mae,
        'rmse': avg_rmse,
        'relative_error': avg_rel_error
    }

# ===============================
# 📊 VISUALIZATION FUNCTION
# ===============================

def visualize_results(model, test_loader, device, save_path="test_results"):
    """Create visualizations of model predictions"""
    
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get first batch
        images, true_depths = next(iter(test_loader))
        images = images.to(device)
        predicted_depths = model(images).cpu()
        
        # Plot first 3 samples
        num_samples = min(3, images.shape[0])
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Depth Estimation Results', fontsize=16)
        
        for i in range(num_samples):
            # Original image (denormalize)
            img = images[i].cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            axes[i, 0].imshow(img.permute(1, 2, 0))
            axes[i, 0].set_title(f'Input Image {i+1}')
            axes[i, 0].axis('off')
            
            # True depth
            true_depth = true_depths[i, 0].cpu().numpy()
            im1 = axes[i, 1].imshow(true_depth, cmap='plasma')
            axes[i, 1].set_title(f'Ground Truth Depth {i+1}')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Predicted depth
            pred_depth = predicted_depths[i, 0].numpy()
            im2 = axes[i, 2].imshow(pred_depth, cmap='plasma')
            axes[i, 2].set_title(f'Predicted Depth {i+1}')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(save_path, 'depth_predictions.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {viz_path}")
        
        # Show plot
        plt.show()

# ===============================
# 🚀 MAIN EXECUTION
# ===============================

def main():
    """Main training and testing pipeline"""
    
    print("🚀 Starting Local Dataset Depth Estimation Training")
    print("="*60)
    
    # Configuration
    BATCH_SIZE = 4  # Smaller batch size for local dataset
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n📊 Creating datasets...")
    try:
        train_dataset, val_dataset, test_dataset = create_datasets()
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"✅ Data loaders created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return
    
    # Train model
    print("\n🏋️ Training model...")
    try:
        output_log, trained_model = train_model(train_loader, val_loader, NUM_EPOCHS)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Save model locally
    print("\n💾 Saving model...")
    try:
        os.makedirs("saved_models", exist_ok=True)
        
        model_save_path = "saved_models/stairs_depth_model.pth"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'output_log': output_log,
            'config': {
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'learning_rate': LEARNING_RATE,
                'architecture': 'VGG16-Encoder + Custom-Decoder',
                'dataset': 'Stairs Local Dataset'
            }
        }, model_save_path)
        
        print(f"✅ Model saved to {model_save_path}")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    # Test model
    print("\n🧪 Testing model...")
    try:
        test_results, avg_metrics = test_model(trained_model, test_loader, device)
        
        # Save test results
        os.makedirs("test_results", exist_ok=True)
        
        results_path = "test_results/test_metrics.json"
        with open(results_path, 'w') as f:
            json.dump({
                'individual_results': test_results,
                'averages': avg_metrics,
                'config': {
                    'num_test_samples': len(test_results),
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS
                }
            }, f, indent=2)
        
        print(f"✅ Test results saved to {results_path}")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    try:
        visualize_results(trained_model, test_loader, device)
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
    
    # Plot training curves from output_log
    print("\n📈 Plotting training curves...")
    try:
        if output_log:
            # Extract validation losses from output_log
            val_losses = [entry[2] for entry in output_log]  # Assuming val_loss is at index 2
            
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Loss During Training')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('test_results/training_curves.png', dpi=150, bbox_inches='tight')
            print("✅ Training curves saved to test_results/training_curves.png")
            plt.show()
        else:
            print("⚠️ No training log available for plotting")
        
    except Exception as e:
        print(f"❌ Error plotting training curves: {e}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING AND TESTING COMPLETE!")
    print("="*60)
    print("📁 Model saved in: saved_models/")
    print("📊 Results saved in: test_results/")
    print("🔄 Model file: stairs_depth_model.pth")

if __name__ == "__main__":
    main() 