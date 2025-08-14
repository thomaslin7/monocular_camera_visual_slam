import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model_vgg16 import DepthEstimationUNet
from tqdm import tqdm
from training import depthLoss, ssim
from dataset_creation import testLoader
from PIL import Image
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the saved model
model = DepthEstimationUNet()
model.load_state_dict(torch.load("./results/bestModelUnet4.pth", map_location=device))
model.to(device)


#evaluate model performance
def evaluateModel(model, testLoader, criterion):
    model.eval()
    totalLoss = 0.0
    with torch.no_grad():
        for images, depths in tqdm(testLoader, desc="Testing"):
            images = images.to(device)
            depths = depths.to(device)

            outputs = model(images)
            loss = criterion(outputs, depths)
            totalLoss += loss.item() * images.size(0)

    avgLoss = totalLoss / len(testLoader.dataset)
    print(f"📊 Average Test Loss: {avgLoss:.4f}")
    return avgLoss


# Visualize predictions
def visualizePredictions(model, testLoader, device, numSamples=3):
    model.eval()
    for idx, (images, trueDepths) in enumerate(testLoader):
        #batch 40
        if idx == 40:
            break

    # images, trueDepths = next(iter(testLoader))
    images = images.to(device)

    with torch.no_grad():
        predictedDepths = model(images)

    numSamples = min(numSamples, images.shape[0])
    fig, axes = plt.subplots(numSamples, 3, figsize=(15, 5 * numSamples))

    if numSamples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Model Predictions vs Ground Truth', fontsize=16)

    for i in range(numSamples):
        # Original image (denormalize)
        img = images[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)

        axes[i, 0].imshow(img.permute(1, 2, 0))
        axes[i, 0].set_title(f'Input Image {i+1}')
        axes[i, 0].axis('off')

        # Ground truth
        trueDepth = trueDepths[i, 0].cpu().numpy()
        im1 = axes[i, 1].imshow(trueDepth, cmap='plasma')
        axes[i, 1].set_title(f'Ground Truth Depth {i+1}')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Prediction
        predDepth = predictedDepths[i, 0].cpu().numpy()
        im2 = axes[i, 2].imshow(predDepth, cmap='plasma')
        axes[i, 2].set_title(f'Predicted Depth {i+1}')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    savePath = os.path.join("results", 'predictionsLarger.png')
    plt.savefig(savePath, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"📸 Predictions visualization saved to: {savePath}")


# visualizePredictions(model, testLoader, device, numSamples=3)
# testLoss = evaluateModel(model, testLoader, depthLoss)



def visualizeSingleImage(model, imgPath, device):
    model.eval()

    # 1️⃣ Load image
    img = Image.open(imgPath).convert("RGB")

    # Resize while keeping aspect ratio
    img.thumbnail((224, 224), Image.Resampling.LANCZOS)





    # 2️⃣ Apply same preprocessing as training
    transform = T.Compose([
        T.Resize((224, 224)),  # change to your input size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    imgTensor = transform(img).unsqueeze(0).to(device)  # add batch dim

    # 3️⃣ Run prediction
    with torch.no_grad():
        predDepth = model(imgTensor)

    # 4️⃣ Plot results (no ground truth)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    denorm = imgTensor[0].cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    denorm = torch.clamp(denorm, 0, 1)
    axes[0].imshow(denorm.permute(1, 2, 0))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Predicted depth
    predDepth = predDepth[0, 0].cpu().numpy()
    im = axes[1].imshow(predDepth, cmap='plasma')
    axes[1].set_title("Predicted Depth")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    savePath = os.path.join("results", 'pred_single_image.png')
    plt.savefig(savePath, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"📸 Prediction saved to: {savePath}")

# Example usage
visualizeSingleImage(model, "./image.png", device)