import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_msssim import ssim
from model_vgg16 import DepthEstimationUNet
from dataset_creation import trainLoader, valLoader, testLoader
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#depthLoss function, combines both l1 loss and ssim for more accurate loss
#ssim focuses on img structure similarity, provides a better output image
def depthLoss(pred, gnd):
    pred = torch.clamp(pred, 0, 1)
    gnd = torch.clamp(gnd, 0, 1)

    l1 = nn.L1Loss()(pred, gnd)
    ssimVal = ssim(pred, gnd, data_range=1.0, size_average=True)

    if torch.isnan(ssimVal):
        return l1
    
    return 0.82 * l1 + 0.18 * (1 - ssimVal)


def trainModel(trainLoader, valLoader, device, numEpochs, learningRate):
    print(f"using: {device}")

    model = DepthEstimationUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=learningRate,
                                 weight_decay=1e-4)

    #lr schduler for fine tuning, half lr if no improvment in 3 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    trainLosses, valLosses = [], []
    bestValLoss = float("inf")

    stopPatience, notImproved = 10, 0

    for epoch in range(numEpochs):
        #train
        model.train()
        trainLoss = 0
        for images, depths in tqdm(trainLoader, desc=f"epoch {epoch+1}/{numEpochs} (train)"):
            images, depths = images.to(device), depths.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = depthLoss(outputs, depths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            trainLoss += loss.item()

        avgTrainLoss = trainLoss / len(trainLoader)
        trainLosses.append(avgTrainLoss)

        #val
        model.eval()
        valLoss = 0
        with torch.no_grad():
            for images, depths in tqdm(valLoader, desc=f"epoch {epoch+1}/{numEpochs} (val)"):
                images, depths = images.to(device), depths.to(device)
                outputs = model(images)
                loss = depthLoss(outputs, depths)
                valLoss += loss.item()

        avgValLoss = valLoss / len(valLoader)
        valLosses.append(avgValLoss)
        scheduler.step(avgValLoss)

        print(f"epoch [{epoch+1}/{numEpochs}] - train Loss: {avgTrainLoss:.4f}, val Loss: {avgValLoss:.4f}")

        #early stopping
        if avgValLoss < bestValLoss:
            bestValLoss = avgValLoss
            torch.save(model.state_dict(), "./results/bestModelUnet.pth")
            print(f"best model saved (val Loss: {avgValLoss:.4f})")
            notImproved = 0
        else:
            notImproved += 1
            if notImproved >= stopPatience:
                print("early stopping triggered \n")
                break
    
    #saves diff curve chart for each model, starts at 1
    baseName = "lossCurve"
    type = ".png"
    count = 1
    chartPath = os.path.join("trainingCharts", f"{baseName}{count}{type}")

    while os.path.exists(chartPath):
        count += 1
        chartPath = os.path.join("trainingCharts", f"{baseName}{count}{type}")

    #plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(trainLosses) + 1), trainLosses, label="Train Loss")
    plt.plot(range(1, len(valLosses) + 1), valLosses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(chartPath, dpi=200)
    plt.close()

    print(f"loss chart saved to: {chartPath}")

    history = {"trainLosses": trainLosses, "valLosses": valLosses}
    return model, history


#main
if __name__ == "__main__":
    print(" Starting training...")
    result = trainModel(trainLoader, valLoader, device, numEpochs=40, learningRate=2e-4)

    model, history = result
    print("TRAINING DONE")
    print(f"final training loss: {history['trainLosses'][-1]:.4f}")
    print(f"final validation loss: {history['valLosses'][-1]:.4f}")

