import os
import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset_handler import HypersimDepthDataset
from tqdm import tqdm

#helper functions
def scaleDepth(depth):
    depth = depth.float()
    # max interest of depth is 10m, scales to values b/w 0-1
    return torch.clamp(depth / 10.0, 0, 1)

def cleanDataset(dataset):
    cleanIndices = []
    print(f"cleaning dataset: {len(dataset)} samples")

    checkIndices = list(range(len(dataset)))

    for i in tqdm(checkIndices, desc="cleaning samples"):
            image, depth = dataset[i]
            #if depth values are bad: nan/<0 or inf/>100, do not add to clean indicies
            if torch.isnan(depth).any() or torch.isinf(depth).any():
                continue
            if depth.min() < 0 or depth.max() > 100:
                continue

            cleanIndices.append(i)

    print(f"kept {len(cleanIndices)} out of {len(checkIndices)} samples")
    return cleanIndices



#dataset creation
def createDataset(basePath):
    print("creating all datasets")
    # get all sequences in hypersim
    allSequences = [d for d in os.listdir(basePath) if os.path.isdir(os.path.join(basePath, d)) and d.startswith('ai_')]
    print(f"found {len(allSequences)} sequences in dataset")

    # split sequences
    trainSize = int(0.7 * len(allSequences))
    valSize = int(0.15 * len(allSequences))

    # assign splits
    trainSequences = allSequences[:trainSize]
    valSequences = allSequences[trainSize:trainSize + valSize]
    testSequences = allSequences[trainSize + valSize:]

    print(f"training seqs: {len(trainSequences)}")
    print(f"val seqs: {len(valSequences)}")
    print(f"test seqs: {len(testSequences)}")

    # transforms
    imageTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    depthTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        scaleDepth
    ])

    # datasets
    trainDataset = HypersimDepthDataset(trainSequences, basePath, imageTransform, depthTransform)
    valDataset = HypersimDepthDataset(valSequences, basePath, imageTransform, depthTransform)
    testDataset = HypersimDepthDataset(testSequences, basePath, imageTransform, depthTransform)

    print(f"{len(trainDataset)} train pairs")
    print(f"{len(valDataset)} val pairs")
    print(f"{len(testDataset)} test pairs")
    print("Datasets created")

    return trainDataset, valDataset, testDataset

#main
print("starting dataset creation...")
trainDataset, valDataset, testDataset = createDataset("../primary_model/dataset/HyperSim/all")

BATCH_SIZE = 16
trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)
testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"data loaders created, bs: {BATCH_SIZE}")

saveDir = os.path.abspath("./depth_cache")
indicesPath  = os.path.join(saveDir, "clean_indices.json")

#cleaning, only run initially to get clean indices, if file exists, skip
if not os.path.isfile(indicesPath):
    print("cleaning training")
    trainCleanIndices = cleanDataset(trainDataset)
    print("cleaning val")
    valCleanIndices = cleanDataset(valDataset)
    print("cleaning test")
    testCleanIndices = cleanDataset(testDataset)

    #save the clean indices
    with open(indicesPath, "w") as f:
        json.dump({
            "train": trainCleanIndices,
            "val": valCleanIndices,
            "test": testCleanIndices
        }, f)
    print(f"clean indices saved to {indicesPath}")

else:
    with open(indicesPath, "r") as f:
        indices = json.load(f)
    print(f"loaded clean indices from {indicesPath}")

#build new clean datasets
trainDatasetClean = Subset(trainDataset, indices["train"])
valDatasetClean = Subset(valDataset, indices["val"])
testDatasetClean = Subset(testDataset, indices["test"])

trainLoader = DataLoader(trainDatasetClean, batch_size=BATCH_SIZE, shuffle=True)
valLoader = DataLoader(valDatasetClean, batch_size=BATCH_SIZE, shuffle=False)
testLoader = DataLoader(testDatasetClean, batch_size=BATCH_SIZE, shuffle=False)

print("cleaned datasets created from saved indices")
print(f"train samples: {len(trainDatasetClean)}")
print(f"val samples: {len(valDatasetClean)}")
print(f"test samples: {len(testDatasetClean)}")
