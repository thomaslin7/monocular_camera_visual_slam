import os
import glob
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


#sequence = ai_xxx_xxx 
class HypersimDepthDataset(Dataset):
    def __init__(self, sequences, basePath, transform=None, depthTransform=None):
        self.basePath = basePath
        self.transform = transform
        self.depthTransform = depthTransform
        #list of rgb, depth sample pairs
        self.samples = []

        #get rgb and depth file pairs
        for seq in sequences:
            seqPath = os.path.join(basePath, seq)
            if os.path.exists(seqPath):
                #only gets the file paths for scene cam 00 in that set for the sake of training time
                rgbDir = os.path.join(seqPath, "images", "scene_cam_00_final_preview")
                depthDir = os.path.join(seqPath, "images", "scene_cam_00_geometry_hdf5")

                if os.path.exists(rgbDir) and os.path.exists(depthDir):
                    #gets all rgb files ending with tonemap.jpg 
                    rgbFiles = glob.glob(os.path.join(rgbDir, "*.tonemap.jpg"))

                    for rgbFile in rgbFiles:
                        #remove tonemap.jpg and replace with depth_meters.hdf5 so we can find matching depth file as well
                        baseName = os.path.basename(rgbFile).replace('.tonemap.jpg', '')
                        depthFile = os.path.join(depthDir, f"{baseName}.depth_meters.hdf5")

                        #sanity check
                        if os.path.exists(depthFile):
                            self.samples.append((rgbFile, depthFile))

        print(f"Found {len(self.samples)} rgb-depth pairs \n")

    def __len__(self):
        return len(self.samples)


    #returns both the actual rgb image and the depth map converted into an image
    def __getitem__(self, idx):
        rgbSampPath, depthSampPath = self.samples[idx]

        #get rgb image
        image = Image.open(rgbSampPath).convert('RGB')

        #get depthmap
        with h5py.File(depthSampPath, 'r') as f:
            dataset_names = list(f.keys())
            if 'depth' in dataset_names:
                depthData = f['depth'][:]
            elif 'depth_meters' in dataset_names:
                depthData = f['depth_meters'][:]
            else:
                depthData = f[dataset_names[0]][:]

        depth = Image.fromarray(depthData.astype(np.float32))

        #apply transforms
        if self.transform:
            image = self.transform(image)

        if self.depthTransform:
            depth = self.depthTransform(depth)

        return image, depth
