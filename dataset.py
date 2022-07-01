from torch.utils.data import Dataset 
import os 
from PIL import Image
import numpy as np 

class AbstractArtDataset(Dataset):
    def __init__(self, rootdir, transforms=None):
        self.rootdir = rootdir
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.rootdir))

    def __getitem__(self, index):
        img = np.array(Image.open(os.path.join(self.rootdir, os.listdir(self.rootdir)[index%len(os.listdir(self.rootdir))])).convert('RGB'))
        if self.transforms:
            img = self.transforms(img)
        return img 
