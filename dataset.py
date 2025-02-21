import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder


class DataframeImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, target_transform=None):
        self.img_labels=df[['image_id', 'dx']]
        self.img_dir = img_dir
        self.transform = transform

        if isinstance(target_transform, LabelEncoder):
            if not hasattr(target_transform, 'classes_'):
                target_transform.fit(df['dx'])
            self.target_transform = lambda x: target_transform.transform([x])[0]
        elif callable(target_transform):
            self.target_transform = target_transform
        else:
            self.target_transform = lambda x: x
            


    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) + '.jpg'
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.tensor(label, dtype=torch.long)
