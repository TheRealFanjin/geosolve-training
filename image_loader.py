import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, root, transform):
        self.data = []  # stores (list of image paths, class index)
        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root, cls)
            for folder in sorted(os.listdir(cls_dir)):
                folder_path = os.path.join(cls_dir, folder)
                images = [os.path.join(folder_path, filename) for filename in sorted(os.listdir(folder_path))]
                self.data.append((images, self.class_to_idx[cls]))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_paths, label = self.data[idx]
        img_transformed = []  # (C, H, W)

        for image in img_paths:
            image = Image.open(image).convert('RGB')
            img_transformed.append(self.transform(image))

        img_transformed = torch.stack(img_transformed)  # (n_images, H, W)
        return img_transformed, label
