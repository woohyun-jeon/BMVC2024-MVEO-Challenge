import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, deeplabv3_resnet101, DeepLabV3_ResNet101_Weights


# define dataset
class LULCDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        label = np.array(label)

        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(label).long()

        return image, label


# ===== Label Smoothing =====
# 2024.10.16 1st rank with the use of DeepLab-v3 & ResNet-101
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.classes = classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))


def label_smoothing_training(data_loader, num_classes, num_epochs=500, learning_rate=1e-4):
    print("[Start] Label Smoothing")
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    # model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model = model.cuda()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(data_loader, desc=f"Label Smoothing Epoch {epoch+1}/{num_epochs}"):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader)}")

    model.eval()
    smooth_scores = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Label Smoothing Evaluation"):
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)['out']
            preds = F.softmax(outputs, dim=1)

            with torch.no_grad():
                true_dist = torch.zeros_like(preds)
                true_dist.fill_(0.1 / (num_classes - 1))
                true_dist.scatter_(1, labels.unsqueeze(1), 0.9)

            smooth_diff = torch.abs(preds - true_dist).mean(dim=(1,2,3)).cpu().numpy()
            smooth_scores.extend(smooth_diff)

    print("[Complete] Label Smoothing")

    return smooth_scores


def save_label_smoothing_results(data_loader, dataset, num_classes, num_epochs, out_path):
    smooth_scores = label_smoothing_training(data_loader, num_classes, num_epochs=num_epochs)
    sorted_indices = np.argsort(smooth_scores)
    sorted_filenames = [dataset.image_files[i] for i in sorted_indices]

    output = pd.DataFrame({
        "id": range(len(sorted_filenames)),
        "imageid": sorted_filenames
    })
    output.to_csv(os.path.join(out_path, 'rank_label_smoothing.csv'), index=False)


def main(image_path, label_path, out_path, transform, num_classes, num_epochs=300):
    dataset = LULCDataset(image_path, label_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    save_label_smoothing_results(data_loader, dataset, num_classes, num_epochs=num_epochs, out_path=out_path)


# execute code
if __name__ == '__main__':
    image_dir = r"C:\pr03_lulc\images"
    label_dir = r"C:\pr03_lulc\labels"
    out_dir = r"C:\pr03_lulc\output"
    os.makedirs(out_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    num_classes = 2
    num_epochs = 300

    main(image_dir, label_dir, out_dir, transform, num_classes, num_epochs)