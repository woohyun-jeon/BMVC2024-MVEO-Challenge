## Achieved 7th place among 31 teams with a 5-fold ensemble using DeepLab v3 and ResNet-50 (score: 0.02278).

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import KFold

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


# define label smoothing loss
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


# define train model
def train_and_evaluate_model(model_type, dataset, num_classes, n_splits=5, num_epochs=300, learning_rate=1e-4):
    print(f"\nTraining with {model_type}")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_smooth_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")

        # set model
        if model_type == 'resnet50':
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        elif model_type == 'resnet101':
            model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        else:
            raise ValueError(f"Model type {model_type} not supported")

        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model = model.cuda()

        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)

        # set dataset
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)

        # execute training
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        full_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        model.eval()
        fold_scores = []
        with torch.no_grad():
            for images, labels in tqdm(full_loader, desc=f"Evaluating Fold {fold + 1}"):
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)['out']
                preds = F.softmax(outputs, dim=1)

                true_dist = torch.zeros_like(preds)
                true_dist.fill_(0.1 / (num_classes - 1))
                true_dist.scatter_(1, labels.unsqueeze(1), 0.9)

                smooth_diff = torch.abs(preds - true_dist).mean(dim=(1, 2, 3)).cpu().numpy()
                fold_scores.extend(smooth_diff)

        # save result from current fold
        sorted_indices = np.argsort(fold_scores)
        sorted_filenames = [dataset.image_files[i] for i in sorted_indices]

        fold_results.append({
            'scores': fold_scores,
            'sorted_files': sorted_filenames
        })
        all_smooth_scores.append(fold_scores)

        fold_submission = pd.DataFrame({
            "id": range(len(sorted_filenames)),
            "imageid": sorted_filenames
        })
        fold_submission.to_csv(f'submission_{model_type}_fold_{fold + 1}.csv', index=False)

    # save ensemble result
    ensemble_scores = np.mean(all_smooth_scores, axis=0)
    sorted_indices = np.argsort(ensemble_scores)
    sorted_filenames = [dataset.image_files[i] for i in sorted_indices]

    ensemble_submission = pd.DataFrame({
        "id": range(len(sorted_filenames)),
        "imageid": sorted_filenames
    })
    ensemble_submission.to_csv(f'submission_{model_type}_ensemble.csv', index=False)

    return fold_results, ensemble_submission


def main(image_path, label_path, transform, num_classes, num_epochs=300):
    dataset = LULCDataset(image_path, label_path, transform=transform)

    # get result from ResNet-50
    resnet50_results, resnet50_ensemble = train_and_evaluate_model(
        'resnet50', dataset, num_classes, n_splits=5, num_epochs=num_epochs
    )

    # get result from ResNet-101
    resnet101_results, resnet101_ensemble = train_and_evaluate_model(
        'resnet101', dataset, num_classes, n_splits=5, num_epochs=num_epochs
    )

    # get ensemble from ResNet-50 & ResNet-101
    final_ensemble = pd.merge(
        resnet50_ensemble.rename(columns={'imageid': 'r50_imageid'}),
        resnet101_ensemble.rename(columns={'imageid': 'r101_imageid'}),
        on='id'
    )

    final_filenames = []
    for _, row in final_ensemble.iterrows():
        r50_idx = list(dataset.image_files).index(row['r50_imageid'])
        r101_idx = list(dataset.image_files).index(row['r101_imageid'])
        final_filenames.append((row['r50_imageid'], (r50_idx + r101_idx) / 2))

    final_filenames.sort(key=lambda x: x[1])
    final_submission = pd.DataFrame({
        "id": range(len(final_filenames)),
        "imageid": [f[0] for f in final_filenames]
    })
    final_submission.to_csv('submission_final_ensemble.csv', index=False)


if __name__ == '__main__':
    image_dir = r"C:\pr03_lulc\images"
    label_dir = r"C:\pr03_lulc\labels"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    num_classes = 2
    num_epochs = 300

    main(image_dir, label_dir, transform, num_classes, num_epochs)