import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Dataset loader
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], os.path.basename(self.image_paths[idx]), self.classes[self.labels[idx]]

# MSFA model using EfficientNet-B0
class MSFA_EfficientNet(nn.Module):
    def __init__(self, resize_size=(64, 64)):
        super(MSFA_EfficientNet, self).__init__()
        self.resize_size = resize_size

        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        self.backbone.eval()

        # Corrected valid layer names
        self.layers_to_extract = [
            "features.2.0.block",  # Output: 24 channels
            "features.4.0.block",  # Output: 80 channels
            "features.6.1.block"   # Output: 192 channels
        ]

        self.feature_extractor = create_feature_extractor(
            self.backbone,
            return_nodes={layer: layer for layer in self.layers_to_extract}
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Match projection channels to extracted outputs
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(24, 128, kernel_size=1),
            nn.Conv2d(80, 128, kernel_size=1),
            nn.Conv2d(192, 128, kernel_size=1)
        ])

        self.global_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        feature_list = [features[layer] for layer in self.layers_to_extract]

        multi_scale_features = []
        for f, proj in zip(feature_list, self.proj_convs):
            resized = F.interpolate(f, size=self.resize_size, mode='bilinear', align_corners=False)
            projected = proj(resized)
            multi_scale_features.append(projected)

        # Attention
        attention_scores = [self.global_fc(F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)) for f in multi_scale_features]
        attention_scores = torch.stack(attention_scores, dim=1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        attention_weights = attention_weights.view(attention_weights.size(0), attention_weights.size(1), 1, 1, 1)
        stacked_features = torch.stack(multi_scale_features, dim=1)
        fused_feature = (attention_weights * stacked_features).sum(dim=1)
        fused_feature = fused_feature.mean(dim=[2, 3])  # Global average pooling
        return fused_feature

# Feature extraction and CSV saving
def extract_and_save_features(input_folder, output_csv, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolderDataset(input_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSFA_EfficientNet(resize_size=(64, 64)).to(device)
    model.eval()

    all_features = []
    with torch.no_grad():
        for images, labels, filenames, classnames in dataloader:
            images = images.to(device)
            features = model(images).cpu().numpy()
            for f, label, fname, cname in zip(features, labels, filenames, classnames):
                all_features.append([fname, cname] + f.tolist())

    feature_dim = len(all_features[0]) - 2
    columns = ['filename', 'class'] + [f'feature_{i}' for i in range(feature_dim)]
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv}")

# Main function to run
if __name__ == "__main__":
    input_folder = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/OutputImages"
    output_csv = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/msfa_features.csv"
    extract_and_save_features(input_folder, output_csv, batch_size=4)
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/msfa_features.csv')
df