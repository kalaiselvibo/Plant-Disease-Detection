import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv

# -------------------------------
# Dataset to recursively load images with relative paths
# -------------------------------
class RecursiveImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.relative_paths = []

        for subdir, _, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(subdir, fname)
                    rel_path = os.path.relpath(full_path, root_dir)  # relative path from root_dir
                    self.image_paths.append(full_path)
                    self.relative_paths.append(rel_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        rel_path = self.relative_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            image = Image.new('RGB', (128, 128), (0, 0, 0))  # black image fallback
        if self.transform:
            image = self.transform(image)
        return image, rel_path

# -------------------------------
# Model Components
# -------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_probs = attn_scores.softmax(dim=-1)

        attn_output = attn_probs @ v  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.fc_out(attn_output)
        return out, attn_probs

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, attn_probs = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_probs

class ViTWithDynamicTokenPruning(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_size=16, num_layers=4, num_heads=8, prune_thresholds=None):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None  # Will initialize after input size known
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        if prune_thresholds is None:
            prune_thresholds = [0.02, 0.03, 0.05, 0.1]
        self.prune_thresholds = prune_thresholds
        self.embed_dim = embed_dim

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N_patches, embed_dim)

        if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1] + 1:
            # Initialize position embedding (1 + N_patches, embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, x.shape[1] + 1, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N_patches, embed_dim)
        x = x + self.pos_embed

        for i, layer in enumerate(self.layers):
            x, attn_probs = layer(x)
            # Dynamic pruning except last layer
            if i < self.num_layers - 1:
                # Remove cls token for pruning
                cls_token = x[:, :1, :]
                tokens = x[:, 1:, :]
                attn_to_cls = attn_probs[:, :, 1:, 0]  # attention weights to cls token (B, num_heads, tokens)
                attn_mean = attn_to_cls.mean(dim=1)  # average over heads (B, tokens)
                thresh = self.prune_thresholds[i]
                keep_mask = attn_mean > thresh  # (B, tokens)

                # Prevent pruning all tokens by ensuring at least one token kept per sample
                keep_mask_sum = keep_mask.sum(dim=1)
                for b in range(B):
                    if keep_mask_sum[b] == 0:
                        # keep token with max attention
                        max_idx = attn_mean[b].argmax()
                        keep_mask[b, max_idx] = True

                # Use keep_mask to select tokens for each sample in batch
                new_tokens = []
                for b in range(B):
                    new_tokens.append(tokens[b, keep_mask[b]])
                # Pad tokens to max kept length in batch for batch tensor
                max_keep = max([nt.shape[0] for nt in new_tokens])
                padded_tokens = []
                for nt in new_tokens:
                    if nt.shape[0] < max_keep:
                        pad_size = max_keep - nt.shape[0]
                        pad = torch.zeros(pad_size, self.embed_dim, device=nt.device)
                        nt = torch.cat([nt, pad], dim=0)
                    padded_tokens.append(nt)
                tokens = torch.stack(padded_tokens, dim=0)
                # Update x with cls token + pruned tokens
                x = torch.cat([cls_token, tokens], dim=1)
                # Update pos_embed accordingly
                # Here, we keep cls pos_embed and repeat pos_embed tokens for max_keep tokens (approximate)
                cls_pos = self.pos_embed[:, 0:1, :]
                # Just repeat the first N tokens pos embedding for max_keep tokens
                patch_pos = self.pos_embed[:, 1:1+max_keep, :]
                x = x + torch.cat([cls_pos.repeat(B,1,1), patch_pos.repeat(B,1,1)], dim=1)

        return x  # (B, tokens_after_pruning, embed_dim)

# -------------------------------
# Main processing script
# -------------------------------
def main():
    image_folder = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/OutputImages"  # main input folder
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = RecursiveImageFolderDataset(image_folder, transform=transform)
    print(f"Total images found recursively: {len(dataset)}")
    print(f"Sample relative paths: {dataset.relative_paths[:5]}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    in_channels = 3
    patch_size = 16
    embed_dim = 128
    num_layers = 4
    num_heads = 8
    prune_thresholds = [0.02, 0.03, 0.05, 0.1]

    model = ViTWithDynamicTokenPruning(in_channels, embed_dim, patch_size, num_layers, num_heads, prune_thresholds)
    model = model.to(device)
    model.eval()

    all_features = []
    all_relative_paths = []

    with torch.no_grad():
        for imgs, rel_paths in dataloader:
            print(f"Processing batch of size: {imgs.size(0)}")
            imgs = imgs.to(device)
            out = model(imgs)
            print(f"Model output shape: {out.shape}")

            # Extract CLS token features
            cls_features = out[:, 0, :].cpu()  # (B, embed_dim)
            print(f"CLS token features shape: {cls_features.shape}")

            all_features.append(cls_features)
            all_relative_paths.extend(rel_paths)

    if len(all_features) == 0:
        print("No features extracted. Exiting.")
        return

    all_features = torch.cat(all_features, dim=0).numpy()

    csv_path = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/vit_dynamic_pruning_features.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["relative_path"] + [f"feat_{i}" for i in range(embed_dim)]
        writer.writerow(header)

        for rel_path, feat in zip(all_relative_paths, all_features):
            writer.writerow([rel_path] + feat.tolist())

    print(f"Features saved to {csv_path}")

if __name__ == "__main__":
    main()
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/vit_dynamic_pruning_features.csv')
df