import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Coordinate Attention (CA) Module
class CoordAtt(nn.Module):
    def __init__(self, inp_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        reduced_channels = max(8, inp_channels // reduction)
        self.conv1 = nn.Conv2d(inp_channels, reduced_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(reduced_channels, inp_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(reduced_channels, inp_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = x * a_h * a_w
        return out

# Attention-guided residual fusion function
def ca_guided_residual_fusion(F_c_MSFA, F_t, ca_module):
    F_c_att = ca_module(F_c_MSFA)
    F_t_att = ca_module(F_t)
    F_fusion = F_c_att + F_t_att + F_c_MSFA * F_t_att
    return F_fusion

# Load CSV file and convert to tensor with shape (B, C, H, W)
def csv_to_tensor(csv_path, batch_size, channels, height, width):
    df = pd.read_csv(csv_path)
    if df.isnull().values.any():
        print(f"Warning: NaN values found in {csv_path}. Filling with 0.")
        df = df.fillna(0)

    # Select only numeric columns (drop non-numeric like filenames)
    df_numeric = df.select_dtypes(include=[np.number])
    data = df_numeric.values.astype('float32')

    total_expected = batch_size * channels * height * width
    if data.size != total_expected:
        raise ValueError(f"CSV numeric data size ({data.size}) does not match expected size "
                         f"({total_expected}) for shape ({batch_size},{channels},{height},{width})")

    tensor = torch.tensor(data, dtype=torch.float32)
    tensor = tensor.view(batch_size, channels, height, width)
    return tensor

if __name__ == "__main__":
    # Set your CSV file paths here
    csv_cnn = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/msfa_features.csv"
    csv_trans = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/vit_dynamic_pruning_features.csv"

    batch_size = 150
    height = 1
    width = 1

    # Load CSVs once to detect numeric channels
    df_cnn = pd.read_csv(csv_cnn).fillna(0)
    df_cnn_num = df_cnn.select_dtypes(include=[np.number])
    channels_cnn = df_cnn_num.shape[1] // (height * width)

    df_trans = pd.read_csv(csv_trans).fillna(0)
    df_trans_num = df_trans.select_dtypes(include=[np.number])
    channels_trans = df_trans_num.shape[1] // (height * width)

    print(f"Detected numeric channels - CNN: {channels_cnn}, Transformer: {channels_trans}")

    if channels_cnn != channels_trans:
        raise ValueError(f"Channel mismatch between CNN features ({channels_cnn}) and Transformer features ({channels_trans}).")

    # Load tensors
    F_c_MSFA = csv_to_tensor(csv_cnn, batch_size, channels_cnn, height, width)
    F_t = csv_to_tensor(csv_trans, batch_size, channels_trans, height, width)

    # Initialize Coordinate Attention module
    ca_module = CoordAtt(inp_channels=channels_cnn, reduction=16)

    # Apply fusion
    F_fusion = ca_guided_residual_fusion(F_c_MSFA, F_t, ca_module)

    print("Fused feature map shape:", F_fusion.shape)
# Read the CSV files into DataFrames
df1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/msfa_features.csv')
df2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/vit_dynamic_pruning_features.csv')
# Check the shapes of both DataFrames
print("Shape of DataFrame 1:", df1.shape)
print("Shape of DataFrame 2:", df2.shape)
# Ensure both DataFrames have the same number of rows
# Truncate the larger DataFrame to match the row count of the smaller one
min_rows = min(df1.shape[0], df2.shape[0])
df1 = df1.iloc[:min_rows]
df2 = df2.iloc[:min_rows]
# Concatenate DataFrames by columns
df_concatenated = pd.concat([df1, df2], axis=1)

# Display the shape of the concatenated DataFrame
print("Shape after concatenation:", df_concatenated.shape)

# Optionally, save the concatenated DataFrame to a new CSV file
df_concatenated.to_csv('/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/concadenate fused_file1.csv', index=False)
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/concadenate fused_file1.csv')
df
