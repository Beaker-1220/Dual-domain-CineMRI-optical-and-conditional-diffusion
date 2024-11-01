from transformer import Transformer
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch
class ppDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.train_dir = os.path.join(root_dir, 'train/input')
        self.train_dir_tar = os.path.join(root_dir, 'train/target')
        self.optical_dir = os.path.join(root_dir, 'feature_maps')
        # self.train_images = sorted(os.listdir(self.train_dir))
        # self.train_images_tar = sorted(os.listdir(self.train_dir_tar))
        # self.optical_images = sorted(os.listdir(self.optical_dir))
        # assert len(self.train_images) == len(self.target_images), "Train and Target folders must have the same number of images."
        self.train_images = []
        self.train_images_tar = []
        self.optical_images_0 = []
        self.optical_images_1 = []

        # 搜索文件夹并构建数据集
        for filename in os.listdir(self.train_dir):
            base_name = os.path.splitext(filename)[0]  # 去掉扩展名
            # 构建需要查找的文件名
            file_0 = f"{base_name}_0.png"  # 假设文件扩展名是 .png
            file_1 = f"{base_name}_1.png"

            if file_0 in os.listdir(self.optical_dir) and file_1 in os.listdir(self.optical_dir):
                self.train_images.append(filename)  # 原始图像
                self.train_images_tar.append(filename)  # GT
                self.optical_images_0.append(file_0)  
                self.optical_images_1.append(file_1)# flow1 and flow0

        # assert len(self.train_images) == len(self.train_images_tar), "Train and Target folders must have the same number of images."


    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        # 获取训练图像和目标图像的路径
        train_img_path = os.path.join(self.train_dir, self.train_images[idx])
        train_img_tar_path = os.path.join(self.train_dir_tar, self.train_images_tar[idx])
        optical_img_0path = os.path.join(self.optical_dir, self.optical_images_0[idx])
        optical_img_1path = os.path.join(self.optical_dir, self.optical_images_1[idx])
        
        # 打开图像并转换为RGB格式（如果需要）
        train_img = Image.open(train_img_path).convert("RGB")
        train_img = train_img.resize((192, 448))
        train_data = np.array(train_img) / 255.0  # 归一化到 [0, 1]
        train_data = train_data.astype(np.float32)  # 转换为 float32
        train_img = torch.tensor(train_data).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        train_img_tar = Image.open(train_img_tar_path).convert("RGB")
        train_img_tar = train_img_tar.resize((192, 448))
        train_data_tar = np.array(train_img_tar) / 255.0  # 归一化到 [0, 1]
        train_data_tar = train_data_tar.astype(np.float32)  # 转换为 float32
        train_img_tar = torch.tensor(train_data_tar).permute(0, 1, 2)  # 改变形状为 [C, H, W]
        
        flow_0 = Image.open(optical_img_0path).convert("RGB")
        flow_0 = flow_0.resize((192, 448))
        flow_0 = np.array(flow_0) / 255.0  # 归一化到 [0, 1]
        flow_0 = flow_0.astype(np.float32)  # 转换为 float32
        flow_0 = torch.tensor(flow_0).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        flow_1 = Image.open(optical_img_1path).convert("RGB")
        flow_1 = flow_1.resize((192, 448))
        flow_1 = np.array(flow_1) / 255.0  # 归一化到 [0, 1]
        flow_1 = flow_1.astype(np.float32)  # 转换为 float32
        flow_1 = torch.tensor(flow_1).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        return train_img, train_img_tar, flow_0, flow_1
# 使用示例
input_dim = 64  # 输入维度
d_model = 256  # 特征维度
num_layers = 6  # 编码器层数
nhead = 4  # 注意力头数
dim_feedforward = 512  # 前馈网络维度
dropout = 0.1  # Dropout 概率
# 训练过程
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for train_img, train_img_tar, flow_0, flow_1 in dataloader:
            optimizer.zero_grad()

            # 假设输入的形状是 [seq_len, batch_size, input_dim]
            # 这里我们需要调整输入的维度
            train_img = train_img.permute(2, 0, 1)  # [C, H, W] -> [H, W, C]
            train_img_tar = train_img_tar.permute(2, 0, 1)

            # 前向传播
            outputs = model(train_img)
            loss = criterion(outputs, train_img_tar)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

transformer_model = Transformer(num_layers, d_model, nhead, dim_feedforward, input_dim, dropout)

criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(transformer_model.parameters(), lr=1e-4)
root_dir = "/root/autodl-tmp/.autodl/dataset1-70"
dataset = ppDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# 开始训练
train_model(transformer_model, dataloader, criterion, optimizer, num_epochs=10)