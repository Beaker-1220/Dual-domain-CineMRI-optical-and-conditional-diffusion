from transformer import Transformer, TransformerEncoder
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tqdm import tqdm
import torchmetrics
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from torchvision.utils import save_image
def save_denoised_images(images, output_dir, epoch, batch_idx):
    """
    保存去噪后的图像到指定目录。
    
    :param images: 一个 [batch, channels, height, width] 形状的 tensor，包含去噪后的图像。
    :param output_dir: 字符串，表示要保存图像的输出目录。
    :param epoch: 当前的 epoch 数，用于生成文件名。
    :param batch_idx: 当前 batch 的索引，用于生成文件名。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历 batch 中的每个图像并保存
    for idx, img in enumerate(images):
        # 将图像数据范围调整到 [0, 1]（若已归一化可跳过此步）
        img = (img - img.min()) / (img.max() - img.min())
        
        # 设置文件路径
        file_path = os.path.join(output_dir, f"denoised_image_epoch{epoch}_batch{batch_idx}_img{idx}.png")
        
        # 保存图像
        save_image(img, file_path)
        print(f"Saved {file_path}")
def compute_metrics(original_tensor, dct_tensor):

    dct_tensor = dct_tensor.clamp(0, 1)

    # 将张量转换为 NumPy 数组并转换为 [H, W, C] 格式
    original_images = original_tensor.cpu().detach().numpy()  # 原始图像
    reconstructed_images = dct_tensor.cpu().detach().numpy() # 重建图像
    # 计算 PSNR 和 SSIM
    psnr_values = []
    ssim_values = []
    mse_values = []
    for orig, recon in zip(original_images, reconstructed_images):
        # 计算 PSNR
        psnr_value = torchmetrics.functional.psnr(torch.tensor(recon), torch.tensor(orig), data_range=1.0)
        psnr_values.append(psnr_value.item())

        # 计算 SSIM
        ssim_value = compare_ssim(orig, recon,channel_axis=0, data_range=1.0)
        ssim_values.append(ssim_value)
        
        mse_value = compare_mse(orig, recon)
        mse_values.append(mse_value)

    return psnr_values, ssim_values, mse_values

class pDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.train_dir = os.path.join(root_dir, 'val/input')
        self.train_dir_tar = os.path.join(root_dir, 'val/target')
        self.train_images = sorted(os.listdir(self.train_dir))
        self.train_images_tar = sorted(os.listdir(self.train_dir_tar))
        self.optical_images = sorted(os.listdir(self.optical_dir))
        assert len(self.train_images) == len(self.train_images_tar ), "Train and Target folders must have the same number of images."


        # assert len(self.train_images) == len(self.train_images_tar), "Train and Target folders must have the same number of images."
    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        # 获取训练图像和目标图像的路径
        train_img_path = os.path.join(self.train_dir, self.train_images[idx])
        train_img_tar_path = os.path.join(self.train_dir_tar, self.train_images_tar[idx])
        
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
    

        return train_img, train_img_tar

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
        train_img = train_img.resize((64, 64))
        train_data = np.array(train_img) / 255.0  # 归一化到 [0, 1]
        train_data = train_data.astype(np.float32)  # 转换为 float32
        train_img = torch.tensor(train_data).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        train_img_tar = Image.open(train_img_tar_path).convert("RGB")
        train_img_tar = train_img_tar.resize((64, 64))
        train_data_tar = np.array(train_img_tar) / 255.0  # 归一化到 [0, 1]
        train_data_tar = train_data_tar.astype(np.float32)  # 转换为 float32
        train_img_tar = torch.tensor(train_data_tar).permute(0, 1, 2)  # 改变形状为 [C, H, W]
        
        flow_0 = Image.open(optical_img_0path).convert("RGB")
        flow_0 = flow_0.resize((64, 64))
        flow_0 = np.array(flow_0) / 255.0  # 归一化到 [0, 1]
        flow_0 = flow_0.astype(np.float32)  # 转换为 float32
        flow_0 = torch.tensor(flow_0).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        flow_1 = Image.open(optical_img_1path).convert("RGB")
        flow_1 = flow_1.resize((64, 64))
        flow_1 = np.array(flow_1) / 255.0  # 归一化到 [0, 1]
        flow_1 = flow_1.astype(np.float32)  # 转换为 float32
        flow_1 = torch.tensor(flow_1).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        return train_img, train_img_tar, flow_0, flow_1
# 使用示例
input_dim = 3  # 输入维度
d_model = 256  # 特征维度
num_layers = 6  # 编码器层数
nhead = 4  # 注意力头数
dim_feedforward = 512  # 前馈网络维度
dropout = 0.1  # Dropout 概率
# 训练过程
def train_model(model, dataloader, criterion, optimizer, num_epochs, test_ratio=0.15, batch_size=16):

    total_size = len(dataloader.dataset)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size
    print('Total number of images: {}'.format(total_size), '\nTotal number of training images: {}'.format(train_size), '\nTotal number of testing images: {}'.format(test_size))

    train_dataset, test_dataset = random_split(dataloader.dataset, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last= True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last= True)
    model.load_state_dict(torch.load('/root/autodl-tmp/.autodl/dataset1-70/transformer/model_weights_epoch_5.pth'))
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        ssim = 0
        psnr = 0
        for k, (train_img, train_img_tar, flow_0, flow_1) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            # 假设输入的形状是 [seq_len, batch_size, input_dim]
            # 这里我们需要调整输入的维度
            train_img = train_img.to(device)  #  [N, H, W, C]
            train_img_tar = train_img_tar.to(device)
            flow_0 = flow_0.to(device)
            flow_1 = flow_1.to(device)

            # 前向传播
            outputs = model(train_img, flow_0, flow_1)
            outputs = outputs.reshape(train_img.shape)
            outputs = outputs.permute(0, 3, 1, 2)
            train_img_tar = train_img_tar.permute(0, 3, 1, 2)
            save_denoised_images(outputs, "/root/autodl-tmp/.autodl/outputs", 1, 0)
            loss = criterion(outputs, train_img_tar)
            psnr_values, ssim_values, mse_values = compute_metrics(train_img_tar, outputs)

            print('mse:', sum(mse_values)/len(mse_values), 'psnr:', sum(psnr_values)/len(psnr_values),'ssim:', sum(ssim_values)/len(ssim_values),'loss:', loss.item())

            # 反向传播
            loss.backward()
            optimizer.step()
            ssim += sum(ssim_values)/len(ssim_values)
            psnr += sum(psnr_values)/len(psnr_values)
            running_loss += loss.item()

        epoch_loss = running_loss
        ssim = ssim/len(dataloader)
        psnr = psnr/len(dataloader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_loss:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}')
        
                # 验证步骤
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for k, (val_img, val_img_tar, flow_0, flow_1) in enumerate(tqdm(test_loader)):
                # 假设输入的形状是 [seq_len, batch_size, input_dim]
                # 这里我们需要调整输入的维度
                val_img = val_img.to(device)  # [N, C, H, W] -> [N, H, W, C]
                val_img_tar = val_img_tar.to(device)
                flow_0 = flow_0.to(device)
                flow_1 = flow_1.to(device)

                # 前向传播
                val_outputs = model(val_img, flow_0, flow_1)
                val_outputs = val_outputs.reshape(val_img.shape)
                val_outputs = val_outputs.permute(0, 3, 1, 2)
                train_img_tar = train_img_tar.permute(0, 3, 1, 2)
                val_loss = criterion(val_outputs, val_img_tar)

                val_psnr_values, val_ssim_values, val_mse_values = compute_metrics(val_img_tar, val_outputs)

                print('val_mse:', sum(val_mse_values)/len(val_mse_values), 'val_psnr:', sum(val_psnr_values)/len(val_psnr_values),'val_ssim:', sum(val_ssim_values)/len(val_ssim_values),'val_loss:', val_loss.item())

                val_running_loss += val_loss.item()
                ssim += sum(val_ssim_values)/len(val_ssim_values)
                psnr += sum(val_psnr_values)/len(val_psnr_values)

        ssim = ssim/len(dataloader)
        psnr = psnr/len(dataloader)
        val_epoch_loss = val_running_loss

        # 打印结果
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}')

        # 将指标保存到txt文件
        with open("/root/autodl-tmp/.autodl/dataset1-70/transformer/validation_metrics.txt", "a") as file:
            file.write(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}\n')

          # 保存模型权重
        weight_path = f"/root/autodl-tmp/.autodl/dataset1-70/transformer/model_weights_epoch_{epoch+6}.pth"
        torch.save(model.state_dict(), weight_path)
        print(f"Model weights saved at {weight_path}")
        
        # 恢复模型到训练模式
        model.train()
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
transformer_model = Transformer(num_layers, d_model, nhead, dim_feedforward, input_dim, dropout)
transformer_model = transformer_model.to(device)
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(transformer_model.parameters(), lr=1e-4)
root_dir = "/root/autodl-tmp/.autodl/dataset1-70"
dataset = ppDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last= True)


# 开始训练
train_model(transformer_model, dataloader, criterion, optimizer, num_epochs=50)