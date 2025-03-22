import os
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from gaussian_diffusion import DiffusionType,CorrectorType,PredictorType,ModelMeanType,ModelVarType,LossType
from gaussian_diffusion import GaussianDiffusion
import torch
from scipy.fftpack import dct, idct
from transformer import Transformer,Transformer_dual
from utils.mri_data_utils.mask_util import create_mask_for_mask_type
from mcddpm_gaussian_diffusion import KspaceGaussianDiffusion
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torchmetrics
import torch as th
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from modelzoo import compute_metrics, transformer_dual, transformer, Unet, save_denoised_images, dct_to_idct, autoencoder, autoencoder_dct, autoencoder_optical
import wandb
import random
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from scipy.fftpack import dct

class fewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.train_dir = os.path.join(root_dir, 'train/input')
        self.train_dir_tar = os.path.join(root_dir, 'train/target')
        self.optical_dir = os.path.join(root_dir, 'feature_maps')
        
        # 初始化数据存储列表
        self.train_images = []
        self.train_images_tar = []
        self.optical_images_0 = []
        self.optical_images_1 = []

        # 遍历输入目录构建数据集
        for filename in os.listdir(self.train_dir):
            base_name = os.path.splitext(filename)[0]
            file_0 = f"{base_name}_0.png"
            file_1 = f"{base_name}_1.png"

            # 验证光流文件存在性
            if (file_0 in os.listdir(self.optical_dir)) and (file_1 in os.listdir(self.optical_dir)):
                self.train_images.append(filename)
                self.train_images_tar.append(filename)
                self.optical_images_0.append(file_0)
                self.optical_images_1.append(file_1)

        # 方法三：选取前1/100数据
        total_samples = len(self.train_images)
        n_samples = max(1, total_samples // 200)  # 确保至少1个样本
        self.train_images = self.train_images[:n_samples]
        self.train_images_tar = self.train_images_tar[:n_samples]
        self.optical_images_0 = self.optical_images_0[:n_samples]
        self.optical_images_1 = self.optical_images_1[:n_samples]

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        # 加载输入图像
        train_img_path = os.path.join(self.train_dir, self.train_images[idx])
        train_img = Image.open(train_img_path).convert("RGB")
        train_img = train_img.resize((256, 256))
        train_data = np.array(train_img) / 255.0  # [0,1]归一化
        train_data = train_data.astype(np.float32)
        train_tensor = torch.tensor(train_data).permute(0,1,2)  # [C,H,W]

        # 加载目标图像
        train_img_tar_path = os.path.join(self.train_dir_tar, self.train_images_tar[idx])
        train_img_tar = Image.open(train_img_tar_path).convert("RGB")
        train_img_tar = train_img_tar.resize((256, 256))
        train_data_tar = np.array(train_img_tar) / 255.0
        train_data_tar = train_data_tar.astype(np.float32)
        train_tensor_tar = torch.tensor(train_data_tar).permute(0,1,2)

        # 加载光流特征0
        flow_0_path = os.path.join(self.optical_dir, self.optical_images_0[idx])
        flow_0 = Image.open(flow_0_path).convert("RGB")
        flow_0 = flow_0.resize((256, 256))
        flow_0_data = np.array(flow_0) / 255.0
        flow_0_data = flow_0_data.astype(np.float32)
        flow_0_tensor = torch.tensor(flow_0_data).permute(0,1,2)

        # 加载光流特征1
        flow_1_path = os.path.join(self.optical_dir, self.optical_images_1[idx])
        flow_1 = Image.open(flow_1_path).convert("RGB")
        flow_1 = flow_1.resize((256, 256))
        flow_1_data = np.array(flow_1) / 255.0
        flow_1_data = flow_1_data.astype(np.float32)
        flow_1_tensor = torch.tensor(flow_1_data).permute(0,1,2)

        # DCT变换处理
        def process_dct(data):
            dct_channels = []
            for channel in range(3):  # RGB三通道
                dct_channel = dct(dct(data[:, :, channel].T, norm='ortho').T, norm='ortho')
                dct_channels.append(dct_channel)
            return np.stack(dct_channels, axis=-1)

        # 输入图像DCT
        dct_image = process_dct(train_data)
        dct_image_tensor = torch.tensor(dct_image).permute(0,1,2)

        # 目标图像DCT
        dct_gt = process_dct(train_data_tar)
        dct_gt_tensor = torch.tensor(dct_gt).permute(0,1,2)

        return (
            train_tensor,          # 原始输入图像
            train_tensor_tar,      # 目标图像
            flow_0_tensor,         # 光流特征0
            flow_1_tensor,         # 光流特征1
            dct_image_tensor,      # DCT变换后的输入
            dct_gt_tensor         # DCT变换后的目标
        )

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
        train_img = train_img.resize((256, 256))
        train_data = np.array(train_img) / 255.0  # 归一化到 [0, 1]
        train_data = train_data.astype(np.float32)  # 转换为 float32
        train_img = torch.tensor(train_data).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        train_img_tar = Image.open(train_img_tar_path).convert("RGB")
        train_img_tar = train_img_tar.resize((256, 256))
        train_data_tar = np.array(train_img_tar) / 255.0  # 归一化到 [0, 1]
        train_data_tar = train_data_tar.astype(np.float32)  # 转换为 float32
        train_img_tar = torch.tensor(train_data_tar).permute(0, 1, 2)  # 改变形状为 [C, H, W]
        
        flow_0 = Image.open(optical_img_0path).convert("RGB")
        flow_0 = flow_0.resize((256, 256))
        flow_0 = np.array(flow_0) / 255.0  # 归一化到 [0, 1]
        flow_0 = flow_0.astype(np.float32)  # 转换为 float32
        flow_0 = torch.tensor(flow_0).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        flow_1 = Image.open(optical_img_1path).convert("RGB")
        flow_1 = flow_1.resize((256, 256))
        flow_1 = np.array(flow_1) / 255.0  # 归一化到 [0, 1]
        flow_1 = flow_1.astype(np.float32)  # 转换为 float32
        flow_1 = torch.tensor(flow_1).permute(0, 1, 2)  # 改变形状为 [C, H, W]
        
        # 对每个通道进行DCT变换
        dct_channels = []
        for channel in range(train_data.shape[2]):  # 针对每个RGB通道分别做DCT
            # 对行和列分别进行DCT变换 (二维DCT)
            dct_channel = dct(dct(train_data[:, :, channel].T, norm='ortho').T, norm='ortho')
            dct_channels.append(dct_channel)

        # 将DCT变换后的各通道堆叠为numpy数组，并转换为张量
        dct_image = np.stack(dct_channels, axis=-1)
        dct_image = torch.tensor(dct_image).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        dct_channels = []
        for channel in range(train_data.shape[2]):  # 针对每个RGB通道分别做DCT
            # 对行和列分别进行DCT变换 (二维DCT)
            dct_channel = dct(dct(train_data_tar[:, :, channel].T, norm='ortho').T, norm='ortho')
            dct_channels.append(dct_channel)

        # 将DCT变换后的各通道堆叠为numpy数组，并转换为张量
        dct_gt = np.stack(dct_channels, axis=-1)
        dct_gt = torch.tensor(dct_gt).permute(0, 1, 2)  # 改变形状为 [C, H, W]


        return train_img, train_img_tar, flow_0, flow_1, dct_image, dct_gt





def compute_ddpm_linear_alpha_beta(num_diffusion_timesteps):
    # DDPM线性调度
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02  # 可以根据需要调整

    # 计算beta
    beta = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    # 计算alpha
    alpha = np.sqrt(1 - beta)

    return alpha, beta
 # Create the dataset
 
 
ACC4 = {
    "mask_type": "random",
    "center_fractions": [0.08],
    "accelerations": [4]
}
mask_fun = create_mask_for_mask_type("random", [0.08],  [4])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例用法
num_timesteps = 1000  # 时间步数
alpha, beta = compute_ddpm_linear_alpha_beta(num_timesteps)
model = KspaceGaussianDiffusion(
    beta_scale=0.8,
    alphas=alpha,
    betas=beta,
    diffusion_type=DiffusionType.DDPM,
    model_mean_type=ModelMeanType.EPSILON,
    model_var_type=ModelVarType.DEFAULT,
    predictor_type=PredictorType.DDPM,
    corrector_type=None,  # 如果没有使用校正器，可以传入 None
    loss_type=LossType.MSE
)

model_ = GaussianDiffusion(
    alphas=alpha,
    betas=beta,
    diffusion_type=DiffusionType.DDPM,
    model_mean_type=ModelMeanType.EPSILON,
    model_var_type=ModelVarType.DEFAULT,
    predictor_type=PredictorType.DDPM,
    corrector_type=None,  # 如果没有使用校正器，可以传入 None
    loss_type=LossType.MSE
)
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
#dataloader
batch_size = 4  # or use config.data.params.batch_size
dataset_dct = fewDataset(root_dir="/root/autodl-tmp/.autodl/dataset1-70")
dataloader_dct = DataLoader(dataset_dct, batch_size=batch_size, shuffle=True, num_workers=0, drop_last= True)
dct_dataset = dataloader_dct
def train_denoising_model(data_loader = dct_dataset, num_steps=200, num_epochs=300, batch_size=1, learning_rate=1e-4, test_ratio=0.15):
    wandb.init(
    # set the wandb project where this run will be logged
    project="DualDomainDiffusion",

    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "transformer+unet",
    "dataset": "Cine-mri",
    "epochs": 300,
    }
    )
            
    optimizer = optim.Adam(Unet.parameters(), lr=learning_rate)
    optimizer_t = optim.Adam(transformer.parameters(), lr=1e-4)
    optimizer_td = optim.Adam(transformer_dual.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    os.makedirs('denoised_images', exist_ok=True)
    # 划分数据集
    total_size = len(data_loader.dataset)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size
    print('Total number of images: {}'.format(total_size), '\nTotal number of training images: {}'.format(train_size), '\nTotal number of testing images: {}'.format(test_size))

    train_dataset, test_dataset = random_split(data_loader.dataset, [train_size, test_size])

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last= True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last= True)
    print('train_loader:',len(train_loader))

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_mse = 0
        Unet.train()
        transformer.train()
        transformer_dual.train()
        autoencoder.eval()
        autoencoder_optical.eval()
        autoencoder_dct.eval()
        # 训练阶段
        for k, (train_img, train_img_tar, flow_0, flow_1, dct_image, dct_gt) in enumerate(tqdm(train_loader)):
            train_img = train_img.to(device).permute(0,3,1,2)  #  [N, H, W, C]
            train_img_tar = train_img_tar.to(device)
            flow_0 = flow_0.to(device).permute(0,3,1,2)
            flow_1 = flow_1.to(device).permute(0,3,1,2)
            dct_image = dct_image.to(device).permute(0,3,1,2)
            optimizer.zero_grad()
            
            dct_image = autoencoder_dct.encode(dct_image).sample()
            train_img = autoencoder.encode(train_img).sample()
            flow_0 = autoencoder_optical.encode(flow_0).sample()
            flow_1 = autoencoder_optical.encode(flow_1).sample()
            train_img = train_img.permute(0,2,3,1)
            flow_0 = flow_0.permute(0,2,3,1)
            flow_1 = flow_1.permute(0,2,3,1)


            # 1. 前向加噪

            t = torch.randint(0, num_steps, (dct_image.size(0),), device=device)  # 随机选择时间步
            # 使用 list comprehension 生成每个 batch 的 mask
            masks = [torch.tensor(mask_fun(shape=dct_image[0].shape), dtype=torch.float32, device=device) for _ in
                    range(batch_size)]

            # 调用函数显示或保存 mask
            masks = torch.stack(masks)

            # 将 mask 放入 model_kwargs
            model_kwargs = {"mask_c": masks}

            noisy_images_pix, noise_pix = model_.q_sample(train_img, t)
            noisy_images_dct, noise_dct = model.q_sample(dct_image, t, model_kwargs=model_kwargs)
            denoised_images_dct = model.sample_loop(model = Unet, shape = noisy_images_dct.shape, model_kwargs=model_kwargs, noise = noisy_images_dct)
            denoised_images = transformer(train_img, flow_0, flow_1)
            denoised_images = denoised_images.reshape(train_img.shape)
            #denoised_images = model_.sample_loop(model = transformer, shape = noisy_images_pix.shape, model_kwargs=model_kwargs, noise = noisy_images_pix, flow_0=flow_0,flow_1=flow_1)
            denoised_images_d = dct_to_idct(denoised_images_dct)
            denoised_images_d = denoised_images_d.to(device)
            denoised_images = transformer_dual(denoised_images_d.permute(0,2,3,1), denoised_images)
            denoised_images = denoised_images.reshape(denoised_images_d.shape)

            denoised_images = autoencoder.decode(denoised_images)
            print(denoised_images.shape)



            save_denoised_images(denoised_images, "/root/autodl-tmp/.autodl/outputs", 1, 0)

            train_img_tar = train_img_tar.permute(0,3,1,2)

            # 2. 计算损失并反向传播
            loss_dct = model.training_losses(Unet, dct_image, t, model_kwargs=model_kwargs)
            loss_dct = loss_dct['loss'].sum()
            loss_g = criterion(denoised_images, train_img_tar)
            loss = loss_dct + loss_g

            loss.backward()
            optimizer.step()
            optimizer_t.step()
            optimizer_td.step()
            


            psnr_values, ssim_values, mse_values = compute_metrics(train_img_tar, denoised_images)


            print('TRAINING: mse:', sum(mse_values)/len(mse_values), 'psnr:', sum(psnr_values)/len(psnr_values),'ssim:',sum(ssim_values)/len(ssim_values),  'loss:', loss.item())
            wandb.log({"loss": loss.item(), "mse": sum(mse_values)/len(mse_values), "psnr": sum(psnr_values)/len(psnr_values), "ssim":sum(ssim_values)/len(ssim_values)})
            total_psnr += sum(psnr_values)/len(psnr_values)
            total_ssim += sum(ssim_values)/len(ssim_values)
            total_mse += sum(mse_values)/len(mse_values)
            total_loss += loss.item()

                        # print(f'eps: {eps}, std: {std} ')
        avg_mse = total_mse / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)
        avg_loss = total_loss
        # 输出当前 epoch 的度量结果
        print(f"Trian process: Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")


        Unet.eval()
        transformer_dual.eval()
        transformer.eval()# 设置模型为评估模式
        
        total_test_loss = 0
        total_test_psnr = 0
        total_test_ssim = 0
        total_test_mse = 0
        with torch.no_grad():
            for k, (train_img, train_img_tar, flow_0, flow_1, dct_image, dct_gt) in enumerate(tqdm(test_loader)):
                train_img = train_img.to(device).permute(0,3,1,2)  #  [N, H, W, C]
                train_img_tar = train_img_tar.to(device)
                flow_0 = flow_0.to(device).permute(0,3,1,2)
                flow_1 = flow_1.to(device).permute(0,3,1,2)
                dct_image = dct_image.to(device).permute(0,3,1,2)
                optimizer.zero_grad()
                
                dct_image = autoencoder_dct.encode(dct_image).sample()
                train_img = autoencoder.encode(train_img).sample()
                flow_0 = autoencoder_optical.encode(flow_0).sample()
                flow_1 = autoencoder_optical.encode(flow_1).sample()
                train_img = train_img.permute(0,2,3,1)
                flow_0 = flow_0.permute(0,2,3,1)
                flow_1 = flow_1.permute(0,2,3,1)


                # 1. 前向加噪

                t = torch.randint(0, num_steps, (dct_image.size(0),), device=device)  # 随机选择时间步
                # 使用 list comprehension 生成每个 batch 的 mask
                masks = [torch.tensor(mask_fun(shape=dct_image[0].shape), dtype=torch.float32, device=device) for _ in
                        range(batch_size)]

                # 调用函数显示或保存 mask
                masks = torch.stack(masks)

                # 将 mask 放入 model_kwargs
                model_kwargs = {"mask_c": masks}

                noisy_images_pix, noise_pix = model_.q_sample(train_img, t)
                noisy_images_dct, noise_dct = model.q_sample(dct_image, t, model_kwargs=model_kwargs)
                denoised_images_dct = model.sample_loop(model = Unet, shape = noisy_images_dct.shape, model_kwargs=model_kwargs, noise = noisy_images_dct)
                
                #denoised_images = model_.sample_loop(model = transformer, shape = noisy_images_pix.shape, model_kwargs=model_kwargs, noise = noisy_images_pix, flow_0=flow_0,flow_1=flow_1)
                denoised_images = transformer(train_img, flow_0, flow_1)
                denoised_images = denoised_images.reshape(train_img.shape)
                denoised_images_d = dct_to_idct(denoised_images_dct).to(device)
                denoised_images = transformer_dual(denoised_images_d.permute(0,2,3,1), denoised_images)
                
                denoised_images = denoised_images.reshape(denoised_images_d.shape)

                denoised_images = autoencoder.decode(denoised_images)
                save_denoised_images(denoised_images, "/root/autodl-tmp/.autodl/outputs", 2, 0)

                train_img_tar = train_img_tar.permute(0,3,1,2)
                # 2. 计算损失并反向传播
                loss_dct = model.training_losses(Unet, dct_image, t, model_kwargs=model_kwargs)
                loss_dct = loss_dct['loss'].sum()
                loss_g = criterion(denoised_images, train_img_tar)
                loss = loss_dct + loss_g
                


                psnr_values, ssim_values, mse_values = compute_metrics(train_img_tar, denoised_images)
                print('EVALUATING: mse:', sum(mse_values)/len(mse_values), 'psnr:', sum(psnr_values)/len(psnr_values),'ssim:',sum(ssim_values)/len(ssim_values),  'loss:', loss.item())
                wandb.log({"loss": loss.item(), "mse": sum(mse_values)/len(mse_values), "psnr": sum(psnr_values)/len(psnr_values), "ssim":sum(ssim_values)/len(ssim_values)})
                total_test_psnr += sum(psnr_values)/len(psnr_values)
                total_test_ssim += sum(ssim_values)/len(ssim_values)
                total_test_mse += sum(mse_values)/len(mse_values)
                total_test_loss += loss.item()
        avg_test_mse = total_test_mse / len(test_loader)
        avg_test_psnr = total_test_psnr / len(test_loader)
        avg_test_ssim = total_test_ssim / len(test_loader)
        avg_test_loss = total_test_loss
        # 输出当前 epoch 的度量结果
        print(
            f"val process: Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_test_loss:.4f}, MSE: {avg_test_mse:.4f}, PSNR: {avg_test_psnr:.4f}, SSIM: {avg_test_ssim:.4f}")
                # 将指标保存到txt文件
        with open("/root/autodl-tmp/.autodl/dataset1-70/Unet/validation_metrics.txt", "a") as file:
            file.write(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {total_test_loss:.4f}, SSIM: {avg_test_ssim:.4f}, PSNR: {avg_test_psnr:.4f}, MSE: {avg_test_mse}\n')

        #   # 保存模型权重
        weight_path = f"/root/autodl-tmp/.autodl/dataset1-70/Unet/Unet_last.pth"
        weight_path_t = f"/root/autodl-tmp/.autodl/dataset1-70/transformer/transformer_last.pth"
        weight_path_d = f"/root/autodl-tmp/.autodl/dataset1-70/Unet/dual_last.pth"
        torch.save(Unet.state_dict(), weight_path)
        torch.save(transformer.state_dict(), weight_path_t)
        torch.save(transformer_dual.state_dict(), weight_path_d)
        print(f"Model weights saved at {weight_path}")
    wandb.finish()

train_denoising_model()