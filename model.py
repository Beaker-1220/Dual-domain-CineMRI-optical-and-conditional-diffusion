import os
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from gaussian_diffusion import DiffusionType,CorrectorType,PredictorType,ModelMeanType,ModelVarType,LossType
from gaussian_diffusion import GaussianDiffusion
import torch
from scipy.fftpack import dct, idct
from transformer import Transformer
from utils.mri_data_utils.mask_util import create_mask_for_mask_type
from mcddpm_gaussian_diffusion import KspaceGaussianDiffusion
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torchmetrics
from unet import UNetModel
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import torch.nn.functional as F
from torchvision.utils import save_image

# DCT 到 IDCT
def dct_to_idct(dct_tensor):
    # 转换为 NumPy 数组
    dct_array = dct_tensor.numpy()

    # 执行 IDCT
    idct_array = idct(dct_array, type=2, norm='ortho', axis=(-2, -1), overwrite_x=True)

    # 转换回 PyTorch 张量
    return torch.tensor(idct_array)

# 转换为图像并计算指标
def compute_metrics(original_tensor, dct_tensor):
    # 执行 IDCT
    idct_tensor = dct_to_idct(dct_tensor)

    # 确保范围在 [0, 1]
    idct_tensor = idct_tensor.clamp(0, 1)

    # 将张量转换为 NumPy 数组并转换为 [H, W, C] 格式
    original_images = original_tensor.permute(0, 2, 3, 1).numpy()  # 原始图像
    reconstructed_images = idct_tensor.permute(0, 2, 3, 1).numpy()  # 重建图像

    # 计算 PSNR 和 SSIM
    psnr_values = []
    ssim_values = []
    for orig, recon in zip(original_images, reconstructed_images):
        # 计算 PSNR
        psnr_value = torchmetrics.functional.psnr(torch.tensor(recon), torch.tensor(orig), data_range=1.0)
        psnr_values.append(psnr_value.item())

        # 计算 SSIM
        ssim_value = compare_ssim(orig, recon, multichannel=True)
        ssim_values.append(ssim_value)

    return psnr_values, ssim_values

class PixelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.train_dir = os.path.join(root_dir)

        self.train_images = sorted(os.listdir(self.train_dir))
        # assert len(self.train_images) == len(self.target_images), "Train and Target folders must have the same number of images."

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        # 获取训练图像和目标图像的路径
        train_img_path = os.path.join(self.train_dir, self.train_images[idx])

        # 打开图像并转换为RGB格式（如果需要）
        train_img = Image.open(train_img_path).convert("RGB")
        train_img = train_img.resize((204, 448))
        train_data = np.array(train_img) / 255.0  # 归一化到 [0, 1]
        train_data = train_data.astype(np.float32)  # 转换为 float32
        train_img = torch.tensor(train_data).permute(0, 1, 2)  # 改变形状为 [C, H, W]

        return train_img



class dctDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.train_dir = os.path.join(root_dir, 'input')
        self.train_dir_tar = os.path.join(root_dir, 'target')
        self.train_images = sorted(os.listdir(self.train_dir))
        self.train_images_tar = sorted(os.listdir(self.train_dir_tar))
        # assert len(self.train_images) == len(self.target_images), "Train and Target folders must have the same number of images."

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

        train_img_tar = Image.open(train_img_tar_path).convert("RGB")
        train_img_tar = train_img_tar.resize((192, 448))
        train_data_tar = np.array(train_img_tar) / 255.0  # 归一化到 [0, 1]
        train_data_tar = train_data_tar.astype(np.float32)  # 转换为 float32


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

        return dct_image, dct_gt
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
num_timesteps = 200
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
batch_size = 1  # or use config.data.params.batch_size
dataset_dct = dctDataset(root_dir='/root/autodl-tmp/.autodl/dataset1-70/train')
dataloader_dct = DataLoader(dataset_dct, batch_size=batch_size, shuffle=True, num_workers=0, drop_last= True)

dct_dataset = dataloader_dct
# # 使用示例
# input_dim = 3  # 输入维度
# output_dim = 3  # 输出维度
# d_model = 256  # 特征维度
# num_layers = 6  # 编码器/解码器层数
# nhead = 8 # 注意力头数
# dim_feedforward = 512  # 前馈网络维度
# dropout = 0.1  # Dropout 概率

# transformer_model = Transformer(num_layers, d_model, nhead, dim_feedforward, input_dim, output_dim, dropout).to(device)
Unet = UNetModel(64, 3, 256, 3, 2, attention_resolutions = [32]).to(device)
def train_denoising_model(data_loader = dct_dataset, num_steps=200, num_units=32, num_epochs=100, batch_size=1, learning_rate=0.001, test_ratio=0.1):
    optimizer = optim.Adam(Unet.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # 使用均方误差损失



    # 划分数据集
    total_size = len(data_loader.dataset)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size
    print('Total number of images: {}'.format(total_size), '\nTotal number of training images: {}'.format(train_size), '\nTotal number of testing images: {}'.format(test_size))

    train_dataset, test_dataset = random_split(data_loader.dataset, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last= True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last= True)

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_mse = 0
        Unet.train()
        # 训练阶段
        for k, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device)  # 将输入数据移动到设备
            labels = labels.to(device)  # 将标签移动到设备

            inputs = inputs.permute(0,3,1,2)
            labels = labels.permute(0,3,1,2)
            optimizer.zero_grad()

            # 1. 前向加噪

            t = torch.randint(0, num_steps, (inputs.size(0),), device=device)  # 随机选择时间步
            # 使用 list comprehension 生成每个 batch 的 mask
            masks = [torch.tensor(mask_fun(shape=inputs[0].permute(1, 2, 0).shape), dtype=torch.float32, device=device) for _ in
                     range(batch_size)]


            # 将多个 mask 堆叠成一个 tensor
            masks = torch.stack(masks)
            # noisy_images = noisy_images.permute(0, 3, 1, 2)
            masks = masks.permute(0, 3, 1, 2)

            # 将 mask 放入 model_kwargs
            model_kwargs = {"mask_c": masks}


            noisy_images, noise = model.q_sample(inputs, t, model_kwargs=model_kwargs)
            denoised_images = Unet(noisy_images, t, masks, y= labels)
            # # 2. 计算模型的epsilon和标准差
            # eps, std = model.p_eps_std(Unet, noisy_images, t, model_kwargs=model_kwargs)


            if (epoch+1) % 10 == 0:
                print('sample loop:')
                denoised_images = model.sample_loop(model= Unet, noise=noise, shape=noisy_images.shape, model_kwargs=model_kwargs)
                torch.save(Unet.state_dict(), f'/root/autodl-tmp/pth/epoch_{epoch+1}.pth')
            # denoised_images = model.ddim_predictor(Unet, noisy_images, t, model_kwargs=model_kwargs)

            # 4. 计算损失
            loss = model.training_losses(Unet, inputs, t, model_kwargs=model_kwargs)
            loss = loss['loss']
            loss = torch.sum(loss)  # 或者 loss.sum() 取决于你的需求
            # 将 PyTorch 张量转换为 NumPy 数组
            # 计算指标
            psnr_values, ssim_values = compute_metrics(labels, denoised_images)

            denoised_images_np = denoised_images.cpu().detach().numpy()
            images_np = labels.cpu().detach().numpy()
            # print('image_np: ', images_np.shape)
            # print('denoised_images_np: ', denoised_images_np.shape)
            # print("denoised_images_np", denoised_images_np.shape)
            # print("images_np", images_np.shape)
            # psnr_val = compare_psnr(denoised_images_np, images_np, data_range=255)  # 计算 PSNR
            # # ssim_val = compare_ssim(denoised_images_np, images_np, channel_axis=1,data_range = 1.0)
            # # print(ssim_val)
            mse_val = compare_mse(images_np, denoised_images_np)
            print('mse:', mse_val, 'psnr:', psnr_values,'ssimL:',  'loss:', loss.item())
            total_psnr += psnr_values
            total_ssim += ssim_values
            total_mse += mse_val
            total_loss += loss.item()
            # 5. 反向传播和优化
            loss.backward()
            optimizer.step()
                        # print(f'eps: {eps}, std: {std} ')
        avg_mse = total_mse / len(train_loader)
        avg_psnr = total_psnr / len(train_loader.dataset)
        avg_ssim = total_ssim / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader.dataset)
        # 输出当前 epoch 的度量结果
        print(f"Trian process: Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        # 确保目标目录存在
        output_dir = 'weights'
        os.makedirs(output_dir, exist_ok=True)

        # 然后再保存模型权重
        torch.save(Unet.state_dict(), f'/root/autodl-tmp/pth/last.pth')
        # # 在每个 epoch 结束后，保存去噪后的图像
        # Unet.eval()  # 设置模型为评估模式
        # # 确保加载的是权重文件的路径
        #
        # weight_path = f'weights/last.pth'
        # Unet.load_state_dict(torch.load(weight_path))
        #
        # total_test_loss = 0
        # total_test_psnr = 0
        # total_test_ssim = 0
        # total_test_mse = 0
        # with torch.no_grad():
        #     for images in tqdm(test_loader):
        #         images = images.to(device)  # 将图像移动到设备上
        #         images = images.permute(0, 3, 1, 2)
        #
        #         t = torch.randint(0, num_steps, (images.size(0),), device=device)
        #         noisy_images, noise = model.q_sample(images, t, model_kwargs = model_kwargs)
        #         optimizer.zero_grad()
        #         denoised_images = model.sample_loop(model=Unet, noise=noisy_images, shape=noisy_images.shape, model_kwargs=model_kwargs)
        #         # denoised_images = model.ddim_predictor(Unet, noisy_images, t, model_kwargs=model_kwargs)
        #
        #         #denoised_images = Unet(noisy_images, torch.tensor(t).repeat(images.size(0)).to(device))
        #
        #         # 计算测试损失
        #         loss = criterion(denoised_images, images)  # 计算与原始图像的损失
        #         total_test_loss += loss.item()
        #         denoised_images_np = denoised_images.cpu().detach().numpy()
        #         images_np = images.cpu().numpy()
        #         psnr_val = compare_psnr(denoised_images_np, images_np, data_range=255)  # 计算 PSNR
        #         # ssim_val = ssim(denoised_images, images, window_size=11, size_average=True)
        #         mse_val = compare_mse(images_np, denoised_images_np)  # 计算 SSIM
        #         total_test_psnr += psnr_val
        #         # total_test_ssim += ssim_val
        #         total_test_mse += mse_val
        # avg_test_mse = total_test_mse / len(test_loader)
        # avg_test_psnr = total_test_psnr / len(test_loader.dataset)
        # avg_test_ssim = total_test_ssim / len(test_loader.dataset)
        # avg_test_loss = total_test_loss / len(test_loader.dataset)
        # # 输出当前 epoch 的度量结果
        # print(
        #     f"test process: Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_test_loss:.4f}, MSE: {avg_test_mse:.4f}, PSNR: {avg_test_psnr:.4f}, SSIM: {avg_test_ssim:.4f}")

train_denoising_model()