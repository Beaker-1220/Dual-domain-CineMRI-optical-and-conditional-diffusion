from transformer import Transformer,Transformer_dual
from unet import UNetModel
import torch
import matplotlib.pyplot as plt
import torchmetrics
from torchvision.utils import save_image
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from scipy.fftpack import dct, idct
import os
import numpy as np
import yaml
from autoencoder import AutoencoderKL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 3  # 输入维度
d_model = 256  # 特征维度
num_layers = 6  # 编码器层数
nhead = 4  # 注意力头数
dim_feedforward = 512  # 前馈网络维度
dropout = 0.1  # Dropout 概率

with open('autoencoder_kl_64x64x3.yaml', 'r') as file:
    config = yaml.safe_load(file)

ddconfig = config['model']['params']['ddconfig']
lossconfig = config['model']['params']['lossconfig']
embed_dim = config['model']['params']['embed_dim']
image_key = None
colorize_nlabels = None
monitor = config['model']['params']['monitor'] 
ckpt_path = '/root/latent-diffusion/logs/2024-10-14T17-59-13_autoencoder_kl_64x64x3/checkpoints/last.ckpt'



autoencoder = AutoencoderKL(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    embed_dim=embed_dim,
    ckpt_path=ckpt_path,
    image_key=image_key,
    colorize_nlabels=colorize_nlabels,
    monitor=monitor
).to(device)

autoencoder_dct = AutoencoderKL(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    embed_dim=embed_dim,
    ckpt_path="/root/latent-diffusion/logs/2024-10-16T02-17-23_autoencoder_kl_64x64x3/checkpoints/last.ckpt",
    image_key=image_key,
    colorize_nlabels=colorize_nlabels,
    monitor=monitor
).to(device)
autoencoder_optical = AutoencoderKL(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    embed_dim=embed_dim,
    ckpt_path='/root/latent-diffusion/logs/2024-10-16T12-14-11_autoencoder_kl_64x64x3/checkpoints/last.ckpt',
    image_key=image_key,
    colorize_nlabels=colorize_nlabels,
    monitor=monitor
).to(device)

transformer = Transformer(num_layers, d_model, nhead, dim_feedforward, input_dim, dropout).to(device)
transformer_dual = Transformer_dual(num_layers, d_model, nhead, dim_feedforward, input_dim, dropout).to(device)
Unet = UNetModel(64, 3, 256, 3, 2, attention_resolutions = [32]).to(device)


def visualize_mask(mask_tensor, idx=0, save_path=None):
    """
    将 mask_tensor 中的第 idx 张 mask 可视化或保存。

    :param mask_tensor: Tensor, 形状为 [batch_size, channels, height, width]
    :param idx: int, batch 内要可视化的第几张 mask
    :param save_path: str, 如果提供路径，将图像保存到指定路径
    """
    mask_image = mask_tensor[idx].detach().cpu()  # 选择第 idx 个 mask, 并移动到 CPU
    mask_image = T.ToPILImage()(mask_image)  # 将张量转换为 PIL 图像
    
    if save_path:
        mask_image.save(save_path)  # 保存图像
        print(f"Mask saved at {save_path}")
    else:
        # 显示图像
        plt.imshow(mask_image, cmap="gray")
        plt.axis("off")
        plt.show()



# DCT 到 IDCT
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
        # print(f"Saved {file_path}")
        
def dct_to_idct(dct_tensor):
    # 转换为 NumPy 数组
    dct_array = dct_tensor.cpu().detach().numpy()

    # 执行 IDCT
    # 对每个通道执行 IDCT 变换
    idct_channels = []
    for channel in range(dct_array.shape[2]):  # 遍历每个 RGB 通道
        # 对行和列分别执行 IDCT 变换 (二维 IDCT)
        idct_channel = idct(idct(dct_array[:, :, channel].T, norm='ortho').T, norm='ortho')
        idct_channels.append(idct_channel)

    # 将 IDCT 变换后的各通道堆叠成 numpy 数组
    idct_image_np = np.stack(idct_channels, axis=-1)

    # 转换回 PyTorch 张量
    return torch.tensor(idct_image_np)

# 转换为图像并计算指标
def compute_metrics(original_tensor, dct_tensor):
    # 执行 IDCT
    idct_tensor = dct_to_idct(dct_tensor)
    original_tensor = dct_to_idct(original_tensor)

    # 确保范围在 [0, 1]
    idct_tensor = idct_tensor.clamp(0, 1)
    original_tensor = original_tensor.clamp(0, 1)


    # 将张量转换为 NumPy 数组并转换为 [H, W, C] 格式
    original_images = original_tensor.cpu().detach().numpy()  # 原始图像
    reconstructed_images = idct_tensor.cpu().detach().numpy() # 重建图像
    # 计算 PSNR 和 SSIM
    psnr_values = []
    ssim_values = []
    mse_values = []
    for orig, recon in zip(original_images, reconstructed_images):
        # 计算 PSNR
        psnr_value = torchmetrics.functional.psnr(torch.tensor(orig), torch.tensor(recon), data_range=1.0)
        psnr_values.append(psnr_value.item())
        # 计算 SSIM
        ssim_value = compare_ssim(orig, recon,channel_axis=0, data_range=1.0)
        ssim_values.append(ssim_value)
        
        mse_value = compare_mse(orig, recon)
        mse_values.append(mse_value)

    return psnr_values, ssim_values, mse_values