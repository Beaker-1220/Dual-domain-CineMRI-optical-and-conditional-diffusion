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
from transformer import TransformerEncoder
from tqdm import tqdm
import torchmetrics
from unet import UNetModel
from pytorch_msssim import ssim
import torch as th
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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
        print(f"Saved {file_path}")
        
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
        train_img = train_img.resize((64, 64))
        train_data = np.array(train_img) / 255.0  # 归一化到 [0, 1]
        train_data = train_data.astype(np.float32)  # 转换为 float32

        train_img_tar = Image.open(train_img_tar_path).convert("RGB")
        train_img_tar = train_img_tar.resize((64, 64))
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
num_timesteps = 4500  # 时间步数
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
batch_size = 16  # or use config.data.params.batch_size
dataset_dct = ppDataset(root_dir="/root/autodl-tmp/.autodl/dataset1-70")
dataloader_dct = DataLoader(dataset_dct, batch_size=batch_size, shuffle=True, num_workers=0, drop_last= True)
dct_dataset = dataloader_dct

input_dim = 3  # 输入维度
d_model = 256  # 特征维度
num_layers = 6  # 编码器层数
nhead = 4  # 注意力头数
dim_feedforward = 512  # 前馈网络维度
dropout = 0.1  # Dropout 概率

#models
transformer = Transformer(num_layers, d_model, nhead, dim_feedforward, input_dim, dropout).to(device)
transformer_dual = Transformer_dual(num_layers, d_model, nhead, dim_feedforward, input_dim, dropout).to(device)
Unet = UNetModel(64, 3, 256, 3, 2, attention_resolutions = [32]).to(device)
def train_denoising_model(data_loader = dct_dataset, num_steps=4500, num_epochs=300, batch_size=16, learning_rate=0.01, test_ratio=0.15):
    optimizer = optim.Adam(Unet.parameters(), lr=learning_rate)
    optimizer_t = optim.Adam(transformer.parameters(), lr=1e-4)
    optimizer_td = optim.Adam(transformer_dual.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    transformer.load_state_dict(torch.load('/root/autodl-tmp/.autodl/dataset1-70/transformer/transformer_last.pth'))
    transformer_dual.load_state_dict(torch.load('/root/autodl-tmp/.autodl/dataset1-70/Unet/dual_last.pth'))
    Unet.load_state_dict(torch.load('/root/autodl-tmp/.autodl/dataset1-70/Unet/Unet_last.pth'))
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
        # 训练阶段
        for k, (train_img, train_img_tar, flow_0, flow_1, dct_image, dct_gt) in enumerate(tqdm(train_loader)):
            dct_image = dct_image.to(device)  # 将输入数据移动到设备
            dct_gt = dct_gt.to(device)  # 将标签移动到设备
            train_img = train_img.to(device)  #  [N, H, W, C]
            train_img_tar = train_img_tar.to(device)
            flow_0 = flow_0.to(device)
            flow_1 = flow_1.to(device)
            dct_image = dct_image.permute(0,3,1,2)
            dct_gt = dct_gt.permute(0,3,1,2)
            optimizer.zero_grad()

            # 1. 前向加噪

            t = torch.randint(0, num_steps, (dct_image.size(0),), device=device)  # 随机选择时间步
            # 使用 list comprehension 生成每个 batch 的 mask
            masks = [torch.tensor(mask_fun(shape=dct_image[0].shape), dtype=torch.float32, device=device) for _ in
                     range(batch_size)]

            # 调用函数显示或保存 mask
            masks = torch.stack(masks)

            # 将 mask 放入 model_kwargs
            model_kwargs = {"mask_c": masks}

            #transform optical featuremaps to pixel space
            image = transformer(train_img, flow_0, flow_1)
            image = image.reshape(train_img.shape)

            #generate noises and add to the images
            noisy_images_pix, noise_pix = model_.q_sample(image, t)
            noisy_images_dct, noise_dct = model.q_sample(dct_image, t, model_kwargs=model_kwargs)

            # predict noise
            N_dct = Unet(noisy_images_dct, model._scale_timesteps(t), masks)
            N_pix = Unet(noisy_images_pix.permute(0, 3, 1, 2), model_._scale_timesteps(t), masks)
            
            #transform back to dct space
            N_dct = dct_to_idct(N_dct).to(device)
            N_dct = N_dct.permute(0, 2, 3, 1)
            N_pix = N_pix.permute(0, 2, 3, 1)
            
            #dual-domain noises transformer
            N = transformer_dual(N_pix, N_dct)
            N = N.reshape(noisy_images_pix.shape)
            noisy_images_pix = noisy_images_pix.permute(0, 3, 1, 2)
            N = N.permute(0, 3, 1, 2)
            noise_pix = noise_pix.permute(0, 3, 1, 2)

            # noisy images sub noise                                                                         #method 1
            denoised_images = _extract_into_tensor(model_.recip_bar_alphas, t, noisy_images_pix.shape) * \
                        (noisy_images_pix - _extract_into_tensor(model_.bar_betas, t, noisy_images_pix.shape) * N)  #method 2
            image = image.permute(0, 3, 1, 2)

            save_denoised_images(denoised_images, "/root/autodl-tmp/.autodl/outputs", 0, 0)
            save_denoised_images(image, "/root/autodl-tmp/.autodl/outputs", 1, 0)


            # 2. 计算损失并反向传播
            loss_dct = model.training_losses(Unet, dct_image, t, model_kwargs=model_kwargs)
            loss_dct = loss_dct['loss'].sum()
            train_img_tar = train_img_tar.permute(0,3,1,2)
            loss_t = criterion(image, train_img_tar)
            loss_s = criterion(N, noise_pix)
            loss = loss_dct + loss_s
            loss.backward()
            optimizer.step()
            optimizer_t.step()
            optimizer_td.step()


            psnr_values, ssim_values, mse_values = compute_metrics(train_img_tar, denoised_images)

 
            print('TRAINING: mse:', sum(mse_values)/len(mse_values), 'psnr:', sum(psnr_values)/len(psnr_values),'ssim:',sum(ssim_values)/len(ssim_values),  'loss:', loss.item())
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
                dct_image = dct_image.to(device)  # 将输入数据移动到设备
                dct_gt = dct_gt.to(device)  # 将标签移动到设备
                train_img = train_img.to(device)  #  [N, H, W, C]
                train_img_tar = train_img_tar.to(device)
                flow_0 = flow_0.to(device)
                flow_1 = flow_1.to(device)
                dct_image = dct_image.permute(0,3,1,2)
                dct_gt = dct_gt.permute(0,3,1,2)
                optimizer.zero_grad()

                # 1. 前向加噪

                t = torch.randint(0, num_steps, (dct_image.size(0),), device=device)  # 随机选择时间步
                # 使用 list comprehension 生成每个 batch 的 mask
                masks = [torch.tensor(mask_fun(shape=dct_image[0].shape), dtype=torch.float32, device=device) for _ in
                        range(batch_size)]

                # 调用函数显示或保存 mask
                masks = torch.stack(masks)

                # 将 mask 放入 model_kwargs
                model_kwargs = {"mask_c": masks}
                visualize_mask(masks, idx=0, save_path="/root/autodl-tmp/.autodl/outputs/mask.png")  # 保存 mask

                #transform optical featuremaps to pixel space
                image = transformer(train_img, flow_0, flow_1)
                image = image.reshape(train_img.shape)

                noisy_images_pix, noise_pix = model_.q_sample(image, t)
                noisy_images_dct, noise_dct = model.q_sample(dct_image, t, model_kwargs=model_kwargs)
                N_dct = Unet(noisy_images_dct, model._scale_timesteps(t), masks)
                N_pix = Unet(noisy_images_pix.permute(0, 3, 1, 2), model_._scale_timesteps(t), masks)
                N_dct = dct_to_idct(N_dct).to(device)
                N_dct = N_dct.clamp(-1, 1).permute(0, 2, 3, 1)
                N_pix = N_pix.clamp(-1, 1).permute(0, 2, 3, 1)
                N = transformer_dual(N_pix, N_dct)
                N = N.reshape(noisy_images_pix.shape)
                # 3. 去噪
                noisy_images_pix = noisy_images_pix.permute(0, 3, 1, 2)
                N = N.permute(0, 3, 1, 2)
                noise_pix = noise_pix.permute(0, 3, 1, 2)
                
                # noisy images sub noise                                                                         #method 1
                denoised_images = _extract_into_tensor(model_.recip_bar_alphas, t, noisy_images_pix.shape) * \
                            (noisy_images_pix - _extract_into_tensor(model_.bar_betas, t, noisy_images_pix.shape) * N)  #method 2
                image = image.permute(0, 3, 1, 2)


                

                
                loss_dct = model.training_losses(Unet, dct_image, t, model_kwargs=model_kwargs)
                loss_dct = loss_dct['loss'].mean()
                train_img_tar = train_img_tar.permute(0,3,1,2)

                loss_s = criterion(N, noise_pix)
                loss = loss_dct + loss_s
                # 计算测试损失

                psnr_values, ssim_values, mse_values = compute_metrics(train_img_tar, denoised_images)

                save_denoised_images(denoised_images, "/root/autodl-tmp/.autodl/outputs", 2, 0)
                print('EVALUATING: mse:', sum(mse_values)/len(mse_values), 'psnr:', sum(psnr_values)/len(psnr_values),'ssim:',sum(ssim_values)/len(ssim_values),  'loss:', loss.item())
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


train_denoising_model()