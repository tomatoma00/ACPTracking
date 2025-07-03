import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化SSIM和PSNR
def gaussian(window_size, sigma):
    """生成高斯滤波器"""
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x ** 2 / float(2 * sigma ** 2)))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """创建高斯窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    """计算SSIM"""
    L = 1.0  # 假设像素值范围为[0, 1]
    pad = window_size // 2
    if window is None:
        window = create_window(window_size, img1.size(1)).to(img1.device)

    mu1 = F.conv2d(img1, window, groups=img1.size(1), padding=pad)
    mu2 = F.conv2d(img2, window, groups=img2.size(1), padding=pad)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=img1.size(1), padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=img2.size(1), padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=img1.size(1), padding=pad) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0  # 假设像素值范围为[0, 1]
    psnr_value = 10 * torch.log10(max_pixel_value ** 2 / mse)
    return psnr_value
  
def psnr2(img1, img2):
    """计算PSNR"""
    # 找出img1中不为0的像素
    mask = img2 != 0
    # 使用mask来选择img1和img2中对应的非零像素
    img1_non_zero = img1[mask]
    img2_non_zero = img2[mask]
    # 计算非零像素的MSE
    mse = torch.mean((img1_non_zero - img2_non_zero) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0  # 假设像素值范围为[0, 1]
    psnr_value = 10 * torch.log10(max_pixel_value ** 2 / mse)
    return psnr_value
  
# 数据加载函数
def load_data(data_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(),  # 确保图像是灰度图
        transforms.ToTensor(),   # 将图像转换为 Tensor
        lambda x: torch.round(x)  # 将像素值四舍五入到 0 或 1
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [batch, 16, 256, 256]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [batch, 32, 128, 128]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, 64, 64]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [batch, 128, 32, 32]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [batch, 256, 16, 16]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [batch, 512, 8, 8]
            nn.ReLU(),
            nn.Flatten(),  # [batch, 512*8*8]
            nn.Linear(512*8*8, 128)  # [batch, 128]
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(128, 512*8*8),  # [batch, 512*8*8]
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),  # [batch, 512, 8, 8]
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # [batch, 256, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # [batch, 128, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # [batch, 64, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # [batch, 32, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # [batch, 16, 256, 256]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # [batch, 1, 512, 512]
            nn.Sigmoid()  # 输出像素值在 [0, 1] 之间
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 训练函数
def train_aae(model, discriminator, dataloader, start_epoch=0, num_epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 划分训练集和评估集
    train_size = int(0.8 * len(dataloader.dataset))  # 80% 数据用于训练
    eval_size = len(dataloader.dataset) - train_size  # 剩余20% 数据用于评估
    train_dataset, eval_dataset = random_split(dataloader.dataset, [train_size, eval_size])

    train_dataloader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader.batch_size, shuffle=False)

    for epoch in range(start_epoch, start_epoch+num_epochs):
        model.train()
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # 计算原始损失
            original_loss = criterion(output, data)

            # 计算SSIM损失
            ssim_loss = 1 - ssim(output, data)

            # 计算PSNR损失
            psnr_loss = 1 / (psnr(output, data) + 1e-6)  # 防止除零
            psnr_loss2 = 1 / (psnr2(output, data) + 1e-6)  # 防止除零
            psnr_loss3 = 1 / (psnr2(data, output) + 1e-6)  # 防止除零

            # 组合损失，权重比例为2:1:1
            loss = 2 * original_loss + ssim_loss + psnr_loss + psnr_loss2 + psnr_loss3

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            if batch_idx % 30 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Loss D: NA, Loss G: NA")
                save_image(output, f"trainres/reconstructed_images_epoch_{epoch}_batch_{batch_idx}.png", nrow=8)
        # 保存模型
        if epoch % 3 ==0:
            torch.save(model.state_dict(), f"ckpt/autoencoder_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"ckpt/discriminator_epoch_{epoch+1}.pth")

# 测试函数
def test_aae(model, dataloader, epoch):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            output = model(data)
            save_image(output, f"reconstructed_images_epoch_{epoch}_batch_{batch_idx}.png", nrow=16)
            break

# 主函数
if __name__ == "__main__":
    data_path = 'datatrain'
    batch_size = 512
    num_epochs = 100
    start_epoch = 199

    # 加载数据
    dataloader = load_data(data_path, batch_size)

    # 实例化模型
    model = Autoencoder().to(device)
    if start_epoch!=0:
        model.load_state_dict(torch.load(f"ckpt/seemsok/autoencoder_epoch_{start_epoch}.pth"))
    discriminator = Discriminator().to(device)

    # 训练模型
    train_aae(model, discriminator, dataloader, start_epoch=start_epoch,  num_epochs=num_epochs,lr=0.0005)

    # 测试模型
    test_aae(model, dataloader, num_epochs)
