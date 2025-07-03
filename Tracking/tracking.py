import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from pinet import BinaryImageNet
from PIL import Image
from torchvision import transforms

# 加载预训练网络
pretrained_net = BinaryImageNet(input_dim=8)
pretrained_net.load_state_dict(torch.load('ckpt/seemsok/binary_image_net500.pth'))  # 加载权重
pretrained_net.eval()  # 设置为评估模式

# 读取参考图像
reference_image_path = 'datatrain/obj5/0002.png'  # 替换为参考图像的路径
reference_image = Image.open(reference_image_path).convert('L')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)) ])
reference_image = transform(reference_image).unsqueeze(0)
x = torch.tensor([[1.619279479980468750e+02,3.241745910644531250e+02,3.037741699218750000e+02,3.237776794433593750e+02,3.031668090820312500e+02,2.454943695068359375e+02,1.625327911376953125e+02,2.459501800537109375e+02]], requires_grad=True)

# 定义损失函数（这里使用均方误差）
criterion = nn.BCELoss()

# 定义优化器
optimizer = optim.Adam([x], lr=0.05)

# 优化过程
num_iterations = 300  # 迭代次数
for i in range(num_iterations):
    optimizer.zero_grad()  # 清空梯度
    generated_image = pretrained_net(x)  # 生成图像
    loss = criterion(generated_image, reference_image)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (i + 1) % 10 == 0:
        print(f'Iteration {i + 1}/{num_iterations}, Loss: {loss.item()}')

# 保存优化后的向量x和生成的图像
print(x)
save_image(generated_image, 'generated_image.png')

print("优化完成！")
