import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# create model
from torch import device

from config import args
from model import VisionTransformer
from utils import create_model, model_parallel

# 加载模型权重
model = nn.DataParallel(VisionTransformer)
model = VisionTransformer()
model.load_state_dict(torch.load('pretrain_weights/vit_base_patch16_224_in21k.pth'),strict=False)
model.eval()

# 数据预处理和转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像尺寸
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 加载并预处理图像
image = Image.open('dataset/SAT/1.jpg')  # 替换为你的图像路径
image = transform(image)
image = image.unsqueeze(0)  # 添加批次维度

# 图像分类
with torch.no_grad():
    output = model(image)

# 预测结果
_, predicted = torch.max(output, 1)
class_names = ['airplane', 'bridge', 'palace', 'ship', 'stadium']  # 替换为你的类别名称列表

# 图像显示
image = image.squeeze(0)  # 移除批次维度
image = image.numpy()
image = np.transpose(image, (1, 2, 0))
plt.imshow(image)
plt.title('Predicted: {}'.format(class_names[predicted.item()]))
plt.axis('off')
plt.show()
