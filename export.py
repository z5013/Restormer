import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np
import os  

def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    else:
        raise ValueError(f"Unsupported task: {task}")
    print(weights)
    return weights, parameters

# ================================
# 配置区：修改这里来切换任务
# ================================
task = 'Single_Image_Defocus_Deblurring'  # 可选: 'Motion_Deblurring', 'Single_Image_Defocus_Deblurring', 'Deraining', 'Real_Denoising'
onnx_output_path = f'restormer_{task.lower().replace("_", "_")}.onnx'  # 自动命名
input_height, input_width = 256, 256  # 推荐使用 256x256 或 512x512，也可设为动态
# ================================

# Get model weights and parameters
parameters = {
    'inp_channels': 3,
    'out_channels': 3,
    'dim': 48,
    'num_blocks': [4, 6, 6, 8],
    'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8],
    'ffn_expansion_factor': 2.66,
    'bias': False,
    'LayerNorm_type': 'WithBias',
    'dual_pixel_task': False
}

weights, parameters = get_weights_and_parameters(task, parameters)

# Load architecture
load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)
model.cpu()  # or model.cpu()

# Load weights
checkpoint = torch.load(weights, map_location='cpu')  # 保持与 model 一致
model.load_state_dict(checkpoint['params'])
model.eval()

# ===================================
# 导出为 ONNX 模型
# ===================================
dummy_input = torch.randn(1, 3, input_height, input_width, device='cpu')  # 与 model 同设备

# 可选：先测试前向传播是否正常
with torch.no_grad():
    output = model(dummy_input)
    print(f"Forward pass successful! Output shape: {output.shape}")

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,               # 包含权重
    opset_version=13,                 # 推荐使用 13 或更高
    do_constant_folding=True,         # 优化常量
    input_names=['input'],            # 输入名
    output_names=['output'],          # 输出名
    dynamic_axes={
        "input": {2: "height", 3: "width"},   # 动态高度和宽度
        "output": {2: "height", 3: "width"}
    },
    verbose=False,                    # 可设为 True 查看详细信息（调试用）
)

print(f"ONNX 模型已成功导出到: {onnx_output_path}")

