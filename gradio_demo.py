import cv2
import numpy as np
from skimage import img_as_ubyte
import onnxruntime as ort
import os
import gradio as gr
from PIL import Image

# -------------------------------
# 模型和参数配置
# -------------------------------
onnx_model_path = 'restormer_deraining.onnx'  # 确保模型文件存在
img_multiple_of = 8

# 检查模型文件是否存在
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

# 加载 ONNX 模型
try:
    ort_session = ort.InferenceSession(onnx_model_path)
    print("✅ ONNX model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {e}")

# -------------------------------
# 推理函数
# -------------------------------
def denoise_image(input_img):
    """
    输入: PIL Image (RGB)
    输出: 去雨后的 PIL Image (RGB)
    """
    if input_img is None:
        return None

    try:
        # 强制转换为 RGB，避免 RGBA 或灰度图问题
        input_img = input_img.convert("RGB")
        img = np.array(input_img)  # HWC, RGB

        # 预处理：归一化 + CHW + batch
        input_ = img.astype(np.float32) / 255.0
        input_ = np.transpose(input_, (2, 0, 1))  # HWC -> CHW
        input_ = np.expand_dims(input_, axis=0)   # (1, C, H, W)

        # 获取原始尺寸
        _, _, h, w = input_.shape

        # Pad to multiple of 8
        H = ((h + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        padh = H - h
        padw = W - w
        input_ = np.pad(input_, ((0, 0), (0, 0), (0, padh), (0, padw)), mode='reflect')

        # ONNX 推理
        ort_inputs = {ort_session.get_inputs()[0].name: input_}
        restored = ort_session.run(None, ort_inputs)[0]  # (1, C, H, W)

        # 后处理：去 padding + 转回 HWC + uint8
        restored = restored[:, :, :h, :w]
        restored = np.clip(restored, 0, 1)
        restored = np.squeeze(restored)  # 去掉 batch 维度
        restored = np.transpose(restored, (1, 2, 0))  # CHW -> HWC
        restored = img_as_ubyte(restored)

        # 转为 PIL 图像输出
        return Image.fromarray(restored)

    except Exception as e:
        print(f"Error during denoising: {e}")
        return None

# -------------------------------
# Gradio 界面构建
# -------------------------------
with gr.Blocks(title="🌧️ Restormer 去雨模型在线演示") as demo:
    gr.Markdown("# 🌦️ 图像去雨 Demo (Restormer + ONNX)")
    gr.Markdown("上传一张带雨的图片，模型将自动去除雨滴。")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="原始图像 (带雨)", type="pil")
            submit_btn = gr.Button("🌧️ 去除雨滴", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="去雨后图像", type="pil")

    # 推理按钮：使用字符串 api_name，避免 None 导致的 schema 错误
    submit_btn.click(
        fn=denoise_image,
        inputs=input_image,
        outputs=output_image,
        api_name="denoise_image"  # ✅ 使用字符串，而不是 None
    )



# -------------------------------
# 启动服务
# -------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=12660,
        share=False,
        show_api=False  # 🔒 关键：禁用 API 文档，防止 schema 错误
    )