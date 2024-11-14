import tensorflow.compat.v1 as tf # import tensorflow as tf
import pywt
import numpy as np
import cv2
import os

# 设置 TensorFlow 配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 动态增长 GPU 内存使用
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

def scale(x, mode=0, axis=None):
    xmax, xmin = x.max(), x.min()
    x = (x - xmin) / (xmax - xmin)
    if mode != 0:
        x = (x - 0.5) * 2
    return x

def DWT(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    norm_LL = scale(LL, 0, 2)
    norm_LH = scale(LH, -1, 2)
    norm_HL = scale(HL, -1, 2)
    return norm_LL, norm_LH, norm_HL

# 加载模型
model = tf.keras.models.load_model("weights-01-0.9875.hdf5")

# 读取测试图像
img_path = 'path_to_your_test_image.jpg'  # 替换为实际测试图像的路径
img = cv2.imread(img_path, 0)  # 以灰度模式读取图像
img = cv2.resize(img, (1280, 960))  # 调整图像大小为 642x482，匹配模型输入

# 对图像进行 DWT 变换
ll, lh, hl = DWT(img)

# 将 DWT 的结果扩展为四维数组，以匹配模型输入
ll = ll[np.newaxis, ..., np.newaxis]
lh = lh[np.newaxis, ..., np.newaxis]
hl = hl[np.newaxis, ..., np.newaxis]

# 进行预测
res = model.predict([ll, lh, hl])
class_res = np.argmax(res, axis=1)

# 打印预测结果
print("Predicted class:", class_res)