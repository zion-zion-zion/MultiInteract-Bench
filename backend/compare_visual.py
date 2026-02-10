import os
import json
import re
from pathlib import Path
from PIL import Image
import torch
import clip
import lpips
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from sklearn.cluster import KMeans


def load_clip_model():
    """加载CLIP模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def load_lpips_model():
    """加载LPIPS模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用VGG作为backbone，net='vgg'或'alex'
    model = lpips.LPIPS(net='vgg').to(device)
    return model, device


def load_vgg_model():
    """加载VGG19模型用于计算Style Loss
    
    VGG19是神经风格迁移中常用的预训练模型，能够提取图像的风格特征。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载预训练的VGG19模型
    vgg = models.vgg19(pretrained=True).features.to(device)
    
    # 冻结模型参数（不需要训练）
    for param in vgg.parameters():
        param.requires_grad = False
    
    return vgg, device


def load_dino_model():
    """加载DINOv2模型用于计算结构相似度
    
    DINOv2（DINOv2: Learning Robust Visual Features without Supervision）是Meta发布的
    无监督学习视觉特征模型，基于Vision Transformer (ViT)架构。
    
    该模型能够提取图像的高层语义和结构特征，对布局、物体位置等结构信息非常敏感。
    
    Returns:
        model: DINOv2模型
        preprocess: 图像预处理函数
        device: 计算设备（cuda或cpu）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 从 torch.hub 加载 DINOv2 模型
        # 使用 'dinov2_vitb14' 模型（ViT-Base, patch size 14x14）
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        
        # 冻结模型参数（不需要训练）
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # 定义图像预处理
        # DINOv2 期望的输入尺寸是 224x224
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return model, preprocess, device
    except Exception as e:
        print(f"Error loading DINOv2 model: {e}")
        print("DINOv2 model not available, DINO similarity will be skipped")
        return None, None, device


def gram_matrix(tensor):
    """计算Gram矩阵
    
    Gram矩阵用于表示图像的风格特征，通过计算特征图之间的相关性来捕捉纹理、颜色等风格信息。
    
    Args:
        tensor: 形状为 (batch_size, channels, height, width) 的特征图
    
    Returns:
        Gram矩阵，形状为 (batch_size, channels, channels)
    """
    batch_size, channels, height, width = tensor.size()
    
    # 将特征图展平为 (batch_size, channels, height * width)
    features = tensor.view(batch_size, channels, height * width)
    
    # 计算 Gram 矩阵: G = F * F^T
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # 归一化
    gram = gram / (channels * height * width)
    
    return gram


def calculate_style_loss(vgg, device, image_path1, image_path2):
    """计算两张图片的Style Loss（Gram Matrix Distance）
    
    Style Loss源自神经风格迁移领域，通过计算特征图的Gram矩阵之间的差异，
    来衡量两张图的艺术风格是否接近。
    
    用途：判断生成的网页是否丢失了原设计的"设计感"或"氛围"。
    
    返回值范围：0-∞，越小越相似
    """
    try:
        # 定义预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # VGG的标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
        
        # 加载并预处理图片
        img1 = Image.open(image_path1).convert('RGB')
        img2 = Image.open(image_path2).convert('RGB')
        
        img1_tensor = preprocess(img1).unsqueeze(0).to(device)
        img2_tensor = preprocess(img2).unsqueeze(0).to(device)
        
        # 选择用于计算风格损失的层（VGG的中间层）
        # 这些层通常用于捕捉图像的风格信息
        style_layers = ['0', '5', '10', '19', '28']  # 对应conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        
        total_style_loss = 0.0
        
        # 逐步提取特征并计算Gram矩阵
        x1 = img1_tensor
        x2 = img2_tensor
        
        layer_weights = {
            '0': 1.0/5,  # conv1_1
            '5': 1.0/5,  # conv2_1
            '10': 1.0/5, # conv3_1
            '19': 1.0/5, # conv4_1
            '28': 1.0/5  # conv5_1
        }
        
        with torch.no_grad():
            for name, layer in vgg._modules.items():
                x1 = layer(x1)
                x2 = layer(x2)
                
                # 如果是选定的风格层，计算该层的风格损失
                if name in style_layers:
                    # 计算两张图片的Gram矩阵
                    gram1 = gram_matrix(x1)
                    gram2 = gram_matrix(x2)
                    
                    # 计算Gram矩阵之间的欧氏距离（即风格损失）
                    layer_loss = torch.mean((gram1 - gram2) ** 2)
                    total_style_loss += layer_weights[name] * layer_loss
        
        return float(total_style_loss.cpu().numpy())
    except Exception as e:
        print(f"Error calculating Style Loss for {image_path1} and {image_path2}: {e}")
        return None


def calculate_clip_similarity(model, preprocess, device, image_path1, image_path2):
    """计算两张图片的CLIP相似度
    
    注意：CLIP的预处理函数会自动将图片调整到相同大小（通常是224x224），
    并进行归一化处理，因此不需要手动resize。
    
    返回值范围：0-1，越大越相似
    """
    try:
        # 加载并预处理图片
        # preprocess会自动resize到模型所需尺寸（ViT-B/32为224x224）
        image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 获取图片特征
            image_features1 = model.encode_image(image1)
            image_features2 = model.encode_image(image2)
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(image_features1, image_features2)
            
            return float(similarity.cpu().numpy()[0])
    except Exception as e:
        print(f"Error calculating CLIP similarity for {image_path1} and {image_path2}: {e}")
        return None


def calculate_lpips_distance(model, device, image_path1, image_path2):
    """计算两张图片的LPIPS距离
    
    LPIPS是基于深度学习的感知相似度指标，能够更好地捕捉人类视觉感知的差异。
    
    注意：LPIPS需要RGB格式的图片，值在[0,1]范围内，且两张图片必须有相同尺寸。
    返回值范围：0-∞，越小越相似（通常小于1表示比较相似）
    """
    try:
        # 加载图片并转换为RGB
        img1 = Image.open(image_path1).convert('RGB')
        img2 = Image.open(image_path2).convert('RGB')
        
        # 将两张图片resize到相同尺寸（使用较小的尺寸）
        width = min(img1.width, img2.width)
        height = min(img1.height, img2.height)
        img1 = img1.resize((width, height), Image.Resampling.LANCZOS)
        img2 = img2.resize((width, height), Image.Resampling.LANCZOS)
        
        # 转换为张量，值归一化到[-1, 1]范围
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img1_tensor = transform(img1).unsqueeze(0).to(device)
        img2_tensor = transform(img2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 计算LPIPS距离
            distance = model(img1_tensor, img2_tensor)
            
            return float(distance.cpu().numpy()[0])
    except Exception as e:
        print(f"Error calculating LPIPS distance for {image_path1} and {image_path2}: {e}")
        return None


# 全局变量，缓存 EasyOCR 实例
_ocr_reader = None

def get_ocr_reader():
    """获取缓存的 EasyOCR Reader 实例
    
    自动检测 CUDA 是否可用，如果可用则使用 GPU 加速，显著提高 OCR 处理速度。
    """
    global _ocr_reader
    
    if _ocr_reader is None:
        try:
            import easyocr
            # 检测 GPU 是否可用
            use_gpu = torch.cuda.is_available()
            
            if use_gpu:
                print("CUDA is available. Initializing EasyOCR with GPU acceleration...")
            else:
                print("CUDA is not available. Initializing EasyOCR on CPU...")
            
            _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            print(f"Using EasyOCR backend (GPU: {use_gpu})")
        except ImportError:
            print("Warning: easyocr not installed. Install with: pip install easyocr")
            return None
        except Exception as e:
            print(f"Warning: Failed to initialize EasyOCR: {e}")
            return None
    
    return _ocr_reader


def extract_text_from_image(image_path):
    """从图片中提取文本
    
    使用 EasyOCR 技术提取图片中的可见文本。
    EasyOCR 是一个基于深度学习的 OCR 工具，支持多种语言，性能优异。
    
    注意：需要安装 easyocr 库
    安装命令：pip install easyocr
    
    返回提取的文本字符串
    """
    try:
        reader = get_ocr_reader()
        if reader is None:
            return ""
        
        # 进行OCR识别
        # EasyOCR 返回格式: [[bbox, text, confidence], ...]
        result = reader.readtext(image_path)
        
        # 提取所有识别出的文本
        text_parts = []
        for item in result:
            if len(item) >= 2:
                text_parts.append(item[1])
        
        # 合并所有文本
        text = ' '.join(text_parts)
        
        return text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""


def longest_common_subsequence_length(text1, text2):
    """计算两个字符串的最长公共子序列（LCS）长度
    
    使用动态规划算法计算LCS长度。
    
    Args:
        text1: 第一个文本字符串
        text2: 第二个文本字符串
    
    Returns:
        LCS长度
    """
    m, n = len(text1), len(text2)
    
    # 创建 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def calculate_text_similarity(image_path1, image_path2):
    """计算两张图片的文本相似度（LCS 长度比率）
    
    通过 OCR 提取两张图片中的文本，然后计算最长公共子序列（LCS）长度比率。
    
    计算公式：LCS_length / min(len(text1), len(text2))
    
    返回值范围：0-1，越大越相似
    
    用途：判断生成的网页是否保留了原设计的文本内容和结构
    """
    try:
        # 从两张图片中提取文本
        text1 = extract_text_from_image(image_path1)
        text2 = extract_text_from_image(image_path2)
        
        # 如果其中一个文本为空，返回0
        if not text1 or not text2:
            return 0.0
        
        # 计算 LCS 长度
        lcs_length = longest_common_subsequence_length(text1, text2)
        
        # 计算 LCS 比率：LCS长度除以较短文本的长度
        min_length = min(len(text1), len(text2))
        lcs_ratio = lcs_length / min_length
        
        return float(lcs_ratio)
    except Exception as e:
        print(f"Error calculating text similarity for {image_path1} and {image_path2}: {e}")
        return None


def calculate_color_histogram_similarity(image_path1, image_path2):
    """计算两张图片的色彩直方图相似度
    
    通过对比RGB空间的色彩分布来衡量两张图片的整体色彩风格是否一致。
    这能够衡量整体的亮度、饱和度和色相是否一致（例如：都是暗黑模式，还是清新明亮风格）。
    
    方法：
    1. 分别计算R、G、B三个通道的直方图（256 bins）
    2. 使用相关系数（correlation）比较直方图的相似性
    3. 对三个通道的结果取平均
    
    返回值范围：0-1，越大越相似
    
    用途：判断生成的网页是否保持了原设计的整体色彩风格
    """
    try:
        # 读取图片
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        
        if img1 is None or img2 is None:
            print(f"Error: Failed to read images for color histogram comparison")
            return None
        
        # 计算直方图
        # 参数说明: [channels], mask, [histSize], [ranges]
        hist1 = []
        hist2 = []
        
        for i in range(3):  # 分别计算R、G、B三个通道
            h1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            h2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            
            # 归一化直方图
            cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
            
            hist1.append(h1)
            hist2.append(h2)
        
        # 计算相关性（correlation）
        # 相关性范围：-1到1，1表示完全正相关
        correlations = []
        for i in range(3):
            corr = cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_CORREL)
            correlations.append(corr)
        
        # 计算平均相关性
        avg_correlation = sum(correlations) / len(correlations)
        
        # 将相关性从[-1, 1]映射到[0, 1]范围
        # 相关性 >= 0 时，相似度 = (1 + correlation) / 2
        # 相关性 < 0 时，相似度 = 0
        similarity = max(0, (1 + avg_correlation) / 2)
        
        return float(similarity)
    except Exception as e:
        print(f"Error calculating color histogram similarity for {image_path1} and {image_path2}: {e}")
        return None


def extract_dominant_colors(image_path, n_colors=5):
    """使用K-means聚类提取图片的主导颜色
    
    Args:
        image_path: 图片路径
        n_colors: 要提取的主导颜色数量，默认为5
    
    Returns:
        dominant_colors: 主导颜色列表，每个颜色为(R, G, B)格式
        color_percentages: 每个主导颜色的占比（像素比例）
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        # 将图片从BGR转换为RGB（OpenCV默认使用BGR）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 将图片调整为较小尺寸以加速处理
        # 缩放到最长边为200像素
        height, width = img.shape[:2]
        scale = min(200 / height, 200 / width)
        img_small = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        # 将图片reshape为一维数组
        pixels = img_small.reshape(-1, 3)
        
        # 使用K-means聚类找到主导颜色
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # 获取聚类中心（即主导颜色）
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        # 计算每个主导颜色的占比
        labels = kmeans.labels_
        total_pixels = len(labels)
        color_counts = np.bincount(labels, minlength=n_colors)
        color_percentages = color_counts / total_pixels
        
        return dominant_colors, color_percentages
    except Exception as e:
        print(f"Error extracting dominant colors from {image_path}: {e}")
        return None, None


def rgb_to_hsv(rgb):
    """将RGB颜色转换为HSV
    
    Args:
        rgb: RGB颜色，值为0-255的整数或0-1的浮点数
    
    Returns:
        hsv: HSV颜色，h为0-360，s和v为0-1
    """
    # 确保RGB值为0-1范围
    rgb_norm = np.array(rgb) / 255.0
    
    # 转换为HSV
    hsv = cv2.cvtColor(np.array([[rgb_norm]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0][0]
    
    # OpenCV返回的H范围是0-180，需要转换到0-360
    h, s, v = hsv
    h = h * 2  # 转换到0-360
    
    return h, s, v


def calculate_hue_distance(hue1, hue2):
    """计算两个色相在色轮上的最小距离
    
    Args:
        hue1: 第一个色相，0-360
        hue2: 第二个色相，0-360
    
    Returns:
        distance: 色相距离，0-180
    """
    diff = abs(hue1 - hue2)
    # 色轮是圆形的，所以距离是差值和360-差值中的较小者
    return min(diff, 360 - diff)


def calculate_dino_similarity(model, preprocess, device, image_path1, image_path2):
    """计算两张图片的DINOv2相似度
    
    DINOv2（DINOv2: Learning Robust Visual Features without Supervision）是Meta发布的
    无监督学习视觉特征模型，基于Vision Transformer (ViT)架构。
    
    该模型能够提取图像的高层语义和结构特征，对布局、物体位置等结构信息非常敏感。
    
    方法：
    1. 使用预训练的DINOv2模型作为特征提取器Φ
    2. 将原始参考截图(I_gt)和模型生成的网页截图(I_pred)映射为高维特征向量
    3. 取最后一层Transformer的[CLS] token的输出作为代表全图结构信息的特征向量
    4. 计算这两个向量之间的余弦相似度
    
    公式：
    v_gt = Φ(I_gt)
    v_pred = Φ(I_pred)
    S_DINO = cosine_similarity(v_gt, v_pred)
    
    返回值范围：0-1，越大越相似
    
    用途：判断生成的网页是否保持了原设计的整体结构布局（例如：导航栏位置、主要内容区域布局等）
    """
    try:
        if model is None or preprocess is None:
            print(f"Warning: DINOv2 model not available")
            return None
        
        # 加载并预处理图片
        img1 = Image.open(image_path1).convert('RGB')
        img2 = Image.open(image_path2).convert('RGB')
        
        img1_tensor = preprocess(img1).unsqueeze(0).to(device)
        img2_tensor = preprocess(img2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 获取图片特征
            # DINOv2 forward 返回的形状是 (batch_size, num_patches+1, feature_dim)
            # num_patches+1 是因为包含了 [CLS] token
            features1 = model(img1_tensor)
            features2 = model(img2_tensor)
            
            # 检查特征输出的形状
            # DINOv2 可能有不同的输出格式，需要适配
            if features1.dim() == 3:
                # 形状为 (batch_size, num_patches+1, feature_dim)
                cls_features1 = features1[:, 0, :]
                cls_features2 = features2[:, 0, :]
            elif features1.dim() == 2:
                # 形状为 (batch_size, feature_dim)，已经是全局特征
                cls_features1 = features1
                cls_features2 = features2
            else:
                print(f"Warning: Unexpected DINOv2 output shape: {features1.shape}")
                return None
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(cls_features1, cls_features2)
            
            return float(similarity.cpu().numpy()[0])
    except FileNotFoundError as e:
        print(f"Error: Image file not found - {e}")
        return None
    except AttributeError as e:
        print(f"Error: Model or image attribute error - {e}")
        return None
    except RuntimeError as e:
        print(f"Error: CUDA out of memory or runtime error - {e}")
        return None
    except Exception as e:
        import traceback
        print(f"Error calculating DINO similarity for {image_path1} and {image_path2}: {e}")
        print(f"Detailed traceback:")
        traceback.print_exc()
        return None


def calculate_ssim(image_path1, image_path2):
    """计算两张图片的SSIM（结构相似性指数）
    
    SSIM（Structural Similarity Index）是一种衡量两张图片相似度的指标，
    它从亮度、对比度和结构三个方面评估图像的相似性。
    
    相比传统的 MSE（均方误差）或 PSNR（峰值信噪比），SSIM 更符合人类视觉感知，
    能够更好地捕捉图像的结构信息。
    
    方法：
    1. 读取两张图片并转换为灰度图
    2. 确保两张图片尺寸相同（resize到较小的尺寸）
    3. 使用滑动窗口计算局部SSIM，然后计算平均值
    
    返回值范围：0-1，越大越相似
    
    用途：判断生成的网页是否保持了原设计的整体结构和视觉质量
    """
    try:
        from scipy.ndimage import uniform_filter
        
        # 读取图片并转换为灰度图
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        
        if img1 is None or img2 is None:
            print(f"Error: Failed to read images for SSIM calculation")
            return None
        
        # 转换为灰度图
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 将两张图片resize到相同尺寸（使用较小的尺寸）
        width = min(img1_gray.shape[1], img2_gray.shape[1])
        height = min(img1_gray.shape[0], img2_gray.shape[0])
        img1_gray = cv2.resize(img1_gray, (width, height), interpolation=cv2.INTER_AREA)
        img2_gray = cv2.resize(img2_gray, (width, height), interpolation=cv2.INTER_AREA)
        
        # 转换为浮点类型
        img1_gray = img1_gray.astype(np.float64)
        img2_gray = img2_gray.astype(np.float64)
        
        # SSIM参数
        C1 = (0.01 * 255) ** 2  # 亮度稳定性常数
        C2 = (0.03 * 255) ** 2  # 对比度稳定性常数
        window_size = 11  # 窗口大小
        
        # 计算均值（使用均匀滤波器模拟滑动窗口）
        mu1 = uniform_filter(img1_gray, window_size)
        mu2 = uniform_filter(img2_gray, window_size)
        
        # 计算方差和协方差
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = uniform_filter(img1_gray ** 2, window_size) - mu1_sq
        sigma2_sq = uniform_filter(img2_gray ** 2, window_size) - mu2_sq
        sigma12 = uniform_filter(img1_gray * img2_gray, window_size) - mu1_mu2
        
        # 计算SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        
        # 计算平均SSIM值
        avg_ssim = np.mean(ssim_map)
        
        return float(avg_ssim)
    except Exception as e:
        print(f"Error calculating SSIM for {image_path1} and {image_path2}: {e}")
        return None


def calculate_comprehensive_visual_score(metrics):
    """计算综合视觉得分 (Comprehensive Visual Quality Index, CVQI)
    
    将8个原子指标聚合为四个正交的视觉维度，然后加权求和得到最终得分。
    
    计算流程：
    1. 指标归一化：将距离指标转换为相似度
    2. 维度计算：计算四个维度得分
    3. 加权汇总：按权重计算最终得分
    
    返回值范围：0-1，越大越好
    
    Args:
        metrics: 包含所有指标的字典，必须包含以下键：
            - clip_similarity
            - lpips_distance
            - style_loss
            - text_similarity
            - color_histogram_similarity
            - dominant_color_similarity
            - ssim
            - dino_similarity
    
    Returns:
        tuple: (综合视觉得分, s_str, s_sty)
            - 综合视觉得分: 范围 [0, 1]
            - s_str: 结构保真度 (Structural Fidelity)
            - s_sty: 风格一致性 (Stylistic Consistency)
    """
    # 1. 指标预处理与归一化
    
    # Type A: 相似度指标 (直接保持不变)
    clip_sim = metrics['clip_similarity']
    ssim = metrics['ssim']
    text_sim = metrics['text_similarity']
    color_hist_sim = metrics['color_histogram_similarity']
    dom_color_sim = metrics['dominant_color_similarity']
    dino_sim = metrics['dino_similarity']
    
    # Type B: 距离指标 (使用指数衰减核转换为相似度)
    # LPIPS: σ = 1.0
    lpips_sim = np.exp(-metrics['lpips_distance'])
    
    # Style Loss: σ = 0.01 (乘以100放大敏感度)
    style_sim = np.exp(-100.0 * metrics['style_loss'])
    
    # 2. 维度计算
    
    # (1) 结构保真度 (Structural Fidelity, S_str)
    # 结合了像素级对齐（SSIM）、感知级布局一致性（LPIPS）和深度结构特征（DINOv2）
    # 
    s_str = (1.0 / 10.0) * ssim + (2.0 / 10.0) * lpips_sim +(7.0 / 10.0) *  dino_sim
    
    # (2) 语义对齐度 (Semantic Alignment, S_sem)
    # 衡量生成图像与Prompt描述的高层语义一致性
    s_sem = clip_sim
    
    # (3) 文本可读性 (Textual Integrity, S_txt)
    # 衡量网页中文字内容的还原程度
    s_txt = text_sim
    
    # (4) 风格一致性 (Stylistic Consistency, S_sty)
    # 衡量色彩分布与纹理风格的综合还原度
    s_sty = (1.0 / 3.0) * (color_hist_sim + dom_color_sim + style_sim)
    
    # 3. 最终汇总公式 (CVQI)
    # 权重配置：
    cvqi = (
        0.50* s_str +   # 结构最重要，保证网页不崩坏
        0.20 * s_sem +   # 语义必须正确
        0.20* s_txt +   # 文字内容是网页的核心
        0.10 * s_sty     # 色差在可接受范围内即可
    )
    
    # 确保结果在 [0, 1] 范围内
    cvqi = max(0.0, min(1.0, cvqi))
    
    return float(cvqi), float(s_str), float(s_sty)


def calculate_dominant_color_similarity(image_path1, image_path2, n_colors=5):
    """计算两张图片的主色调分布相似度
    
    使用K-means聚类提取每张图片的前N个主导颜色，然后计算它们在色轮上的距离。
    如果两个网页都以"科技蓝"为主，它们的相似度就高。
    
    方法：
    1. 对每张图片使用K-means聚类提取前N个主导颜色及其占比
    2. 将RGB颜色转换为HSV，主要关注色相（Hue）
    3. 使用贪心算法寻找最优的一对一匹配，计算加权相似度
    
    返回值范围：0-1，越大越相似
    
    用途：判断生成的网页是否保持了原设计的主色调风格（例如：都是科技蓝风格）
    
    算法说明：
    - 使用贪心匹配策略（类似匈牙利算法的简化版），确保每个颜色只被匹配一次
    - 匹配权重综合考虑色相相似度和颜色占比
    - 结果自然地在0-1范围内，无需裁剪
    """
    try:
        # 提取两张图片的主导颜色
        colors1, percentages1 = extract_dominant_colors(image_path1, n_colors)
        colors2, percentages2 = extract_dominant_colors(image_path2, n_colors)
        
        if colors1 is None or colors2 is None:
            return None
        
        # 将RGB颜色转换为HSV（关注色相）
        hues1 = np.array([rgb_to_hsv(color)[0] for color in colors1])
        hues2 = np.array([rgb_to_hsv(color)[0] for color in colors2])
        
        # 计算色相距离矩阵
        # distances[i, j] 表示图片1的第i个主导颜色与图片2的第j个主导颜色的色相距离
        distances = np.zeros((n_colors, n_colors))
        for i in range(n_colors):
            for j in range(n_colors):
                distances[i, j] = calculate_hue_distance(hues1[i], hues2[j])
        
        # 将距离转换为相似度（0-180映射到1-0）
        similarities = 1 - (distances / 180.0)
        
        # 贪心匹配算法：寻找最优的一对一匹配
        # 思路：每次选择相似度最高的未匹配对，确保每个颜色只被匹配一次
        matched1 = set()  # 已匹配的图片1的颜色索引
        matched2 = set()  # 已匹配的图片2的颜色索引
        
        # 按相似度降序排序所有可能的匹配对
        matches = []
        for i in range(n_colors):
            for j in range(n_colors):
                matches.append((similarities[i, j], i, j))
        
        # 按相似度降序排序
        matches.sort(key=lambda x: x[0], reverse=True)
        
        # 执行贪心匹配
        total_weighted_similarity = 0.0
        total_weight = 0.0
        
        for similarity, i, j in matches:
            # 如果两个颜色都未被匹配，则建立匹配
            if i not in matched1 and j not in matched2:
                # 使用两个颜色的占比的几何平均作为权重
                # 这样可以确保双向对称
                weight = np.sqrt(percentages1[i] * percentages2[j])
                total_weighted_similarity += weight * similarity
                total_weight += weight
                matched1.add(i)
                matched2.add(j)
        
        # 计算加权平均相似度
        if total_weight > 0:
            final_similarity = total_weighted_similarity / total_weight
        else:
            final_similarity = 0.0
        
        # 理论上结果应该在[0, 1]范围内，但添加边界检查以防数值计算误差
        final_similarity = max(0.0, min(1.0, final_similarity))
        
        return float(final_similarity)
    except Exception as e:
        print(f"Error calculating dominant color similarity for {image_path1} and {image_path2}: {e}")
        return None

def parse_folder_name(folder_name):
    """解析文件夹名称，提取model_name和item
    
    匹配格式: {model_name}_{item}
    例如: doubao-seed-1-6_Apple_1766504959 -> model_name=doubao-seed-1-6, item=Apple_1766504959
    例如: Doubao-Seed-1.8_Discord_1766509759 -> model_name=Doubao-Seed-1.8, item=Discord_1766509759
    """
    # 从第一个下划线处分割，前面的部分是model_name，后面的是item
    parts = folder_name.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def find_matching_folders(build_dir, dataset_dir):
    """找出匹配的文件夹对"""
    matching_pairs = []
    
    build_folders = os.listdir(build_dir)
    
    for build_folder in build_folders:
        build_folder_path = os.path.join(build_dir, build_folder)
        if not os.path.isdir(build_folder_path):
            continue
        
        model_name, item = parse_folder_name(build_folder)
        if model_name is None or item is None:
            continue
        
        # 在dataset_dir中寻找匹配的item文件夹
        dataset_folder_path = os.path.join(dataset_dir, item)
        if os.path.exists(dataset_folder_path) and os.path.isdir(dataset_folder_path):
            matching_pairs.append((build_folder_path, dataset_folder_path, build_folder))
    
    return matching_pairs


def get_step_images(folder_path, start_from=1):
    """获取从指定步数开始的step图片"""
    step_images = {}
    
    files = os.listdir(folder_path)
    step_pattern = re.compile(r'^step_(\d+)\.png$')
    
    for file in files:
        match = step_pattern.match(file)
        if match:
            step_num = int(match.group(1))
            if step_num >= start_from:
                step_images[step_num] = file
    
    # 按步数排序
    return dict(sorted(step_images.items()))


def process_folder_pair(model, preprocess, lpips_model, vgg_model, dino_model, dino_preprocess, device, build_folder_path, dataset_folder_path, build_folder_name):
    """处理一对文件夹，计算CLIP、LPIPS、Style Loss、Text Similarity、Color Histogram、Dominant Color、DINO、SSIM和综合视觉得分并更新metadata.json"""
    print(f"\nProcessing: {build_folder_name}")
    
    metadata_path = os.path.join(build_folder_path, "metadata.json")
    
    # 检查metadata.json是否存在
    if not os.path.exists(metadata_path):
        print(f"  Warning: metadata.json not found in {build_folder_path}")
        return
    
    # 读取metadata.json
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 检查sequence是否存在
    if 'sequence' not in metadata:
        print(f"  Warning: 'sequence' not found in metadata.json")
        return
    
    # 读取dataset文件夹的metadata.json以获取total_steps
    dataset_metadata_path = os.path.join(dataset_folder_path, "metadata.json")
    dataset_total_steps = None
    if os.path.exists(dataset_metadata_path):
        try:
            with open(dataset_metadata_path, 'r', encoding='utf-8') as f:
                dataset_metadata = json.load(f)
                dataset_total_steps = dataset_metadata.get('total_steps')
                print(f"  Dataset total_steps: {dataset_total_steps}")
        except Exception as e:
            print(f"  Warning: Failed to read dataset metadata.json: {e}")
    else:
        print(f"  Warning: dataset metadata.json not found at {dataset_metadata_path}")
    
    # 获取build文件夹中的step图片（从step_01开始）
    build_step_images = get_step_images(build_folder_path, start_from=1)
    print(f"  Build folder step images: {list(build_step_images.keys())}")
    
    # 获取dataset文件夹中的step图片（从step_01开始）
    dataset_step_images = get_step_images(dataset_folder_path, start_from=1)
    print(f"  Dataset folder step images: {list(dataset_step_images.keys())}")
    
    # 以build文件夹的图片数量为准
    updated_sequence = []
    
    for step_entry in metadata['sequence']:
        if 'image' not in step_entry:
            updated_sequence.append(step_entry)
            continue
        
        image_filename = step_entry['image']
        
        # 从图片文件名中提取步数
        match = re.match(r'^step_(\d+)\.png$', image_filename)
        if not match:
            updated_sequence.append(step_entry)
            continue
        
        step_num = int(match.group(1))
        
        # 检查两个文件夹中是否都有对应的图片
        if step_num in build_step_images and step_num in dataset_step_images:
            build_image_path = os.path.join(build_folder_path, build_step_images[step_num])
            dataset_image_path = os.path.join(dataset_folder_path, dataset_step_images[step_num])
            
            # 将相似度添加到image字段后面
            step_entry_with_similarity = step_entry.copy()
            
            # 检查是否需要计算CLIP相似度
            if 'clip_similarity' not in step_entry:
                clip_similarity = calculate_clip_similarity(model, preprocess, device, build_image_path, dataset_image_path)
                step_entry_with_similarity['clip_similarity'] = clip_similarity
                if clip_similarity is not None:
                    print(f"  Step {step_num}: CLIP similarity calculated = {clip_similarity:.4f}")
                else:
                    print(f"  Step {step_num}: CLIP similarity calculation failed")
            else:
                clip_similarity = step_entry['clip_similarity']
                step_entry_with_similarity['clip_similarity'] = clip_similarity
                print(f"  Step {step_num}: CLIP similarity already exists = {clip_similarity}, skipping")
            
            # 检查是否需要计算LPIPS距离
            if 'lpips_distance' not in step_entry:
                lpips_distance = calculate_lpips_distance(lpips_model, device, build_image_path, dataset_image_path)
                step_entry_with_similarity['lpips_distance'] = lpips_distance
                if lpips_distance is not None:
                    print(f"  Step {step_num}: LPIPS distance calculated = {lpips_distance:.4f}")
                else:
                    print(f"  Step {step_num}: LPIPS distance calculation failed")
            else:
                lpips_distance = step_entry['lpips_distance']
                step_entry_with_similarity['lpips_distance'] = lpips_distance
                print(f"  Step {step_num}: LPIPS distance already exists = {lpips_distance}, skipping")
            
            # 检查是否需要计算Style Loss
            if 'style_loss' not in step_entry:
                style_loss = calculate_style_loss(vgg_model, device, build_image_path, dataset_image_path)
                step_entry_with_similarity['style_loss'] = style_loss
                if style_loss is not None:
                    print(f"  Step {step_num}: Style Loss calculated = {style_loss:.4f}")
                else:
                    print(f"  Step {step_num}: Style Loss calculation failed")
            else:
                style_loss = step_entry['style_loss']
                step_entry_with_similarity['style_loss'] = style_loss
                print(f"  Step {step_num}: Style Loss already exists = {style_loss}, skipping")
            
            # 检查是否需要计算Text Similarity
            if 'text_similarity' not in step_entry:
                text_similarity = calculate_text_similarity(build_image_path, dataset_image_path)
                step_entry_with_similarity['text_similarity'] = text_similarity
                if text_similarity is not None:
                    print(f"  Step {step_num}: Text Similarity calculated = {text_similarity:.4f}")
                else:
                    print(f"  Step {step_num}: Text Similarity calculation failed")
            else:
                text_similarity = step_entry['text_similarity']
                step_entry_with_similarity['text_similarity'] = text_similarity
                print(f"  Step {step_num}: Text Similarity already exists = {text_similarity}, skipping")
            
            # 检查是否需要计算Color Histogram Similarity
            if 'color_histogram_similarity' not in step_entry:
                color_histogram_similarity = calculate_color_histogram_similarity(build_image_path, dataset_image_path)
                step_entry_with_similarity['color_histogram_similarity'] = color_histogram_similarity
                if color_histogram_similarity is not None:
                    print(f"  Step {step_num}: Color Histogram Similarity calculated = {color_histogram_similarity:.4f}")
                else:
                    print(f"  Step {step_num}: Color Histogram Similarity calculation failed")
            else:
                color_histogram_similarity = step_entry['color_histogram_similarity']
                step_entry_with_similarity['color_histogram_similarity'] = color_histogram_similarity
                print(f"  Step {step_num}: Color Histogram Similarity already exists = {color_histogram_similarity}, skipping")
            
            # 检查是否需要计算Dominant Color Similarity
            if 'dominant_color_similarity' not in step_entry:
                dominant_color_similarity = calculate_dominant_color_similarity(build_image_path, dataset_image_path)
                step_entry_with_similarity['dominant_color_similarity'] = dominant_color_similarity
                if dominant_color_similarity is not None:
                    print(f"  Step {step_num}: Dominant Color Similarity calculated = {dominant_color_similarity:.4f}")
                else:
                    print(f"  Step {step_num}: Dominant Color Similarity calculation failed")
            else:
                dominant_color_similarity = step_entry['dominant_color_similarity']
                step_entry_with_similarity['dominant_color_similarity'] = dominant_color_similarity
                print(f"  Step {step_num}: Dominant Color Similarity already exists = {dominant_color_similarity}, skipping")
            
            # 删除layout_similarity字段（如果存在）
            if 'layout_similarity' in step_entry_with_similarity:
                del step_entry_with_similarity['layout_similarity']
                print(f"  Step {step_num}: Removed deprecated layout_similarity field")
            
            # 检查是否需要计算DINO相似度
            if 'dino_similarity' not in step_entry:
                dino_similarity = calculate_dino_similarity(dino_model, dino_preprocess, device, build_image_path, dataset_image_path)
                step_entry_with_similarity['dino_similarity'] = dino_similarity
                if dino_similarity is not None:
                    print(f"  Step {step_num}: DINO similarity calculated = {dino_similarity:.4f}")
                else:
                    print(f"  Step {step_num}: DINO similarity calculation failed")
            else:
                dino_similarity = step_entry['dino_similarity']
                step_entry_with_similarity['dino_similarity'] = dino_similarity
                print(f"  Step {step_num}: DINO similarity already exists = {dino_similarity}, skipping")
            
            # 检查是否需要计算SSIM
            if 'ssim' not in step_entry:
                ssim = calculate_ssim(build_image_path, dataset_image_path)
                step_entry_with_similarity['ssim'] = ssim
                if ssim is not None:
                    print(f"  Step {step_num}: SSIM calculated = {ssim:.4f}")
                else:
                    print(f"  Step {step_num}: SSIM calculation failed")
            else:
                ssim = step_entry['ssim']
                step_entry_with_similarity['ssim'] = ssim
                print(f"  Step {step_num}: SSIM already exists = {ssim}, skipping")
            
            # 检查是否可以计算综合视觉得分 (CVQI)
            # CVQI 每次运行都会重新计算，不使用增量模式
            all_metrics_exist = all(key in step_entry_with_similarity for key in [
                'clip_similarity', 'lpips_distance', 'style_loss', 'text_similarity',
                'color_histogram_similarity', 'dominant_color_similarity', 'ssim', 'dino_similarity'
            ])
            
            if all_metrics_exist:
                # 构建指标字典
                metrics = {
                    'clip_similarity': step_entry_with_similarity['clip_similarity'],
                    'lpips_distance': step_entry_with_similarity['lpips_distance'],
                    'style_loss': step_entry_with_similarity['style_loss'],
                    'text_similarity': step_entry_with_similarity['text_similarity'],
                    'color_histogram_similarity': step_entry_with_similarity['color_histogram_similarity'],
                    'dominant_color_similarity': step_entry_with_similarity['dominant_color_similarity'],
                    'ssim': step_entry_with_similarity['ssim'],
                    'dino_similarity': step_entry_with_similarity['dino_similarity']
                }
                
                # 检查是否有None值
                if all(metrics[key] is not None for key in metrics):
                    cvqi, s_str, s_sty = calculate_comprehensive_visual_score(metrics)
                    step_entry_with_similarity['comprehensive_visual_score'] = cvqi
                    step_entry_with_similarity['s_str'] = s_str
                    step_entry_with_similarity['s_sty'] = s_sty
                    print(f"  Step {step_num}: CVQI = {cvqi:.4f}, S_str = {s_str:.4f}, S_sty = {s_sty:.4f}")
                else:
                    print(f"  Step {step_num}: Cannot calculate CVQI - some metrics are None")
            else:
                print(f"  Step {step_num}: Cannot calculate CVQI - not all metrics available yet")
            
            updated_sequence.append(step_entry_with_similarity)
        else:
            # 如果缺少图片，保留原始条目
            print(f"  Step {step_num}: Missing image in one of the folders")
            updated_sequence.append(step_entry)
    
    # 更新metadata
    metadata['sequence'] = updated_sequence
    
    # 计算 visual_score: 使用线性递增权重计算加权平均
    # 权重随着步骤增加而线性递增，后面的步骤权重更高
    # 以 dataset 的 total_steps 为准，多了的步骤不参与计算
    if dataset_total_steps and dataset_total_steps > 0:
        total_planned_steps = dataset_total_steps
        weighted_sum = 0.0
        total_weight = 0.0
        executed_steps = set()  # 记录已执行的步骤编号
        
        # 遍历已执行的步骤，累加加权分数
        # 只考虑步骤编号 <= dataset_total_steps 的步骤，超过这个范围的步骤不参与计算
        for step_entry in updated_sequence:
            if 'comprehensive_visual_score' in step_entry and step_entry['comprehensive_visual_score'] is not None:
                # 获取步骤编号（从 image 字段提取）
                image_filename = step_entry.get('image', '')
                match = re.match(r'^step_(\d+)\.png$', image_filename)
                if match:
                    step_num = int(match.group(1))
                    # 只计算 dataset_total_steps 范围内的步骤
                    if step_num <= dataset_total_steps:
                        # 线性递增权重：步骤越后，权重越高
                        # 例如，5个步骤：权重分别为 1, 2, 3, 4, 5
                        weight = step_num
                        cvqi_value = step_entry['comprehensive_visual_score']
                        
                        weighted_sum += cvqi_value * weight
                        total_weight += weight
                        executed_steps.add(step_num)
        
        # 检查是否有缺失的步骤
        missing_steps = []
        for step_num in range(1, total_planned_steps + 1):
            if step_num not in executed_steps:
                missing_steps.append(step_num)
                # 缺失的步骤 CVQI 记为 0，但权重仍然计入
                # 例如：如果总计划5步，第5步缺失，则总权重仍然是 1+2+3+4+5 = 15
                total_weight += step_num
        
        # 计算加权平均
        if total_weight > 0:
            visual_score = weighted_sum / total_weight
            metadata['visual_score'] = float(visual_score)
            if missing_steps:
                print(f"  Visual Score calculated (weighted average): {weighted_sum:.4f} (weighted sum) / {total_weight:.4f} (total weight) = {visual_score:.4f}")
                print(f"  Missing steps: {missing_steps} (treated as CVQI=0 with respective weights)")
            else:
                print(f"  Visual Score calculated (weighted average): {weighted_sum:.4f} (weighted sum) / {total_weight:.4f} (total weight) = {visual_score:.4f}")
        else:
            print(f"  Warning: No valid weight found, skipping visual_score calculation")
    else:
        print(f"  Warning: total_planned_steps not found or is 0, skipping visual_score calculation")
    
    # 写回文件
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  Updated metadata.json with CLIP, LPIPS, Style Loss, Text, Color Histogram, Dominant Color, DINO, SSIM, CVQI, S_str, S_sty and visual_score")


def main():
    # 定义目录路径
    base_dir = Path(__file__).parent
    build_dir = base_dir / "build_dataset_multi_turn"
    dataset_dir = base_dir / "dataset_multi_turn"
    
    print("=" * 80)
    print("CLIP, LPIPS, Style Loss, Text, Color Histogram, Dominant Color, DINO, SSIM, CVQI, S_str, S_sty Comparison")
    print("=" * 80)
    print(f"Build directory: {build_dir}")
    print(f"Dataset directory: {dataset_dir}")
    print()
    
    # 加载CLIP模型
    print("Loading CLIP model...")
    clip_model, preprocess, device = load_clip_model()
    print(f"CLIP model loaded on {device}")
    
    # 加载LPIPS模型
    print("Loading LPIPS model...")
    lpips_model, device = load_lpips_model()
    print(f"LPIPS model loaded on {device}")
    
    # 加载VGG19模型（用于Style Loss）
    print("Loading VGG19 model for Style Loss...")
    vgg_model, device = load_vgg_model()
    print(f"VGG19 model loaded on {device}")
    
    # 加载DINOv2模型（用于结构相似度）
    print("Loading DINOv2 model for structural similarity...")
    dino_model, dino_preprocess, device = load_dino_model()
    if dino_model is not None:
        print(f"DINOv2 model loaded on {device}")
    else:
        print("DINOv2 model not available, structural similarity will be skipped")
    
    # 查找匹配的文件夹对
    print("\nFinding matching folder pairs...")
    matching_pairs = find_matching_folders(build_dir, dataset_dir)
    print(f"Found {len(matching_pairs)} matching pairs")
    
    # 处理每一对文件夹
    for i, (build_folder_path, dataset_folder_path, build_folder_name) in enumerate(matching_pairs, 1):
        print(f"\n{'='*80}")
        print(f"Processing pair {i}/{len(matching_pairs)}")
        process_folder_pair(clip_model, preprocess, lpips_model, vgg_model, dino_model, dino_preprocess, device, build_folder_path, dataset_folder_path, build_folder_name)
    
    # 统计每个 model_name 的平均 visual_score
    print("\n" + "=" * 80)
    print("Visual Score Statistics by Model Name")
    print("=" * 80)
    
    # 收集所有 build 文件夹，按 model_name 和 item 组织
    model_item_folders = {}
    all_items = set()
    
    for build_folder in os.listdir(build_dir):
        build_folder_path = os.path.join(build_dir, build_folder)
        if not os.path.isdir(build_folder_path):
            continue
        
        # 解析 model_name 和 item
        model_name, item = parse_folder_name(build_folder)
        if model_name is None or item is None:
            continue
        
        # 按模型名称分组
        if model_name not in model_item_folders:
            model_item_folders[model_name] = {}
        
        model_item_folders[model_name][item] = (build_folder, build_folder_path)
        all_items.add(item)
    
    # 使用全部item进行统计
    all_items_list = list(all_items)
    print(f"\nUsing all {len(all_items_list)} items from the dataset")
    
    # 遍历每个模型，统计所有item
    model_scores_all = {}
    model_counts_all = {}
    model_scores_valid = {}
    model_counts_valid = {}
    
    for model_name, item_folders in model_item_folders.items():
        model_scores_all[model_name] = 0.0
        model_counts_all[model_name] = 0
        model_scores_valid[model_name] = 0.0
        model_counts_valid[model_name] = 0
        
        # 统计所有 item
        for item in all_items_list:
            if item not in item_folders:
                # 该模型在这个 item 上没有文件夹，算为0（方式1）
                model_scores_all[model_name] += 0.0
                model_counts_all[model_name] += 1
                continue
            
            build_folder, build_folder_path = item_folders[item]
            
            # 读取 metadata.json
            metadata_path = os.path.join(build_folder_path, "metadata.json")
            if not os.path.exists(metadata_path):
                print(f"  Warning: metadata.json not found in {build_folder}")
                # 算为0
                model_scores_all[model_name] += 0.0
                model_counts_all[model_name] += 1
                continue
            
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"  Warning: Failed to read metadata.json from {build_folder}: {e}")
                # 算为0
                model_scores_all[model_name] += 0.0
                model_counts_all[model_name] += 1
                continue
            
            # 检查 sequence 是否有效（非空且有 visual_score）
            has_valid_sequence = 'sequence' in metadata and len(metadata['sequence']) > 0 and 'visual_score' in metadata
            
            # 方式1: 计算所有 sequence（包括空的，空的算为0）
            visual_score_all = metadata.get('visual_score', 0.0) if has_valid_sequence else 0.0
            
            model_scores_all[model_name] += visual_score_all
            model_counts_all[model_name] += 1
            
            # 方式2: 不计算空的 sequence（只计算有效的）
            if has_valid_sequence:
                visual_score_valid = metadata['visual_score']
                model_scores_valid[model_name] += visual_score_valid
                model_counts_valid[model_name] += 1
    
    # 计算并输出方式1的结果（计算所有 sequence）
    print("\n【方式1: 计算所有 sequence（包括空的，空的算为0）】")
    if model_scores_all:
        sorted_models = sorted(model_scores_all.keys())
        
        for model_name in sorted_models:
            avg_score = model_scores_all[model_name] / model_counts_all[model_name]
            print(f"  {model_name}: {avg_score:.4f} (sum: {model_scores_all[model_name]:.4f}, count: {model_counts_all[model_name]})")
        
        total_sum = sum(model_scores_all.values())
        total_count = sum(model_counts_all.values())
        overall_avg = total_sum / total_count
        print(f"\n  Overall Average: {overall_avg:.4f} (sum: {total_sum:.4f}, count: {total_count})")
    else:
        print("  No model data found")
    
    # 计算并输出方式2的结果（不计算空的 sequence）
    print("\n【方式2: 不计算空的 sequence（只计算有效的）】")
    if model_scores_valid:
        sorted_models = sorted(model_scores_valid.keys())
        
        for model_name in sorted_models:
            avg_score = model_scores_valid[model_name] / model_counts_valid[model_name]
            print(f"  {model_name}: {avg_score:.4f} (sum: {model_scores_valid[model_name]:.4f}, count: {model_counts_valid[model_name]})")
        
        total_sum = sum(model_scores_valid.values())
        total_count = sum(model_counts_valid.values())
        overall_avg = total_sum / total_count
        print(f"\n  Overall Average: {overall_avg:.4f} (sum: {total_sum:.4f}, count: {total_count})")
    else:
        print("  No valid model data found")
    
    # 统计每个 model_name 的平均 success_ratio
    print("\n" + "=" * 80)
    print("Success Ratio Statistics by Model Name")
    print("=" * 80)
    
    model_success_all = {}
    model_counts_success_all = {}
    model_success_valid = {}
    model_counts_success_valid = {}
    
    for model_name, item_folders in model_item_folders.items():
        model_success_all[model_name] = 0.0
        model_counts_success_all[model_name] = 0
        model_success_valid[model_name] = 0.0
        model_counts_success_valid[model_name] = 0
        
        # 统计所有 item
        for item in all_items_list:
            if item not in item_folders:
                # 该模型在这个 item 上没有文件夹，算为0（方式1）
                model_success_all[model_name] += 0.0
                model_counts_success_all[model_name] += 1
                continue
            
            build_folder, build_folder_path = item_folders[item]
            
            # 读取 metadata.json
            metadata_path = os.path.join(build_folder_path, "metadata.json")
            if not os.path.exists(metadata_path):
                # 算为0
                model_success_all[model_name] += 0.0
                model_counts_success_all[model_name] += 1
                continue
            
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                # 算为0
                model_success_all[model_name] += 0.0
                model_counts_success_all[model_name] += 1
                continue
            
            # 检查 sequence 是否有效（非空）
            has_valid_sequence = 'sequence' in metadata and len(metadata['sequence']) > 0
            
            # 获取 success_ratio
            success_ratio = metadata.get('success_ratio', 0.0)
            
            # 方式1: 计算所有 success_ratio（包括空的，空的算为0）
            model_success_all[model_name] += success_ratio
            model_counts_success_all[model_name] += 1
            
            # 方式2: 不计算空的 sequence（只计算有效的）
            if has_valid_sequence:
                model_success_valid[model_name] += success_ratio
                model_counts_success_valid[model_name] += 1
    
    # 计算并输出 success_ratio 方式1的结果（计算所有）
    print("\n【方式1: 计算所有 success_ratio（包括空的，空的算为0）】")
    if model_success_all:
        sorted_models = sorted(model_success_all.keys())
        
        for model_name in sorted_models:
            avg_ratio = model_success_all[model_name] / model_counts_success_all[model_name]
            print(f"  {model_name}: {avg_ratio:.4f} (sum: {model_success_all[model_name]:.4f}, count: {model_counts_success_all[model_name]})")
        
        total_sum = sum(model_success_all.values())
        total_count = sum(model_counts_success_all.values())
        overall_avg = total_sum / total_count
        print(f"\n  Overall Average: {overall_avg:.4f} (sum: {total_sum:.4f}, count: {total_count})")
    else:
        print("  No model data found")
    
    # 计算并输出 success_ratio 方式2的结果（不计算空的 sequence）
    print("\n【方式2: 不计算空的 sequence（只计算有效的）】")
    if model_success_valid:
        sorted_models = sorted(model_success_valid.keys())
        
        for model_name in sorted_models:
            avg_ratio = model_success_valid[model_name] / model_counts_success_valid[model_name]
            print(f"  {model_name}: {avg_ratio:.4f} (sum: {model_success_valid[model_name]:.4f}, count: {model_counts_success_valid[model_name]})")
        
        total_sum = sum(model_success_valid.values())
        total_count = sum(model_counts_success_valid.values())
        overall_avg = total_sum / total_count
        print(f"\n  Overall Average: {overall_avg:.4f} (sum: {total_sum:.4f}, count: {total_count})")
    else:
        print("  No valid model data found")
    
    print("=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
