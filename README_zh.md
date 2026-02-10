<div align="center">

# MultiInteract-Bench

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

**评测大模型根据图像序列还原网页交互能力的 Benchmark**

---

[📚 文档](#安装) • [🚀 快速开始](#快速开始) • [📊 评测指标](#评测指标) • [📖 使用示例](#使用示例)

</div>

## 📋 概述

MultiInteract-Bench 是一个综合评估框架，旨在评估多模态大语言模型在复现网页交互方面的能力。它提供了标准化的评估流程，包括：

- **模型复现**：使用大语言模型根据图像序列重新生成网页组件
- **自动化测试**：自动捕获生成网页的截图
- **视觉指标**：计算多种评估指标以评估视觉保真度
- **结果分析**：对不同模型的统计分析和比较

## 📁 项目结构

```
MultiInteract-Bench/
├── frontend/                    # 前端 Vue.js 应用
│   ├── src/                    # 源代码
│   │   ├── components/         # Vue 组件
│   │   │   └── dataset/       # 用于测试的生成组件
│   │   └── ...
│   ├── public/                 # 静态资源
│   └── ...
├── backend/                    # 后端评测脚本（原始版本）
│   ├── build.py               # 模型复现脚本
│   ├── recorder_build.py      # 截图捕获脚本
│   ├── compare_visual.py      # 视觉指标计算
│   └── calculate_model_metrics.py  # 统计分析
├── scripts/                    # 统一评测脚本（推荐使用）
│   └── benchmark.py          # 主评测脚本（4个阶段）
├── dataset_examples/           # 数据集示例文件夹
│   ├── Spotify_1766618072/   # 示例任务1
│   └── Stripe_1766502847/     # 示例任务2
├── outputs/                    # 输出目录
│   ├── reproduced_configs/     # 生成的配置文件
│   ├── build_dataset_multi_turn/  # 捕获的截图
│   └── model_metrics_summary.csv  # 评测结果
├── requirements.txt            # Python 依赖
├── README.md                  # 英文文档
├── README_zh.md               # 本文档（中文文档）
└── PROJECT_STRUCTURE.md       # 项目结构详细说明
```

## 📊 评测指标

MultiInteract-Bench 使用 8 个原子指标来评估生成网页的质量：

### 1. CLIP 相似度 (CLIP Similarity)

测量生成图像与参考图像之间的高级语义对齐度。

- **范围**：0-1（越高越好）
- **模型**：OpenAI CLIP (ViT-B/32)

### 2. LPIPS 距离 (LPIPS Distance)

捕获人类视觉差异的感知距离度量。

- **范围**：0-∞（越低越好）
- **模型**：基于 VGG 的 LPIPS

### 3. 风格损失 (Style Loss)

使用 Gram 矩阵评估艺术风格一致性。

- **范围**：0-∞（越低越好）
- **模型**：VGG19 神经风格迁移特征

### 4. 文本相似度 (Text Similarity)

使用 OCR 和 LCS 测量文本内容保真度。

- **范围**：0-1（越高越好）
- **方法**：最长公共子序列 (LCS)

### 5. 颜色直方图相似度 (Color Histogram Similarity)

比较 RGB 通道的整体颜色分布。

- **范围**：0-1（越高越好）
- **方法**：直方图相关系数

### 6. 主色调相似度 (Dominant Color Similarity)

使用 K-means 聚类评估主色一致性。

- **范围**：0-1（越高越好）
- **方法**：HSV 色相距离与贪心匹配

### 7. DINO 相似度 (DINO Similarity)

评估结构布局和高级视觉特征。

- **范围**：0-1（越高越好）
- **模型**：Meta DINOv2 (ViT-Base)

### 8. 结构相似度 (SSIM)

像素级结构保真度的结构相似度指数。

- **范围**：0-1（越高越好）
- **方法**：滑动窗口 SSIM

### 综合视觉质量指数 (CVQI)

8 个原子指标被聚合为一个综合得分（CVQI），用于评估整体视觉质量，涵盖结构、风格、语义和文本等多个维度。CVQI 得分范围为 0-1（越高越好）。

**Visual Score**：每个任务的最终评测指标，通过计算所有步骤的 CVQI 得分的加权平均值得出，其中后面的步骤获得更高的权重，以强调交互的最终状态。

---

## 🛠️ 安装配置

### 环境要求

- **Python**: 3.8+
- **Node.js**: 16+（用于前端）
- **CUDA-capable GPU**: 推荐以加快处理速度
- **Conda**: 推荐用于环境管理

### 方案一：使用 Conda（推荐）

#### 第一步：创建并激活 Conda 环境

```bash
# 创建新的 conda 环境
conda create -n multinteract python=3.10 -y

# 激活环境
conda activate multinteract
```

#### 第二步：安装带 CUDA 支持的 PyTorch

```bash
# 对于 CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 对于 CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 仅 CPU 版本
conda install pytorch torchvision cpuonly -c pytorch
```

#### 第三步：安装 Python 依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `torch` & `torchvision`: 深度学习框架
- `openai-clip`: 用于语义相似度的视觉语言模型
- `lpips`: 感知图像相似度指标
- `opencv-python`: 图像处理
- `scikit-learn`: 机器学习工具
- `easyocr`: 用于文本提取的 OCR
- `playwright`: 用于截图捕获的浏览器自动化

#### 第四步：安装前端依赖

```bash
cd frontend
npm install
cd ..
```

#### 第五步：安装 Playwright 浏览器

```bash
npx playwright install chromium
```

### 方案二：仅使用 pip

如果不想使用 conda，可以直接使用 pip 安装依赖：

```bash
# 创建虚拟环境（可选但推荐）
python -m venv multinteract_env
source multinteract_env/bin/activate  # Windows: multinteract_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装前端依赖
cd frontend
npm install
cd ..

# 安装 Playwright 浏览器
npx playwright install chromium
```

### 验证安装

验证安装是否成功：

```bash
# 检查 Python 版本
python --version

# 检查 PyTorch 安装
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查其他依赖
python -c "import clip, lpips, cv2; print('所有依赖安装成功！')"
```

---

## 📊 数据集

<details>
<summary>📥 <strong>下载完整数据集</strong></summary>

完整的 MultiInteract-Bench 数据集可在 HuggingFace 上获取：

[![下载数据集](https://img.shields.io/badge/Download-HuggingFace-yellow.svg)](https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench/resolve/main/dataset_multi_turn.zip)

```bash
# 下载数据集 zip 文件
wget https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench/resolve/main/dataset_multi_turn.zip

# 或使用 curl
curl -L -o dataset_multi_turn.zip https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench/resolve/main/dataset_multi_turn.zip

# 解压数据集
unzip dataset_multi_turn.zip -d ./

# 验证下载
ls dataset_multi_turn/ | head -5
```

**注意**：解压后，数据集将位于 `./dataset_multi_turn/` 目录中。

</details>

### 数据集结构

每个数据集任务遵循以下结构：

```
task_name/
├── metadata.json           # 任务元数据和步骤描述
├── step_00.png           # 初始状态（任何交互之前）
├── step_01.png           # 步骤1交互之后的状态
├── step_02.png           # 步骤2交互之后的状态
└── ...                   # 额外的交互步骤
```

### 示例任务

`dataset_examples/` 中包含两个示例任务：

| 任务名称 | 描述 | 步骤数 |
|---------|------|--------|
| `Spotify_1766618072/` | 音乐播放器界面 | 6 步 |
| `Stripe_1766502847/` | 支付表单界面 | 6 步 |

---

## 🚀 快速开始

本节提供快速上手指南，帮助您开始使用 MultiInteract-Bench。详细的配置选项请参阅[使用示例](#使用示例)。

### 第一阶段：模型复现（构建阶段）

使用大语言模型从数据集生成 Vue 组件：

```bash
python scripts/benchmark.py build \
    --dataset-dir ./dataset_multi_turn \
    --output-vue-dir ./frontend/src/components/dataset \
    --output-config-dir ./outputs/reproduced_configs \
    --limit 10  # 用于测试，只处理前10个任务
```

**注意**：运行前，请编辑 `scripts/benchmark.py` 并在 `MODEL_LIST` 变量中配置您的模型 API。

### 第二阶段：启动前端服务

```bash
cd frontend
npm run dev
```

前端将在 `http://localhost:1234` 上可用

### 第三阶段：自动化截图捕获（录制阶段）

捕获生成网页的截图：

```bash
python scripts/benchmark.py record \
    --base-url http://localhost:1234 \
    --config-dir ./outputs/reproduced_configs \
    --output-dir ./outputs/build_dataset_multi_turn \
    --headless
```

### 第四阶段：计算视觉指标（比较阶段）

计算评估指标：

```bash
python scripts/benchmark.py compare \
    --build-dir ./outputs/build_dataset_multi_turn \
    --dataset-dir ./dataset_multi_turn
```

这将计算每个步骤的所有 8 个指标并更新元数据文件。

### 第五阶段：统计分析（指标阶段）

生成汇总统计：

```bash
python scripts/benchmark.py metrics \
    --base-dir ./outputs/build_dataset_multi_turn \
    --output-csv ./outputs/model_metrics_summary.csv
```

结果将保存到 `model_metrics_summary.csv`。

---

## 📖 使用示例

### 使用特定模型测试

```bash
python scripts/benchmark.py build \
    --models model-name-1 model-name-2
```

### 测试特定任务

```bash
python scripts/benchmark.py record \
    --task-id model-name_TaskName_1234567890
```

### 测试特定模型前缀的所有任务

```bash
python scripts/benchmark.py record \
    --model-prefix gpt-4o
```

### 跳过已处理的任务

```bash
python scripts/benchmark.py build --skip-processed
```

### 重新运行所有任务

```bash
python scripts/benchmark.py build --no-skip-processed
```

### 限制处理数量（用于测试）

```bash
python scripts/benchmark.py build --limit 5
```

---

## 📤 输出格式

### Metadata.json 结构

在比较阶段后，每个任务的 metadata.json 包含：

```json
{
  "id": "model-name_TaskName_1234567890",
  "total_planned_steps": 5,
  "successful_steps": 5,
  "success_ratio": 1.0,
  "visual_score": 0.8543,
  "sequence": [
    {
      "step_index": 1,
      "image": "step_01.png",
      "clip_similarity": 0.9234,
      "lpips_distance": 0.1234,
      "style_loss": 0.0012,
      "text_similarity": 0.8765,
      "color_histogram_similarity": 0.9234,
      "dominant_color_similarity": 0.8901,
      "dino_similarity": 0.9456,
      "ssim": 0.8765,
      "comprehensive_visual_score": 0.9123,
      "s_str": 0.8901,
      "s_sty": 0.9234
    },
    ...
  ]
}
```

### CSV 输出格式

指标汇总 CSV 包含：

- **Model**: 模型名称
- **Count**: 评估的任务数量
- **SSIM**: 平均 SSIM 得分
- **LPIPS**: 平均 LPIPS 距离
- **DINO Similarity**: 平均 DINO 相似度
- **S_str**: 结构保真度得分
- **CLIP**: 平均 CLIP 相似度
- **Text Similarity**: 平均文本相似度
- **S_sty**: 风格一致性得分
- **CVQI**: 综合视觉质量指数
- **Success Ratio**: 成功执行步骤的比例
- **Visual Score**: 步骤间加权平均 CVQI

---

## ⚙️ 配置说明

### 模型配置

编辑 `scripts/benchmark.py` 来配置您的模型：

```python
MODEL_LIST = [
    {
        "name": "gpt-4o",
        "model_id": "gpt-4o",
        "api_key": "your-api-key",
        "base_url": "https://api.openai.com/v1",
        "description": "OpenAI GPT-4o"
    },
    {
        "name": "claude-3-opus",
        "model_id": "claude-3-opus-20240229",
        "api_key": "your-anthropic-api-key",
        "base_url": "https://api.anthropic.com/v1",
        "description": "Anthropic Claude 3 Opus"
    }
]
```

### 路径配置

默认路径（可通过命令行参数修改）：

- 数据集：`./dataset_multi_turn`
- Vue 输出：`./frontend/src/components/dataset`
- 配置输出：`./outputs/reproduced_configs`
- 截图输出：`./outputs/build_dataset_multi_turn`
- 前端 URL：`http://localhost:1234`

---

## ⚡ 性能优化

### GPU 加速

当 CUDA 可用时，基准测试自动使用 GPU 加速：
- CLIP 模型
- LPIPS 模型
- DINOv2 模型
- VGG19 模型

EasyOCR 在检测到 CUDA 时也支持 GPU 加速。

### 批量处理

使用 `--limit` 标志测试任务子集：

```bash
python scripts/benchmark.py build --limit 5
```

### 无头浏览器

在无头模式下运行 Playwright 以加快截图捕获：

```bash
python scripts/benchmark.py record --headless
```

---

## 🔧 常见问题

### CUDA 显存不足

减少批次大小或使用 CPU 模式：

```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/benchmark.py compare
```

### Playwright 超时

在 `scripts/benchmark.py` 中增加超时时间：

```python
page.wait_for_selector(selector, timeout=10000)
```

### 缺少依赖

安装所有必需的包：

```bash
pip install -r requirements.txt
npx playwright install chromium
```

---

## 📧 联系方式

如有问题、建议或反馈，请联系：

**邮箱**: yangtiankun25@mails.ucas.cn

---

## 📄 许可证

本项目作为研究基准测试提供。

---

## 📚 引用

如果您在研究中使用 MultiInteract-Bench，请引用：

```
MultiInteract-Bench: 用于评估从图像序列还原网页交互能力的基准测试