# MultiInteract-Bench 项目结构说明

## 整理后的目录结构

```
MultiInteract-Bench/
├── frontend/                    # 前端 Vue.js 应用
│   ├── src/                    # 源代码
│   │   ├── components/         # Vue 组件
│   │   │   └── dataset/       # 生成组件存放目录
│   │   ├── main.ts            # 入口文件
│   │   ├── App.vue            # 主应用组件
│   │   └── style.css          # 全局样式
│   ├── public/                # 静态资源
│   ├── index.html            # HTML 模板
│   ├── package.json          # npm 依赖配置
│   ├── vite.config.ts        # Vite 构建配置
│   ├── tailwind.config.js    # Tailwind CSS 配置
│   └── tsconfig.json         # TypeScript 配置
│
├── backend/                   # 后端评测脚本（原始版本）
│   ├── build.py              # 模型复现脚本
│   ├── recorder_build.py     # 截图捕获脚本
│   ├── compare_visual.py     # 视觉指标计算
│   └── calculate_model_metrics.py  # 统计分析
│
├── scripts/                   # 统一评测脚本（推荐使用）
│   └── benchmark.py         # 主评测脚本（整合4个阶段）
│
├── dataset_examples/          # 数据集示例
│   ├── Spotify_1766618072/  # 示例任务1：音乐播放器
│   │   ├── metadata.json    # 任务元数据
│   │   ├── step_00.png      # 初始状态
│   │   └── step_*.png       # 各步骤状态
│   └── Stripe_1766502847/    # 示例任务2：支付表单
│       ├── metadata.json
│       └── step_*.png
│
├── outputs/                   # 输出目录
│   ├── reproduced_configs/    # 生成的配置文件
│   ├── build_dataset_multi_turn/  # 捕获的截图
│   └── model_metrics_summary.csv  # 评测结果汇总
│
├── requirements.txt          # Python 依赖
├── README.md                # 英文文档
├── README_zh.md             # 中文文档
├── PROJECT_STRUCTURE.md     # 本文件
└── .gitignore               # Git 忽略配置
```

## 文件说明

### 核心评测脚本

#### `scripts/benchmark.py` ⭐（推荐使用）

统一的评测脚本，整合了 4 个评测阶段：

1. **build 阶段**：使用大模型根据图像序列复现网页
2. **record 阶段**：自动截图测试生成的网页
3. **compare 阶段**：计算视觉评测指标
4. **metrics 阶段**：统计和分析评测结果

**使用方法**：
```bash
# 完整流程
python scripts/benchmark.py build [选项]
python scripts/benchmark.py record [选项]
python scripts/benchmark.py compare [选项]
python scripts/benchmark.py metrics [选项]
```

### 原始脚本（已移至 backend/）

以下文件为原始的独立脚本，已保留在 `backend/` 目录中供参考：

- `backend/build.py`：模型复现脚本
- `backend/recorder_build.py`：截图捕获脚本
- `backend/compare_visual.py`：视觉指标计算
- `backend/calculate_model_metrics.py`：统计分析

### 前端文件

前端是一个基于 Vue 3 + TypeScript + Tailwind CSS 的应用：

- `frontend/src/App.vue`：主应用组件
- `frontend/src/main.ts`：应用入口
- `frontend/vite.config.ts`：Vite 构建配置
- `frontend/package.json`：npm 依赖管理

**运行前端**：
```bash
cd frontend
npm install
npm run dev
```

### 数据集文件

数据集包含多个任务，每个任务包含：

- `metadata.json`：任务描述和步骤信息
- `step_00.png`：初始状态截图
- `step_*.png`：各交互步骤后的截图

**下载完整数据集**：
```
https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench
```

### 输出文件

评测过程中的输出文件：

- `outputs/reproduced_configs/`：生成的配置文件（JSON格式）
- `outputs/build_dataset_multi_turn/`：捕获的截图和元数据
- `outputs/model_metrics_summary.csv`：最终的评测结果汇总

## 评测流程

### 完整评测流程

```
1. Build 阶段
   └─> 使用 LLM 从数据集生成 Vue 组件
       ├─> 输出: frontend/src/components/dataset/*.vue
       └─> 输出: outputs/reproduced_configs/*.json

2. Record 阶段
   └─> 自动截图测试生成的网页
       └─> 输出: outputs/build_dataset_multi_turn/{task_id}/step_*.png

3. Compare 阶段
   └─> 计算视觉评测指标
       └─> 更新: outputs/build_dataset_multi_turn/{task_id}/metadata.json

4. Metrics 阶段
   └─> 统计分析评测结果
       └─> 输出: outputs/model_metrics_summary.csv
```

### 快速开始

```bash
# 1. 配置模型 API（编辑 scripts/benchmark.py 中的 MODEL_LIST）

# 2. 生成组件（测试前10个任务）
python scripts/benchmark.py build --limit 10

# 3. 启动前端服务（新终端）
cd frontend && npm run dev

# 4. 截图测试
python scripts/benchmark.py record --headless

# 5. 计算指标
python scripts/benchmark.py compare

# 6. 生成结果
python scripts/benchmark.py metrics
```

## 配置说明

### 模型配置

在 `scripts/benchmark.py` 中配置要测试的模型：

```python
MODEL_LIST = [
    {
        "name": "gpt-4o",
        "model_id": "gpt-4o",
        "api_key": "your-api-key",
        "base_url": "https://api.openai.com/v1",
        "description": "OpenAI GPT-4o"
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

## 评测指标

MultiInteract-Bench 使用 8 个原子指标和 1 个综合指标：

### 原子指标

1. **CLIP Similarity**：语义对齐度（0-1，越高越好）
2. **LPIPS Distance**：感知距离（0-∞，越低越好）
3. **Style Loss**：风格损失（0-∞，越低越好）
4. **Text Similarity**：文本相似度（0-1，越高越好）
5. **Color Histogram Similarity**：颜色直方图相似度（0-1，越高越好）
6. **Dominant Color Similarity**：主色调相似度（0-1，越高越好）
7. **DINO Similarity**：结构相似度（0-1，越高越好）
8. **SSIM**：结构相似度指数（0-1，越高越好）

### 综合指标

**CVQI (Comprehensive Visual Quality Index)**：
```
CVQI = 0.50 × S_str + 0.20 × S_sem + 0.20 × S_txt + 0.10 × S_sty
```

其中：
- `S_str`：结构保真度（SSIM + LPIPS + DINO）
- `S_sem`：语义对齐（CLIP）
- `S_txt`：文本完整性（Text Similarity）
- `S_sty`：风格一致性（Color + Dominant Color + Style）

## 常见命令

### 测试特定模型

```bash
python scripts/benchmark.py build --models gpt-4o claude-3-opus
```

### 测试特定任务

```bash
python scripts/benchmark.py record --task-id gpt-4o_TaskName_1234567890
```

### 测试特定模型前缀的所有任务

```bash
python scripts/benchmark.py record --model-prefix gpt-4o
```

### 跳过已处理的任务

```bash
python scripts/benchmark.py build --skip-processed
```

### 重新运行所有任务

```bash
python scripts/benchmark.py build --no-skip-processed
```

### 限制处理数量（测试用）

```bash
python scripts/benchmark.py build --limit 5
```

## 注意事项

1. **首次使用前**：请先配置模型 API（编辑 `scripts/benchmark.py`）
2. **GPU 加速**：推荐使用 CUDA GPU 以加快处理速度
3. **前端端口**：默认使用 1234 端口，确保端口未被占用
4. **Playwright 浏览器**：运行前需要安装 `npx playwright install chromium`
5. **数据集下载**：完整数据集需从 HuggingFace 下载

## 联系方式

如有问题或建议，请联系：yangtiankun25@mails.ucas.cn