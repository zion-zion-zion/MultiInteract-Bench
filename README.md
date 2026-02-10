<div align="center">

# MultiInteract-Bench

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

**A Benchmark for Evaluating Web Interaction Reconstruction from Image Sequences**

---

[📚 Documentation](#installation) • [🚀 Quick Start](#quick-start) • [📊 Evaluation Metrics](#evaluation-metrics) • [📖 Examples](#usage-examples)

</div>

## 📋 Abstract

MultiInteract-Bench is a comprehensive evaluation framework designed to assess the capabilities of multimodal large language models in reproducing web-based interactions. It provides a standardized pipeline for:

- **Model Reproduction**: Using LLMs to regenerate web components from image sequences
- **Automated Testing**: Automated screenshot capture of generated web pages
- **Visual Metrics**: Computing multiple evaluation metrics to assess visual fidelity
- **Result Analysis**: Statistical analysis and comparison across different models

## Project Structure

```
MultiInteract-Bench/
├── frontend/                    # Frontend Vue.js application
│   ├── src/                    # Source code
│   │   ├── components/         # Vue components
│   │   │   └── dataset/       # Generated components for testing
│   │   └── ...
│   ├── public/                 # Static assets
│   └── ...
├── backend/                    # Backend evaluation scripts (original)
│   ├── build.py               # Model reproduction script
│   ├── recorder_build.py      # Screenshot capture script
│   ├── compare_visual.py      # Visual metrics calculation
│   └── calculate_model_metrics.py  # Statistical analysis
├── scripts/                    # Unified evaluation scripts
│   └── benchmark.py          # Main benchmark script (4 phases)
├── dataset_examples/           # Dataset example folders
│   ├── Spotify_1766618072/   # Example task 1
│   └── Stripe_1766502847/     # Example task 2
├── outputs/                    # Output directory
│   ├── reproduced_configs/     # Generated config files
│   ├── build_dataset_multi_turn/  # Captured screenshots
│   └── model_metrics_summary.csv  # Evaluation results
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── README_zh.md             # Chinese documentation
```

## Evaluation Metrics

MultiInteract-Bench employs 8 atomic metrics to evaluate the quality of generated web pages:

### 1. CLIP Similarity
Measures high-level semantic alignment between generated and reference images.
- **Range**: 0-1 (higher is better)
- **Model**: OpenAI CLIP (ViT-B/32)

### 2. LPIPS Distance
Perceptual distance metric that captures human visual differences.
- **Range**: 0-∞ (lower is better)
- **Model**: LPIPS with VGG backbone

### 3. Style Loss
Evaluates artistic style consistency using Gram matrices.
- **Range**: 0-∞ (lower is better)
- **Model**: VGG19 neural style transfer features

### 4. Text Similarity
Measures text content preservation using OCR and LCS.
- **Range**: 0-1 (higher is better)
- **Method**: Longest Common Subsequence (LCS)

### 5. Color Histogram Similarity
Compares overall color distribution across RGB channels.
- **Range**: 0-1 (higher is better)
- **Method**: Histogram correlation coefficient

### 6. Dominant Color Similarity
Evaluates primary color consistency using K-means clustering.
- **Range**: 0-1 (higher is better)
- **Method**: HSV hue distance with greedy matching

### 7. DINO Similarity
Assesses structural layout and high-level visual features.
- **Range**: 0-1 (higher is better)
- **Model**: Meta DINOv2 (ViT-Base)

### 8. SSIM
Structural Similarity Index for pixel-level structural fidelity.
- **Range**: 0-1 (higher is better)
- **Method**: Sliding window SSIM

### Comprehensive Visual Quality Index (CVQI)

The 8 atomic metrics are aggregated into a comprehensive score (CVQI) that captures overall visual quality across multiple dimensions including structure, style, semantics, and text. The CVQI score ranges from 0-1 (higher is better).

**Visual Score**: The final evaluation metric for each task, calculated as a weighted average of CVQI scores across all steps, where later steps receive higher weights to emphasize the final state of the interaction.

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8+
- **Node.js**: 16+ (for frontend)
- **CUDA-capable GPU**: Recommended for faster processing
- **Conda**: Recommended for environment management

### Option 1: Using Conda (Recommended)

#### Step 1: Create and Activate Conda Environment

```bash
# Create a new conda environment
conda create -n multinteract python=3.10 -y

# Activate the environment
conda activate multinteract
```

#### Step 2: Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU-only
conda install pytorch torchvision cpuonly -c pytorch
```

#### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` & `torchvision`: Deep learning framework
- `openai-clip`: Vision-language model for semantic similarity
- `lpips`: Perceptual image similarity metric
- `opencv-python`: Image processing
- `scikit-learn`: Machine learning utilities
- `easyocr`: OCR for text extraction
- `playwright`: Browser automation for screenshot capture

#### Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

#### Step 5: Install Playwright Browsers

```bash
npx playwright install chromium
```

### Option 2: Using pip Only

If you prefer not to use conda, you can install dependencies directly with pip:

```bash
# Create virtual environment (optional but recommended)
python -m venv multinteract_env
source multinteract_env/bin/activate  # On Windows: multinteract_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..

# Install Playwright browsers
npx playwright install chromium
```

### Verify Installation

To verify your installation:

```bash
# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check other dependencies
python -c "import clip, lpips, cv2; print('All dependencies installed successfully!')"
```

## 📊 Dataset

<details>
<summary>📥 <strong>Download Full Dataset</strong></summary>

The complete MultiInteract-Bench dataset is available on HuggingFace:

[![Download Dataset](https://img.shields.io/badge/Download-HuggingFace-yellow.svg)](https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench/resolve/main/dataset_multi_turn.zip)

```bash
# Download the dataset zip file
wget https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench/resolve/main/dataset_multi_turn.zip

# Or using curl
curl -L -o dataset_multi_turn.zip https://huggingface.co/datasets/zionzionzion/MultiInteract-Bench/resolve/main/dataset_multi_turn.zip

# Unzip the dataset
unzip dataset_multi_turn.zip -d ./

# Verify the download
ls dataset_multi_turn/ | head -5
```

**Note**: After unzipping, the dataset will be in the `./dataset_multi_turn/` directory.

</details>

### Dataset Structure

Each dataset task follows this structure:

```
task_name/
├── metadata.json           # Task metadata and step descriptions
├── step_00.png           # Initial state (before any interaction)
├── step_01.png           # State after step 1 interaction
├── step_02.png           # State after step 2 interaction
└── ...                   # Additional interaction steps
```

### Sample Tasks

Two example tasks are included in `dataset_examples/`:

| Task Name | Description | Steps |
|-----------|-------------|-------|
| `Spotify_1766618072/` | Music player interface | 6 steps |
| `Stripe_1766502847/` | Payment form interface | 6 steps |

---

## 🚀 Quick Start

This section provides a quick walkthrough to get started with MultiInteract-Bench. For detailed configuration options, see [Usage Examples](#usage-examples).

### Phase 1: Model Reproduction (Build Phase)

Use LLMs to generate Vue components from the dataset:

```bash
python scripts/benchmark.py build \
    --dataset-dir ./dataset_multi_turn \
    --output-vue-dir ./frontend/src/components/dataset \
    --output-config-dir ./outputs/reproduced_configs \
    --limit 10  # Process first 10 tasks for testing
```

**Note**: Before running, edit `scripts/benchmark.py` and configure your model APIs in the `MODEL_LIST` variable.

### Step 2: Start Frontend Server

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:1234`

### Step 3: Automated Screenshot Capture (Record Phase)

Capture screenshots of generated web pages:

```bash
python scripts/benchmark.py record \
    --base-url http://localhost:1234 \
    --config-dir ./outputs/reproduced_configs \
    --output-dir ./outputs/build_dataset_multi_turn \
    --headless
```

### Step 4: Compute Visual Metrics (Compare Phase)

Calculate evaluation metrics:

```bash
python scripts/benchmark.py compare \
    --build-dir ./outputs/build_dataset_multi_turn \
    --dataset-dir ./dataset_multi_turn
```

This will compute all 8 metrics for each step and update metadata files.

### Step 5: Statistical Analysis (Metrics Phase)

Generate summary statistics:

```bash
python scripts/benchmark.py metrics \
    --base-dir ./outputs/build_dataset_multi_turn \
    --output-csv ./outputs/model_metrics_summary.csv
```

Results will be saved to `model_metrics_summary.csv`.

## Usage Examples

### Test with Specific Models

```bash
python scripts/benchmark.py build \
    --models model-name-1 model-name-2
```

### Test a Specific Task

```bash
python scripts/benchmark.py record \
    --task-id model-name_TaskName_1234567890
```

### Test All Tasks for a Model Prefix

```bash
python scripts/benchmark.py record \
    --model-prefix gpt-4o
```

### Skip Already Processed Tasks

```bash
python scripts/benchmark.py build --skip-processed
```

### Re-run All Tasks

```bash
python scripts/benchmark.py build --no-skip-processed
```

## Output Format

### Metadata.json Structure

After the compare phase, each task's metadata.json contains:

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

### CSV Output Format

The metrics summary CSV contains:

- **Model**: Model name
- **Count**: Number of tasks evaluated
- **SSIM**: Average SSIM score
- **LPIPS**: Average LPIPS distance
- **DINO Similarity**: Average DINO similarity
- **S_str**: Structural Fidelity score
- **CLIP**: Average CLIP similarity
- **Text Similarity**: Average text similarity
- **S_sty**: Stylistic Consistency score
- **CVQI**: Comprehensive Visual Quality Index
- **Success Ratio**: Ratio of successfully executed steps
- **Visual Score**: Weighted average CVQI across steps

## Configuration

### Model Configuration

Edit `scripts/benchmark.py` to configure your models:

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

## Performance Optimization

### GPU Acceleration

The benchmark automatically uses CUDA when available for:
- CLIP model
- LPIPS model
- DINOv2 model
- VGG19 model

EasyOCR also supports GPU acceleration when CUDA is detected.

### Batch Processing

Use the `--limit` flag to test with a subset of tasks:

```bash
python scripts/benchmark.py build --limit 5
```

### Headless Browser

Run Playwright in headless mode for faster screenshot capture:

```bash
python scripts/benchmark.py record --headless
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/benchmark.py compare
```

### Playwright Timeout

Increase timeout in `scripts/benchmark.py`:
```python
page.wait_for_selector(selector, timeout=10000)
```

### Missing Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
npx playwright install chromium
```

## Contact

For questions, issues, or suggestions, please contact:

**Email**: yangtiankun25@mails.ucas.cn

## License

This project is provided as a benchmark for research purposes.

## Citation

If you use MultiInteract-Bench in your research, please cite:

```
MultiInteract-Bench: A Benchmark for Evaluating Web Interaction Reconstruction from Image Sequences