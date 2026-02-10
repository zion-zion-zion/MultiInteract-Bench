#!/usr/bin/env python3
"""
MultiInteract-Bench - 统一评测脚本

整合了以下功能：
1. build: 使用模型复现数据集中的示例
2. record: 根据配置文件自动截图测试生成的网页
3. compare: 计算各种视觉评测指标
4. metrics: 统计和分析评测结果

使用方法:
    python scripts/benchmark.py build [选项]
    python scripts/benchmark.py record [选项]
    python scripts/benchmark.py compare [选项]
    python scripts/benchmark.py metrics [选项]
"""

import os
import sys
import json
import argparse
import base64
import time
import glob
import re
from pathlib import Path
from collections import defaultdict

# ============================================================================
# Phase 1: Build - 使用模型复现数据集
# ============================================================================

def build_phase(args):
    """Phase 1: 使用大模型复现数据集"""
    from openai import OpenAI
    
    # 配置
    MODEL_LIST = [
        {
            "name": "model-name",
            "model_id": "model-id",
            "api_key": "your-api-key",
            "base_url": "your-api-base-url",
            "description": "Brief description of the model and its capabilities"
        }
    ]
    
    DATASET_DIR = args.dataset_dir
    OUTPUT_VUE_DIR = args.output_vue_dir
    OUTPUT_CONFIG_DIR = args.output_config_dir
    
    # 创建输出目录
    os.makedirs(OUTPUT_VUE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CONFIG_DIR, exist_ok=True)
    
    def encode_image_to_base64(image_path):
        """将图片编码为 base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_prompt_from_metadata(metadata):
        """根据 metadata 创建提示词"""
        sequence = metadata.get("sequence", [])
        
        prompt = """You are an expert Frontend Engineer specializing in Vue 3. Your task is to recreate the Vue component EXACTLY as shown in the provided images.

## CRITICAL INSTRUCTIONS
1. **YOU MUST RECREATE THE COMPONENT EXACTLY AS IT APPEARS IN THE IMAGES**
2. Analyze each image carefully - the layout, colors, fonts, spacing, and all visual details must match
3. The component must be pixel-perfect to what you see in the images
4. Use the same visual style, proportions, and positioning as shown

## Step Sequence
You will see multiple images representing each step of the interaction. Below are the step descriptions:

"""
        
        for i, step in enumerate(sequence, 1):
            action = step.get("action", {})
            desc = action.get("desc", "")
            prompt += f"\n**Step {i}**: {desc}\n"
        
        prompt += """

## Requirements
- Vue 3 Composition API (`<script setup>`)
- Tailwind CSS for styling
- Add `data-testid` attributes to all interactive elements
- Single file Vue component (template, script, style all in one file)
- The component must look EXACTLY like the images - same layout, colors, spacing, everything
Requirements for Playwright Steps:
- Define 3-5 distinct steps to demonstrate the full flow.
- Step 1 is always the initial state.
- **Strict Selector Matching**: 
    - Every `selector` you define MUST strictly correspond to an element existing in your generated `vue_code`.
    - **Valid Selectors**: Use specific CSS selectors matching the attributes you added (e.g., `[data-testid="open-modal"]`) or robust Playwright text locators (e.g., `button:has-text("Save")`).
    - **No Hallucinations**: Do not use selectors for IDs or classes that are not explicitly written in the Vue template.
    - Ensure the selector targets an element that is currently visible in the DOM (consider `v-if` logic).


Output Format (JSON Only):
{{
  "vue_code": "<template>... <button data-testid='create-btn'>Create</button> ...</template>...",
  "playwright_steps": [
    {{ "step": 1, "action": "wait", "desc": "Initial State: Component loaded" }},
    {{ "step": 2, "action": "click", "selector": "[data-testid='create-btn']", "desc": "Open the modal using the test id" }},
    {{ "step": 3, "action": "type", "selector": "input[type='text']", "value": "Test Value", "desc": "Fill form input" }},
    {{ "step": 4, "action": "click", "selector": "button:has-text('Save')", "desc": "Submit and change view" }}
  ]
}}
"""
        return prompt
    
    def process_dataset_folder(folder_path):
        """处理单个数据集文件夹"""
        folder_name = os.path.basename(folder_path)
        metadata_path = os.path.join(folder_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            print(f"⚠️  Skipping {folder_name}: No metadata.json found")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"❌ Error reading metadata for {folder_name}: {e}")
            return None
        
        sequence = metadata.get("sequence", [])
        images_content = []
        
        for step_info in sequence:
            image_filename = step_info.get("image")
            if image_filename:
                image_path = os.path.join(folder_path, image_filename)
                if os.path.exists(image_path):
                    base64_image = encode_image_to_base64(image_path)
                    images_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
                else:
                    print(f"⚠️  Image not found: {image_path}")
        
        if not images_content:
            print(f"⚠️  No images found for {folder_name}")
            return None
        
        text_prompt = create_prompt_from_metadata(metadata)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ]
        
        messages[0]["content"].extend(images_content)
        
        return {
            "folder_name": folder_name,
            "metadata": metadata,
            "messages": messages
        }
    
    def is_processed(folder_name, model_name):
        """检查某个文件夹和模型是否已处理"""
        config_filename = f"{model_name}_{folder_name}.json"
        config_path = os.path.join(OUTPUT_CONFIG_DIR, config_filename)
        return os.path.exists(config_path)
    
    def call_llm_to_reproduce(processed_data, model_config):
        """调用大模型生成代码"""
        folder_name = processed_data["folder_name"]
        messages = processed_data["messages"]
        metadata = processed_data["metadata"]
        model_name = model_config["name"]
        model_id = model_config["model_id"]
        
        base_id = f"{model_name}_{folder_name}"
        
        print(f"🔄 Processing: {folder_name} with {model_name}")
        print(f"   - Steps: {len(metadata.get('sequence', []))}")
        print(f"   - Model: {model_id}")
        
        try:
            client = OpenAI(
                api_key=model_config["api_key"],
                base_url=model_config["base_url"]
            )
            
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=8000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            vue_filename = f"{base_id}.vue"
            vue_path = os.path.join(OUTPUT_VUE_DIR, vue_filename)
            
            with open(vue_path, 'w', encoding='utf-8') as f:
                f.write(result.get("vue_code", ""))
            
            config_data = {
                "id": base_id,
                "url_param": base_id,
                "original_folder": folder_name,
                "model_name": model_name,
                "model_id": model_id,
                "model_description": model_config["description"],
                "total_steps": metadata.get("total_steps", len(result.get("playwright_steps", []))),
                "steps": result.get("playwright_steps", []),
                "source_file": f"src/components/dataset/{vue_filename}",
                "original_dataset": f"{DATASET_DIR}/{folder_name}"
            }
            
            config_filename = f"{base_id}.json"
            config_path = os.path.join(OUTPUT_CONFIG_DIR, config_filename)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Completed: {folder_name} with {model_name}")
            print(f"   - Vue file: {vue_path}")
            print(f"   - Config file: {config_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {folder_name} with {model_name}: {e}")
            return False
    
    # 确定要使用的模型列表
    if args.models:
        models_to_use = [m for m in MODEL_LIST if m["name"] in args.models]
        print(f"🎯 Using specific models: {args.models}")
    else:
        models_to_use = MODEL_LIST
        print(f"📋 Total models to use: {len(models_to_use)}")
    
    print("\n🤖 Models:")
    for i, model in enumerate(models_to_use, 1):
        print(f"   {i}. {model['name']} - {model['description']}")
    print()
    
    # 遍历数据集文件夹
    folders = []
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path):
            if args.folder and item != args.folder:
                continue
            folders.append(item_path)
    
    if args.folder:
        print(f"🎯 Processing specific folder: {args.folder}")
    else:
        print(f"📁 Total folders to process: {len(folders)}")
    
    if args.limit:
        folders = folders[:args.limit]
        print(f"⚠️  Limited to {args.limit} folders for testing")
    
    print(f"💾 Output directories:")
    print(f"   - Vue files: {OUTPUT_VUE_DIR}")
    print(f"   - Config files: {OUTPUT_CONFIG_DIR}")
    
    if args.skip_processed:
        print(f"✅ Skip processed tasks enabled")
    else:
        print(f"⚠️  Skip processed tasks disabled")
    
    print("-" * 60)
    
    total_tasks = len(folders) * len(models_to_use)
    processed_count = 0
    skipped_count = 0
    success_count = 0
    fail_count = 0
    
    task_number = 0
    for model in models_to_use:
        model_name = model["name"]
        
        for i, folder_path in enumerate(folders, 1):
            folder_name = os.path.basename(folder_path)
            task_number += 1
            
            if args.skip_processed and is_processed(folder_name, model_name):
                print(f"\n[{task_number}/{total_tasks}] ⏭️  Skipping: {folder_name} with {model_name}")
                skipped_count += 1
                continue
            
            print(f"\n[{task_number}/{total_tasks}] 📦 Processing: {folder_name} with {model_name}")
            
            try:
                processed_data = process_dataset_folder(folder_path)
                if processed_data:
                    success = call_llm_to_reproduce(processed_data, model)
                    processed_count += 1
                    
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    fail_count += 1
                
                if task_number < total_tasks:
                    print("   ⏸️  Waiting 2 seconds...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                fail_count += 1
                continue
    
    print("\n" + "=" * 60)
    print("📈 Build Summary")
    print("=" * 60)
    print(f"Total tasks: {total_tasks}")
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print("=" * 60)


# ============================================================================
# Phase 2: Record - 自动截图测试
# ============================================================================

def record_phase(args):
    """Phase 2: 自动截图测试生成的网页"""
    try:
        from PIL import Image, ImageStat
        from playwright.sync_api import sync_playwright
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    except ImportError:
        print("❌ Error: Missing required dependencies")
        print("Please install: playwright, pillow")
        return
    
    BASE_URL = args.base_url
    CONFIG_DIR = args.config_dir
    OUTPUT_DIR = args.output_dir
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CONTINUE_ON_ERROR = False
    HEADLESS_MODE = args.headless
    
    def load_configs(specific_task_id=None, model_prefix=None):
        """加载配置文件"""
        tasks = []
        files = glob.glob(os.path.join(CONFIG_DIR, "*.json"))
        
        for f in files:
            with open(f, 'r') as file:
                task = json.load(file)
                task_id = task.get('id', '')
                
                should_include = False
                
                if specific_task_id is not None:
                    should_include = (task_id == specific_task_id)
                elif model_prefix is not None:
                    should_include = task_id.startswith(model_prefix)
                else:
                    should_include = True
                
                if should_include:
                    tasks.append(task)
        
        return tasks
    
    def is_blank_image(image_path):
        """检查图片是否为空白"""
        try:
            if not os.path.exists(image_path):
                return False
            
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            stat = ImageStat.Stat(img)
            std_dev = stat.stddev
            std_dev_sum = sum(std_dev) if isinstance(std_dev, (list, tuple)) else std_dev
            mean_color = stat.mean
            
            is_low_variance = std_dev_sum < 3.0
            is_mostly_white = all(c > 240 for c in mean_color)
            
            return is_low_variance and is_mostly_white
            
        except Exception as e:
            print(f"  [Warning] Error checking blank image: {e}")
            return False
    
    def has_navigation_error(task_dir):
        """检查任务目录中的 metadata.json 是否包含导航失败错误"""
        try:
            metadata_path = os.path.join(task_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                return False
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            error_field = metadata.get('error', '')
            if error_field and 'Navigation failed:' in str(error_field):
                return True
            
            return False
            
        except Exception as e:
            return False
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    tasks = load_configs(args.task_id, args.model_prefix)
    
    if len(tasks) == 0:
        if args.task_id:
            print(f"[Error] Task '{args.task_id}' not found in {CONFIG_DIR}")
        elif args.model_prefix:
            print(f"[Error] No tasks found with model prefix '{args.model_prefix}'")
        else:
            print(f"[Warning] No tasks found in {CONFIG_DIR}")
        return
    
    if args.task_id:
        print(f"Loaded 1 task: {args.task_id}")
    elif args.model_prefix:
        print(f"Loaded {len(tasks)} tasks with model prefix: {args.model_prefix}")
    else:
        print(f"Loaded {len(tasks)} multi-step tasks.")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS_MODE)
        context = browser.new_context(viewport={"width": 1000, "height": 800})
        page = context.new_page()
        
        for task in tasks:
            print(f"\n--- Processing: {task['id']} ---")
            
            task_dir = os.path.join(OUTPUT_DIR, task['id'])
            
            if os.path.exists(task_dir):
                if has_navigation_error(task_dir):
                    print(f"  [Info] Directory exists but contains navigation error, reprocessing...")
                    import shutil
                    try:
                        shutil.rmtree(task_dir)
                        os.makedirs(task_dir)
                    except Exception as e:
                        print(f"  [Error] Failed to clean directory: {e}")
                        continue
                else:
                    print(f"  Directory {task_dir} already exists, skipping...")
                    continue
            else:
                os.makedirs(task_dir)
            
            url = f"{BASE_URL}/?component={task['url_param']}"
            print(f"Navigating to {url}")
            
            try:
                page.goto(url)
            except Exception as goto_error:
                print(f"  [Error] Failed to navigate to {url}: {goto_error}")
                continue
            
            try:
                page.wait_for_selector("body", timeout=10000)
                print("  Page loaded successfully")
            except Exception as e:
                print(f"  [Warning] Page load timeout: {e}")
            
            time.sleep(2.0)
            
            print("Capturing Step 0 (Initial State)...")
            page.locator("body").screenshot(path=f"{task_dir}/step_00.png")
            time.sleep(0.5)
            
            if is_blank_image(f"{task_dir}/step_00.png"):
                print(f"  [Error] Initial screenshot is blank. Terminating task.")
                continue
            
            executed_steps = []
            
            steps = task['steps']
            if isinstance(steps, str):
                try:
                    if '"playwright_steps":' in steps:
                        start = steps.find('[')
                        end = steps.rfind(']') + 1
                        if start >= 0 and end > start:
                            steps = json.loads(steps[start:end])
                    else:
                        steps = json.loads(steps)
                except Exception as parse_error:
                    print(f"  [Error] Failed to parse steps: {parse_error}")
                    continue
            
            for i, step in enumerate(steps):
                step_num = i + 1
                
                if not isinstance(step, dict):
                    continue
                
                step_desc = step.get('desc', 'No description')
                print(f"  Step {step_num}: {step_desc}")
                
                try:
                    action_type = step.get('action')
                    if not action_type:
                        continue
                    
                    selector = step.get('selector')
                    
                    if action_type == 'click' and selector:
                        try:
                            page.wait_for_selector(selector, timeout=5000, state='visible')
                            page.click(selector)
                        except:
                            element = page.query_selector(selector)
                            if element:
                                element.click()
                            else:
                                raise
                    elif action_type == 'fill' and selector:
                        page.wait_for_selector(selector, timeout=5000, state='visible')
                        page.fill(selector, step['value'])
                    elif action_type == 'type' and selector:
                        page.wait_for_selector(selector, timeout=5000, state='visible')
                        page.type(selector, step['value'])
                    elif action_type == 'hover' and selector:
                        page.wait_for_selector(selector, timeout=5000, state='visible')
                        page.hover(selector)
                    elif 'wait' in action_type:
                        wait_time = step.get('value', 1)
                        time.sleep(1)
                    elif action_type == 'dragAndDrop':
                        source = step.get('source')
                        target = step.get('target')
                        if source and target:
                            page.wait_for_selector(source, timeout=5000, state='visible')
                            page.wait_for_selector(target, timeout=5000, state='visible')
                            page.drag_and_drop(source, target)
                    
                    time.sleep(1)
                    
                    screenshot_name = f"step_{step_num:02d}.png"
                    page.locator("body").screenshot(path=f"{task_dir}/{screenshot_name}")
                    
                    executed_steps.append({
                        "step_index": step_num,
                        "action": step,
                        "image": screenshot_name
                    })
                    
                except Exception as e:
                    print(f"  [Error] Step {step_num} failed: {e}")
                    break
            
            source_code = ""
            try:
                src_path = os.path.join(PROJECT_ROOT, task['source_file'])
                with open(src_path, "r") as f:
                    source_code = f.read()
            except:
                source_code = "// Source file not found"
            
            description = task.get('description', '')
            
            total_planned_steps = len(steps) if steps else 0
            successful_steps = len(executed_steps)
            success_ratio = successful_steps / total_planned_steps if total_planned_steps > 0 else 0
            
            final_metadata = {
                "id": task['id'],
                "description": description,
                "ground_truth_code": source_code,
                "total_planned_steps": total_planned_steps,
                "successful_steps": successful_steps,
                "success_ratio": round(success_ratio, 4),
                "sequence": executed_steps
            }
            
            if 'meta' in task:
                final_metadata['meta'] = task['meta']
            
            with open(f"{task_dir}/metadata.json", "w") as f:
                json.dump(final_metadata, f, indent=2)
            
            print(f"Saved {task['id']} with {len(executed_steps)} steps.")
        
        browser.close()


# ============================================================================
# Phase 3: Compare - 计算视觉评测指标
# ============================================================================

def compare_phase(args):
    """Phase 3: 计算视觉评测指标"""
    try:
        import torch
        import clip
        import lpips
        import torchvision.models as models
        import torchvision.transforms as transforms
        import cv2
        from sklearn.cluster import KMeans
        from scipy.ndimage import uniform_filter
        from PIL import Image
    except ImportError as e:
        print(f"❌ Error: Missing required dependencies - {e}")
        print("Please install: torch, torchvision, clip, lpips, opencv-python, scikit-learn, scipy, pillow, easyocr")
        return
    
    # 全局变量，缓存 EasyOCR 实例
    _ocr_reader = None
    
    def get_ocr_reader():
        """获取缓存的 EasyOCR Reader 实例"""
        global _ocr_reader
        
        if _ocr_reader is None:
            try:
                import easyocr
                use_gpu = torch.cuda.is_available()
                _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                print(f"Using EasyOCR backend (GPU: {use_gpu})")
            except ImportError:
                print("Warning: easyocr not installed")
                return None
            except Exception as e:
                print(f"Warning: Failed to initialize EasyOCR: {e}")
                return None
        
        return _ocr_reader
    
    def load_clip_model():
        """加载CLIP模型"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
    
    def load_lpips_model():
        """加载LPIPS模型"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = lpips.LPIPS(net='vgg').to(device)
        return model, device
    
    def load_vgg_model():
        """加载VGG19模型用于计算Style Loss"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vgg = models.vgg19(pretrained=True).features.to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg, device
    
    def load_dino_model():
        """加载DINOv2模型"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return model, preprocess, device
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            return None, None, device
    
    def gram_matrix(tensor):
        """计算Gram矩阵"""
        batch_size, channels, height, width = tensor.size()
        features = tensor.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (channels * height * width)
        return gram
    
    def calculate_clip_similarity(model, preprocess, device, image_path1, image_path2):
        """计算CLIP相似度"""
        try:
            image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
            image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features1 = model.encode_image(image1)
                image_features2 = model.encode_image(image2)
                similarity = torch.cosine_similarity(image_features1, image_features2)
                return float(similarity.cpu().numpy()[0])
        except Exception as e:
            print(f"Error calculating CLIP similarity: {e}")
            return None
    
    def calculate_lpips_distance(model, device, image_path1, image_path2):
        """计算LPIPS距离"""
        try:
            img1 = Image.open(image_path1).convert('RGB')
            img2 = Image.open(image_path2).convert('RGB')
            
            width = min(img1.width, img2.width)
            height = min(img1.height, img2.height)
            img1 = img1.resize((width, height), Image.Resampling.LANCZOS)
            img2 = img2.resize((width, height), Image.Resampling.LANCZOS)
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            img1_tensor = transform(img1).unsqueeze(0).to(device)
            img2_tensor = transform(img2).unsqueeze(0).to(device)
            
            with torch.no_grad():
                distance = model(img1_tensor, img2_tensor)
                return float(distance.cpu().numpy()[0])
        except Exception as e:
            print(f"Error calculating LPIPS distance: {e}")
            return None
    
    def calculate_style_loss(vgg, device, image_path1, image_path2):
        """计算Style Loss"""
        try:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img1 = Image.open(image_path1).convert('RGB')
            img2 = Image.open(image_path2).convert('RGB')
            
            img1_tensor = preprocess(img1).unsqueeze(0).to(device)
            img2_tensor = preprocess(img2).unsqueeze(0).to(device)
            
            style_layers = ['0', '5', '10', '19', '28']
            
            x1 = img1_tensor
            x2 = img2_tensor
            
            total_style_loss = 0.0
            layer_weights = {'0': 1.0/5, '5': 1.0/5, '10': 1.0/5, '19': 1.0/5, '28': 1.0/5}
            
            with torch.no_grad():
                for name, layer in vgg._modules.items():
                    x1 = layer(x1)
                    x2 = layer(x2)
                    
                    if name in style_layers:
                        gram1 = gram_matrix(x1)
                        gram2 = gram_matrix(x2)
                        layer_loss = torch.mean((gram1 - gram2) ** 2)
                        total_style_loss += layer_weights[name] * layer_loss
            
            return float(total_style_loss.cpu().numpy())
        except Exception as e:
            print(f"Error calculating Style Loss: {e}")
            return None
    
    def extract_text_from_image(image_path):
        """从图片中提取文本"""
        try:
            reader = get_ocr_reader()
            if reader is None:
                return ""
            
            result = reader.readtext(image_path)
            text_parts = [item[1] for item in result if len(item) >= 2]
            text = ' '.join(text_parts)
            return text
        except Exception as e:
            return ""
    
    def longest_common_subsequence_length(text1, text2):
        """计算LCS长度"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def calculate_text_similarity(image_path1, image_path2):
        """计算文本相似度"""
        try:
            text1 = extract_text_from_image(image_path1)
            text2 = extract_text_from_image(image_path2)
            
            if not text1 or not text2:
                return 0.0
            
            lcs_length = longest_common_subsequence_length(text1, text2)
            min_length = min(len(text1), len(text2))
            lcs_ratio = lcs_length / min_length
            
            return float(lcs_ratio)
        except Exception as e:
            return None
    
    def calculate_color_histogram_similarity(image_path1, image_path2):
        """计算色彩直方图相似度"""
        try:
            img1 = cv2.imread(image_path1)
            img2 = cv2.imread(image_path2)
            
            if img1 is None or img2 is None:
                return None
            
            hist1 = []
            hist2 = []
            
            for i in range(3):
                h1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
                h2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
                cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
                hist1.append(h1)
                hist2.append(h2)
            
            correlations = []
            for i in range(3):
                corr = cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_CORREL)
                correlations.append(corr)
            
            avg_correlation = sum(correlations) / len(correlations)
            similarity = max(0, (1 + avg_correlation) / 2)
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating color histogram similarity: {e}")
            return None
    
    def extract_dominant_colors(image_path, n_colors=5):
        """提取主导颜色"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            height, width = img.shape[:2]
            scale = min(200 / height, 200 / width)
            img_small = cv2.resize(img, (int(width * scale), int(height * scale)))
            
            pixels = img_small.reshape(-1, 3)
            
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            total_pixels = len(labels)
            color_counts = np.bincount(labels, minlength=n_colors)
            color_percentages = color_counts / total_pixels
            
            return dominant_colors, color_percentages
        except Exception as e:
            return None, None
    
    def rgb_to_hsv(rgb):
        """RGB转HSV"""
        rgb_norm = np.array(rgb) / 255.0
        hsv = cv2.cvtColor(np.array([[rgb_norm]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        h = h * 2
        return h, s, v
    
    def calculate_hue_distance(hue1, hue2):
        """计算色相距离"""
        diff = abs(hue1 - hue2)
        return min(diff, 360 - diff)
    
    def calculate_dominant_color_similarity(image_path1, image_path2, n_colors=5):
        """计算主色调相似度"""
        try:
            colors1, percentages1 = extract_dominant_colors(image_path1, n_colors)
            colors2, percentages2 = extract_dominant_colors(image_path2, n_colors)
            
            if colors1 is None or colors2 is None:
                return None
            
            hues1 = np.array([rgb_to_hsv(color)[0] for color in colors1])
            hues2 = np.array([rgb_to_hsv(color)[0] for color in colors2])
            
            distances = np.zeros((n_colors, n_colors))
            for i in range(n_colors):
                for j in range(n_colors):
                    distances[i, j] = calculate_hue_distance(hues1[i], hues2[j])
            
            similarities = 1 - (distances / 180.0)
            
            matched1 = set()
            matched2 = set()
            
            matches = []
            for i in range(n_colors):
                for j in range(n_colors):
                    matches.append((similarities[i, j], i, j))
            
            matches.sort(key=lambda x: x[0], reverse=True)
            
            total_weighted_similarity = 0.0
            total_weight = 0.0
            
            for similarity, i, j in matches:
                if i not in matched1 and j not in matched2:
                    weight = np.sqrt(percentages1[i] * percentages2[j])
                    total_weighted_similarity += weight * similarity
                    total_weight += weight
                    matched1.add(i)
                    matched2.add(j)
            
            if total_weight > 0:
                final_similarity = total_weighted_similarity / total_weight
            else:
                final_similarity = 0.0
            
            final_similarity = max(0.0, min(1.0, final_similarity))
            return float(final_similarity)
        except Exception as e:
            print(f"Error calculating dominant color similarity: {e}")
            return None
    
    def calculate_dino_similarity(model, preprocess, device, image_path1, image_path2):
        """计算DINO相似度"""
        try:
            if model is None or preprocess is None:
                return None
            
            img1 = Image.open(image_path1).convert('RGB')
            img2 = Image.open(image_path2).convert('RGB')
            
            img1_tensor = preprocess(img1).unsqueeze(0).to(device)
            img2_tensor = preprocess(img2).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features1 = model(img1_tensor)
                features2 = model(img2_tensor)
                
                if features1.dim() == 3:
                    cls_features1 = features1[:, 0, :]
                    cls_features2 = features2[:, 0, :]
                elif features1.dim() == 2:
                    cls_features1 = features1
                    cls_features2 = features2
                else:
                    return None
                
                similarity = torch.cosine_similarity(cls_features1, cls_features2)
                return float(similarity.cpu().numpy()[0])
        except Exception as e:
            print(f"Error calculating DINO similarity: {e}")
            return None
    
    def calculate_ssim(image_path1, image_path2):
        """计算SSIM"""
        try:
            img1 = cv2.imread(image_path1)
            img2 = cv2.imread(image_path2)
            
            if img1 is None or img2 is None:
                return None
            
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            width = min(img1_gray.shape[1], img2_gray.shape[1])
            height = min(img1_gray.shape[0], img2_gray.shape[0])
            img1_gray = cv2.resize(img1_gray, (width, height), interpolation=cv2.INTER_AREA)
            img2_gray = cv2.resize(img2_gray, (width, height), interpolation=cv2.INTER_AREA)
            
            img1_gray = img1_gray.astype(np.float64)
            img2_gray = img2_gray.astype(np.float64)
            
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            window_size = 11
            
            mu1 = uniform_filter(img1_gray, window_size)
            mu2 = uniform_filter(img2_gray, window_size)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = uniform_filter(img1_gray ** 2, window_size) - mu1_sq
            sigma2_sq = uniform_filter(img2_gray ** 2, window_size) - mu2_sq
            sigma12 = uniform_filter(img1_gray * img2_gray, window_size) - mu1_mu2
            
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            
            ssim_map = numerator / denominator
            avg_ssim = np.mean(ssim_map)
            
            return float(avg_ssim)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return None
    
    def calculate_comprehensive_visual_score(metrics):
        """计算综合视觉得分"""
        import numpy as np
        
        clip_sim = metrics['clip_similarity']
        ssim = metrics['ssim']
        text_sim = metrics['text_similarity']
        color_hist_sim = metrics['color_histogram_similarity']
        dom_color_sim = metrics['dominant_color_similarity']
        dino_sim = metrics['dino_similarity']
        
        lpips_sim = np.exp(-metrics['lpips_distance'])
        style_sim = np.exp(-100.0 * metrics['style_loss'])
        
        s_str = (1.0 / 10.0) * ssim + (2.0 / 10.0) * lpips_sim + (7.0 / 10.0) * dino_sim
        s_sem = clip_sim
        s_txt = text_sim
        s_sty = (1.0 / 3.0) * (color_hist_sim + dom_color_sim + style_sim)
        
        cvqi = (
            0.50 * s_str +
            0.20 * s_sem +
            0.20 * s_txt +
            0.10 * s_sty
        )
        
        cvqi = max(0.0, min(1.0, cvqi))
        
        return float(cvqi), float(s_str), float(s_sty)
    
    def parse_folder_name(folder_name):
        """解析文件夹名称"""
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
        
        return dict(sorted(step_images.items()))
    
    build_dir = args.build_dir
    dataset_dir = args.dataset_dir
    
    print("=" * 80)
    print("Visual Metrics Comparison")
    print("=" * 80)
    print(f"Build directory: {build_dir}")
    print(f"Dataset directory: {dataset_dir}")
    print()
    
    # 加载模型
    print("Loading models...")
    clip_model, preprocess, device = load_clip_model()
    print(f"CLIP model loaded on {device}")
    
    lpips_model, device = load_lpips_model()
    print(f"LPIPS model loaded on {device}")
    
    vgg_model, device = load_vgg_model()
    print(f"VGG19 model loaded on {device}")
    
    dino_model, dino_preprocess, device = load_dino_model()
    if dino_model is not None:
        print(f"DINOv2 model loaded on {device}")
    else:
        print("DINOv2 model not available")
    
    # 查找匹配的文件夹对
    print("\nFinding matching folder pairs...")
    matching_pairs = find_matching_folders(build_dir, dataset_dir)
    print(f"Found {len(matching_pairs)} matching pairs")
    
    # 处理每一对文件夹
    for i, (build_folder_path, dataset_folder_path, build_folder_name) in enumerate(matching_pairs, 1):
        print(f"\n{'='*80}")
        print(f"Processing pair {i}/{len(matching_pairs)}: {build_folder_name}")
        
        metadata_path = os.path.join(build_folder_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            print(f"  Warning: metadata.json not found")
            continue
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if 'sequence' not in metadata:
            print(f"  Warning: 'sequence' not found")
            continue
        
        dataset_metadata_path = os.path.join(dataset_folder_path, "metadata.json")
        dataset_total_steps = None
        if os.path.exists(dataset_metadata_path):
            try:
                with open(dataset_metadata_path, 'r', encoding='utf-8') as f:
                    dataset_metadata = json.load(f)
                    dataset_total_steps = dataset_metadata.get('total_steps')
                    print(f"  Dataset total_steps: {dataset_total_steps}")
            except Exception as e:
                pass
        
        build_step_images = get_step_images(build_folder_path, start_from=1)
        dataset_step_images = get_step_images(dataset_folder_path, start_from=1)
        
        updated_sequence = []
        
        for step_entry in metadata['sequence']:
            if 'image' not in step_entry:
                updated_sequence.append(step_entry)
                continue
            
            image_filename = step_entry['image']
            match = re.match(r'^step_(\d+)\.png$', image_filename)
            if not match:
                updated_sequence.append(step_entry)
                continue
            
            step_num = int(match.group(1))
            
            if step_num in build_step_images and step_num in dataset_step_images:
                build_image_path = os.path.join(build_folder_path, build_step_images[step_num])
                dataset_image_path = os.path.join(dataset_folder_path, dataset_step_images[step_num])
                
                step_entry_with_similarity = step_entry.copy()
                
                # CLIP相似度
                if 'clip_similarity' not in step_entry:
                    clip_similarity = calculate_clip_similarity(clip_model, preprocess, device, build_image_path, dataset_image_path)
                    step_entry_with_similarity['clip_similarity'] = clip_similarity
                    if clip_similarity is not None:
                        print(f"  Step {step_num}: CLIP = {clip_similarity:.4f}")
                else:
                    clip_similarity = step_entry['clip_similarity']
                
                # LPIPS距离
                if 'lpips_distance' not in step_entry:
                    lpips_distance = calculate_lpips_distance(lpips_model, device, build_image_path, dataset_image_path)
                    step_entry_with_similarity['lpips_distance'] = lpips_distance
                    if lpips_distance is not None:
                        print(f"  Step {step_num}: LPIPS = {lpips_distance:.4f}")
                else:
                    lpips_distance = step_entry['lpips_distance']
                
                # Style Loss
                if 'style_loss' not in step_entry:
                    style_loss = calculate_style_loss(vgg_model, device, build_image_path, dataset_image_path)
                    step_entry_with_similarity['style_loss'] = style_loss
                    if style_loss is not None:
                        print(f"  Step {step_num}: Style Loss = {style_loss:.4f}")
                else:
                    style_loss = step_entry['style_loss']
                
                # Text Similarity
                if 'text_similarity' not in step_entry:
                    text_similarity = calculate_text_similarity(build_image_path, dataset_image_path)
                    step_entry_with_similarity['text_similarity'] = text_similarity
                    if text_similarity is not None:
                        print(f"  Step {step_num}: Text Sim = {text_similarity:.4f}")
                else:
                    text_similarity = step_entry['text_similarity']
                
                # Color Histogram Similarity
                if 'color_histogram_similarity' not in step_entry:
                    color_histogram_similarity = calculate_color_histogram_similarity(build_image_path, dataset_image_path)
                    step_entry_with_similarity['color_histogram_similarity'] = color_histogram_similarity
                    if color_histogram_similarity is not None:
                        print(f"  Step {step_num}: Color Hist = {color_histogram_similarity:.4f}")
                else:
                    color_histogram_similarity = step_entry['color_histogram_similarity']
                
                # Dominant Color Similarity
                if 'dominant_color_similarity' not in step_entry:
                    dominant_color_similarity = calculate_dominant_color_similarity(build_image_path, dataset_image_path)
                    step_entry_with_similarity['dominant_color_similarity'] = dominant_color_similarity
                    if dominant_color_similarity is not None:
                        print(f"  Step {step_num}: Dom Color = {dominant_color_similarity:.4f}")
                else:
                    dominant_color_similarity = step_entry['dominant_color_similarity']
                
                # DINO Similarity
                if 'dino_similarity' not in step_entry:
                    dino_similarity = calculate_dino_similarity(dino_model, dino_preprocess, device, build_image_path, dataset_image_path)
                    step_entry_with_similarity['dino_similarity'] = dino_similarity
                    if dino_similarity is not None:
                        print(f"  Step {step_num}: DINO = {dino_similarity:.4f}")
                else:
                    dino_similarity = step_entry['dino_similarity']
                
                # SSIM
                if 'ssim' not in step_entry:
                    ssim = calculate_ssim(build_image_path, dataset_image_path)
                    step_entry_with_similarity['ssim'] = ssim
                    if ssim is not None:
                        print(f"  Step {step_num}: SSIM = {ssim:.4f}")
                else:
                    ssim = step_entry['ssim']
                
                # 综合视觉得分
                all_metrics_exist = all(key in step_entry_with_similarity for key in [
                    'clip_similarity', 'lpips_distance', 'style_loss', 'text_similarity',
                    'color_histogram_similarity', 'dominant_color_similarity', 'ssim', 'dino_similarity'
                ])
                
                if all_metrics_exist:
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
                    
                    if all(metrics[key] is not None for key in metrics):
                        cvqi, s_str, s_sty = calculate_comprehensive_visual_score(metrics)
                        step_entry_with_similarity['comprehensive_visual_score'] = cvqi
                        step_entry_with_similarity['s_str'] = s_str
                        step_entry_with_similarity['s_sty'] = s_sty
                        print(f"  Step {step_num}: CVQI = {cvqi:.4f}, S_str = {s_str:.4f}, S_sty = {s_sty:.4f}")
                
                updated_sequence.append(step_entry_with_similarity)
            else:
                updated_sequence.append(step_entry)
        
        metadata['sequence'] = updated_sequence
        
        # 计算 visual_score
        if dataset_total_steps and dataset_total_steps > 0:
            total_planned_steps = dataset_total_steps
            weighted_sum = 0.0
            total_weight = 0.0
            executed_steps = set()
            
            for step_entry in updated_sequence:
                if 'comprehensive_visual_score' in step_entry and step_entry['comprehensive_visual_score'] is not None:
                    image_filename = step_entry.get('image', '')
                    match = re.match(r'^step_(\d+)\.png$', image_filename)
                    if match:
                        step_num = int(match.group(1))
                        if step_num <= dataset_total_steps:
                            weight = step_num
                            cvqi_value = step_entry['comprehensive_visual_score']
                            weighted_sum += cvqi_value * weight
                            total_weight += weight
                            executed_steps.add(step_num)
            
            missing_steps = []
            for step_num in range(1, total_planned_steps + 1):
                if step_num not in executed_steps:
                    missing_steps.append(step_num)
                    total_weight += step_num
            
            if total_weight > 0:
                visual_score = weighted_sum / total_weight
                metadata['visual_score'] = float(visual_score)
                print(f"  Visual Score: {visual_score:.4f}")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  Updated metadata.json")
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


# ============================================================================
# Phase 4: Metrics - 统计评测结果
# ============================================================================

def metrics_phase(args):
    """Phase 4: 统计和分析评测结果"""
    import csv
    
    METRICS = [
        'ssim', 'lpips_distance', 'dino_similarity', 'clip_similarity',
        'text_similarity', 'comprehensive_visual_score'
    ]
    
    GLOBAL_METRICS = ['visual_score', 'success_ratio']
    
    def is_valid_sequence(metadata):
        """检查sequence是否有效"""
        sequence = metadata.get('sequence', [])
        successful_steps = metadata.get('successful_steps', 0)
        return len(sequence) > 0 and successful_steps > 0
    
    def load_all_metadata(base_dir):
        """加载所有任务的metadata.json文件"""
        all_data = []
        
        for task_dir in os.listdir(base_dir):
            metadata_path = os.path.join(base_dir, task_dir, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                continue
            
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    model_name = metadata.get('id', task_dir).split('_')[0]
                    
                    all_data.append({
                        'model': model_name,
                        'task_id': metadata.get('id', task_dir),
                        'metadata': metadata
                    })
            except Exception as e:
                print(f"Error loading {metadata_path}: {e}")
        
        return all_data
    
    def calculate_average_for_group(group, group_type, baseline_count=None):
        """计算一个组的指标平均值"""
        if not group:
            result = {metric: 0 for metric in METRICS}
            result['s_str'] = 0
            result['s_sty'] = 0
            for metric in GLOBAL_METRICS:
                result[metric] = 0
            return result
        
        task_metrics = []
        
        for metadata in group:
            sequence = metadata.get('sequence', [])
            task_avg = {}
            
            if sequence:
                for metric in METRICS:
                    values = [step.get(metric) for step in sequence if metric in step and step.get(metric) is not None]
                    if values:
                        task_avg[metric] = sum(values) / len(values)
                
                values_s_str = [step.get('s_str') for step in sequence if 's_str' in step and step.get('s_str') is not None]
                values_s_sty = [step.get('s_sty') for step in sequence if 's_sty' in step and step.get('s_sty') is not None]
                
                task_avg['s_str'] = sum(values_s_str) / len(values_s_str) if values_s_str else 0.0
                task_avg['s_sty'] = sum(values_s_sty) / len(values_s_sty) if values_s_sty else 0.0
            else:
                for metric in METRICS:
                    task_avg[metric] = 0.0
                task_avg['s_str'] = 0.0
                task_avg['s_sty'] = 0.0
            
            for metric in GLOBAL_METRICS:
                task_avg[metric] = metadata.get(metric, 0.0)
            
            task_metrics.append(task_avg)
        
        averages = {}
        
        for metric in METRICS:
            if group_type == 'all' and baseline_count is not None:
                total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
                averages[metric] = total_sum / baseline_count
            else:
                if task_metrics:
                    total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
                    averages[metric] = total_sum / len(task_metrics)
                else:
                    averages[metric] = 0.0
        
        for metric in ['s_str', 's_sty']:
            if group_type == 'all' and baseline_count is not None:
                total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
                averages[metric] = total_sum / baseline_count
            else:
                if task_metrics:
                    total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
                    averages[metric] = total_sum / len(task_metrics)
                else:
                    averages[metric] = 0.0
        
        for metric in GLOBAL_METRICS:
            if group_type == 'all' and baseline_count is not None:
                total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
                averages[metric] = total_sum / baseline_count
            else:
                if task_metrics:
                    total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
                    averages[metric] = total_sum / len(task_metrics)
                else:
                    averages[metric] = 0.0
        
        return averages
    
    def save_to_csv(results, output_path):
        """保存结果到CSV文件"""
        MODEL_ORDER = [
            'gemini-3-pro', 'gemini-3-flash', 'Doubao-Seed-1.8', 'doubao-seed-1-6',
            'gpt-4o', 'Qwen3-VL-235B', 'Qwen3-VL-30B', 'Qwen3-VL-8B'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            baseline_count = results[0]['baseline_count'] if results else 0
            
            header_all = ['Model', f'Count (Baseline: {baseline_count})', 'SSIM', 'LPIPS',
                          'DINO Similarity', 'S_str', 'CLIP', 'Text Similarity', 'S_sty',
                          'CVQI', 'Success Ratio', 'Visual Score']
            writer.writerow(header_all)
            
            def get_model_order(model_name):
                try:
                    return MODEL_ORDER.index(model_name)
                except ValueError:
                    return len(MODEL_ORDER)
            
            for result in sorted(results, key=lambda x: get_model_order(x['model'])):
                row = [result['model'], "100%"]
                metrics = result['all_metrics']
                row.extend([
                    f"{metrics.get('ssim', 0):.4f}",
                    f"{metrics.get('lpips_distance', 0):.4f}",
                    f"{metrics.get('dino_similarity', 0):.4f}",
                    f"{metrics.get('s_str', 0):.4f}",
                    f"{metrics.get('clip_similarity', 0):.4f}",
                    f"{metrics.get('text_similarity', 0):.4f}",
                    f"{metrics.get('s_sty', 0):.4f}",
                    f"{metrics.get('comprehensive_visual_score', 0):.4f}",
                    f"{metrics.get('success_ratio', 0):.4f}",
                    f"{metrics.get('visual_score', 0):.4f}"
                ])
                writer.writerow(row)
            
            writer.writerow([])
            
            header_valid = ['Model', 'Count (Valid)', 'SSIM', 'LPIPS', 'DINO Similarity',
                            'S_str', 'CLIP', 'Text Similarity', 'S_sty', 'CVQI',
                            'Success Ratio', 'Visual Score']
            writer.writerow(header_valid)
            
            for result in sorted(results, key=lambda x: get_model_order(x['model'])):
                percentage = (result['valid_count'] / baseline_count * 100) if baseline_count > 0 else 0
                row = [result['model'], f"{percentage:.2f}%"]
                metrics = result['valid_metrics']
                row.extend([
                    f"{metrics.get('ssim', 0):.4f}",
                    f"{metrics.get('lpips_distance', 0):.4f}",
                    f"{metrics.get('dino_similarity', 0):.4f}",
                    f"{metrics.get('s_str', 0):.4f}",
                    f"{metrics.get('clip_similarity', 0):.4f}",
                    f"{metrics.get('text_similarity', 0):.4f}",
                    f"{metrics.get('s_sty', 0):.4f}",
                    f"{metrics.get('comprehensive_visual_score', 0):.4f}",
                    f"{metrics.get('success_ratio', 0):.4f}",
                    f"{metrics.get('visual_score', 0):.4f}"
                ])
                writer.writerow(row)
        
        print(f"Results saved to {output_path}")
    
    base_dir = args.base_dir
    output_csv = args.output_csv
    
    print("Loading all metadata files...")
    all_data = load_all_metadata(base_dir)
    print(f"Loaded {len(all_data)} tasks")
    
    # 按模型分组
    model_data = defaultdict(lambda: {'all': [], 'valid': []})
    
    for item in all_data:
        model = item['model']
        metadata = item['metadata']
        model_data[model]['all'].append(metadata)
        if is_valid_sequence(metadata):
            model_data[model]['valid'].append(metadata)
    
    max_count = max(len(groups['all']) for groups in model_data.values())
    print(f"Baseline count: {max_count}")
    
    # 计算平均值
    results = []
    
    for model, groups in sorted(model_data.items()):
        all_metrics = calculate_average_for_group(groups['all'], 'all', max_count)
        valid_metrics = calculate_average_for_group(groups['valid'], 'valid')
        
        results.append({
            'model': model,
            'all_count': len(groups['all']),
            'valid_count': len(groups['valid']),
            'baseline_count': max_count,
            'all_metrics': all_metrics,
            'valid_metrics': valid_metrics
        })
    
    # 保存到CSV
    save_to_csv(results, output_csv)
    
    # 打印摘要
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal models: {len(results)}")
    print(f"Total tasks: {len(all_data)}")
    
    print("\n" + "-"*80)
    print("ALL DATA (including empty sequences)")
    print("-"*80)
    for result in sorted(results, key=lambda x: x['model']):
        print(f"\n{result['model']} ({result['all_count']} tasks):")
        metrics = result['all_metrics']
        print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
        print(f"  LPIPS: {metrics.get('lpips_distance', 0):.4f}")
        print(f"  DINO: {metrics.get('dino_similarity', 0):.4f}")
        print(f"  S_str: {metrics.get('s_str', 0):.4f}")
        print(f"  CLIP: {metrics.get('clip_similarity', 0):.4f}")
        print(f"  Text: {metrics.get('text_similarity', 0):.4f}")
        print(f"  S_sty: {metrics.get('s_sty', 0):.4f}")
        print(f"  CVQI: {metrics.get('comprehensive_visual_score', 0):.4f}")
        print(f"  Visual Score: {metrics.get('visual_score', 0):.4f}")
        print(f"  Success Ratio: {metrics.get('success_ratio', 0):.4f}")
    
    print("\n" + "-"*80)
    print("VALID DATA (only non-empty sequences)")
    print("-"*80)
    for result in sorted(results, key=lambda x: x['model']):
        print(f"\n{result['model']} ({result['valid_count']} valid tasks):")
        metrics = result['valid_metrics']
        print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
        print(f"  LPIPS: {metrics.get('lpips_distance', 0):.4f}")
        print(f"  DINO: {metrics.get('dino_similarity', 0):.4f}")
        print(f"  S_str: {metrics.get('s_str', 0):.4f}")
        print(f"  CLIP: {metrics.get('clip_similarity', 0):.4f}")
        print(f"  Text: {metrics.get('text_similarity', 0):.4f}")
        print(f"  S_sty: {metrics.get('s_sty', 0):.4f}")
        print(f"  CVQI: {metrics.get('comprehensive_visual_score', 0):.4f}")
        print(f"  Visual Score: {metrics.get('visual_score', 0):.4f}")
        print(f"  Success Ratio: {metrics.get('success_ratio', 0):.4f}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MultiInteract-Bench - 统一评测脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # Phase 1: Build - 使用模型复现数据集
  python scripts/benchmark.py build --limit 2 --models model-name-1 model-name-2
  
  # Phase 2: Record - 自动截图测试
  python scripts/benchmark.py record --headless
  
  # Phase 3: Compare - 计算视觉评测指标
  python scripts/benchmark.py compare
  
  # Phase 4: Metrics - 统计评测结果
  python scripts/benchmark.py metrics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='评测阶段')
    
    # Build phase
    build_parser = subparsers.add_parser('build', help='Phase 1: 使用模型复现数据集')
    build_parser.add_argument('--dataset-dir', type=str, default='./dataset_multi_turn',
                             help='数据集目录 (默认: ./dataset_multi_turn)')
    build_parser.add_argument('--output-vue-dir', type=str, default='./frontend/src/components/dataset',
                             help='Vue组件输出目录 (默认: ./frontend/src/components/dataset)')
    build_parser.add_argument('--output-config-dir', type=str, default='./outputs/reproduced_configs',
                             help='配置文件输出目录 (默认: ./outputs/reproduced_configs)')
    build_parser.add_argument('--limit', type=int, default=None,
                             help='限制处理的文件夹数量（用于测试）')
    build_parser.add_argument('--folder', type=str, default=None,
                             help='只处理指定的文件夹')
    build_parser.add_argument('--models', type=str, nargs='+', default=None,
                             help='只使用指定的模型')
    build_parser.add_argument('--skip-processed', action='store_true', default=True,
                             help='跳过已处理的任务（默认: True）')
    build_parser.add_argument('--no-skip-processed', action='store_false', dest='skip_processed',
                             help='重新处理所有任务')
    
    # Record phase
    record_parser = subparsers.add_parser('record', help='Phase 2: 自动截图测试')
    record_parser.add_argument('--base-url', type=str, default='http://localhost:1234',
                              help='前端服务URL (默认: http://localhost:1234)')
    record_parser.add_argument('--config-dir', type=str, default='./outputs/reproduced_configs',
                              help='配置文件目录 (默认: ./outputs/reproduced_configs)')
    record_parser.add_argument('--output-dir', type=str, default='./outputs/build_dataset_multi_turn',
                              help='截图输出目录 (默认: ./outputs/build_dataset_multi_turn)')
    record_parser.add_argument('--headless', action='store_true', default=True,
                              help='无头模式运行浏览器（默认: True）')
    record_parser.add_argument('--no-headless', action='store_false', dest='headless',
                              help='显示浏览器界面')
    record_parser.add_argument('--task-id', type=str, default=None,
                              help='只处理指定的任务ID')
    record_parser.add_argument('--model-prefix', type=str, default=None,
                              help='只处理指定模型前缀的任务')
    
    # Compare phase
    compare_parser = subparsers.add_parser('compare', help='Phase 3: 计算视觉评测指标')
    compare_parser.add_argument('--build-dir', type=str, default='./outputs/build_dataset_multi_turn',
                               help='生成的截图目录 (默认: ./outputs/build_dataset_multi_turn)')
    compare_parser.add_argument('--dataset-dir', type=str, default='./dataset_multi_turn',
                               help='原始数据集目录 (默认: ./dataset_multi_turn)')
    
    # Metrics phase
    metrics_parser = subparsers.add_parser('metrics', help='Phase 4: 统计评测结果')
    metrics_parser.add_argument('--base-dir', type=str, default='./outputs/build_dataset_multi_turn',
                               help='评测结果目录 (默认: ./outputs/build_dataset_multi_turn)')
    metrics_parser.add_argument('--output-csv', type=str, default='./outputs/model_metrics_summary.csv',
                               help='输出CSV文件路径 (默认: ./outputs/model_metrics_summary.csv)')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_phase(args)
    elif args.command == 'record':
        record_phase(args)
    elif args.command == 'compare':
        compare_phase(args)
    elif args.command == 'metrics':
        metrics_phase(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()