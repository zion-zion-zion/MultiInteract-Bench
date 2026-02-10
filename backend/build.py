import os
import json
import base64
import time
from openai import OpenAI
from typing import List, Dict, Optional

# ============================================================================
# 配置部分
# ============================================================================

# 主流支持图片上传的大模型列表
MODEL_LIST = [
   
    {
        "name": "model-name",
        "model_id": "model-id",
        "api_key": "your-api-key",
        "base_url": "your-api-base-url",
        "description": "Brief description of the model and its capabilities"
        
    }

]

# 配置路径
DATASET_DIR = "./dataset_multi_turn"
OUTPUT_VUE_DIR = "./src/components/dataset"
OUTPUT_CONFIG_DIR = "./reproduced_configs"

# 创建输出目录
os.makedirs(OUTPUT_VUE_DIR, exist_ok=True)
os.makedirs(OUTPUT_CONFIG_DIR, exist_ok=True)

# ============================================================================
# 辅助函数
# ============================================================================

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
    
    # 读取 metadata
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"❌ Error reading metadata for {folder_name}: {e}")
        return None
    
    # 收集所有步骤的图片
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
    
    # 创建提示词
    text_prompt = create_prompt_from_metadata(metadata)
    
    # 构建消息
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
    
    # 添加图片到消息
    messages[0]["content"].extend(images_content)
    
    return {
        "folder_name": folder_name,
        "metadata": metadata,
        "messages": messages
    }

# ============================================================================
# 断点续传函数（简化版）
# ============================================================================

def is_processed(folder_name, model_name):
    """检查某个文件夹和模型是否已处理（通过检查配置文件是否存在）"""
    config_filename = f"{model_name}_{folder_name}.json"
    config_path = os.path.join(OUTPUT_CONFIG_DIR, config_filename)
    return os.path.exists(config_path)

# ============================================================================
# 核心处理函数
# ============================================================================

def call_llm_to_reproduce(processed_data, model_config):
    """调用大模型生成代码"""
    folder_name = processed_data["folder_name"]
    messages = processed_data["messages"]
    metadata = processed_data["metadata"]
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    
    # 生成新的文件名格式: {model_name}_{item}
    base_id = f"{model_name}_{folder_name}"
    
    print(f"🔄 Processing: {folder_name} with {model_name}")
    print(f"   - Steps: {len(metadata.get('sequence', []))}")
    print(f"   - Model: {model_id}")
    
    try:
        # 创建 OpenAI 客户端
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        # 调用 API
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            
            max_tokens=8000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # 保存 Vue 文件 (使用新格式: {model_name}_{item}.vue)
        vue_filename = f"{base_id}.vue"
        vue_path = os.path.join(OUTPUT_VUE_DIR, vue_filename)
        
        with open(vue_path, 'w', encoding='utf-8') as f:
            f.write(result.get("vue_code", ""))
        
        # 保存配置文件 (使用新格式: {model_name}_{item}.json)
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
            "original_dataset": f"dataset_multi_turn/{folder_name}"
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

def run_reproduction(
    limit=None, 
    specific_folder=None, 
    specific_models=None,
    skip_processed=True
):
    """
    执行复现任务
    
    Args:
        limit: 限制处理的文件夹数量（用于测试）
        specific_folder: 只处理指定的文件夹名称
        specific_models: 只处理指定的模型名称列表
        skip_processed: 是否跳过已处理的任务（检查配置文件是否存在）
    """
    # 确定要使用的模型列表
    if specific_models:
        models_to_use = [m for m in MODEL_LIST if m["name"] in specific_models]
        print(f"🎯 Using specific models: {specific_models}")
    else:
        models_to_use = MODEL_LIST
        print(f"📋 Total models to use: {len(models_to_use)}")
    
    # 显示模型列表
    print("\n🤖 Models:")
    for i, model in enumerate(models_to_use, 1):
        print(f"   {i}. {model['name']} - {model['description']}")
    print()
    
    # 遍历数据集文件夹
    folders = []
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path):
            if specific_folder and item != specific_folder:
                continue
            folders.append(item_path)
    
    if specific_folder and folders:
        print(f"🎯 Processing specific folder: {specific_folder}")
    else:
        print(f"📁 Total folders to process: {len(folders)}")
    
    if limit:
        folders = folders[:limit]
        print(f"⚠️  Limited to {limit} folders for testing")
    
    print(f"💾 Output directories:")
    print(f"   - Vue files: {OUTPUT_VUE_DIR}")
    print(f"   - Config files: {OUTPUT_CONFIG_DIR}")
    
    if skip_processed:
        print(f"✅ Skip processed tasks enabled (check existing config files)")
    else:
        print(f"⚠️  Skip processed tasks disabled (reprocess all)")
    
    print("-" * 60)
    
    # 统计信息
    total_tasks = len(folders) * len(models_to_use)
    processed_count = 0
    skipped_count = 0
    success_count = 0
    fail_count = 0
    
    # 遍历每个文件夹和模型组合
    task_number = 0
    for model in models_to_use:
        model_name = model["name"]
        
        for i, folder_path in enumerate(folders, 1):
            folder_name = os.path.basename(folder_path)
            task_number += 1
            
            # 检查是否已处理（通过检查配置文件是否存在）
            if skip_processed and is_processed(folder_name, model_name):
                print(f"\n[{task_number}/{total_tasks}] ⏭️  Skipping: {folder_name} with {model_name} (already processed)")
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
                
                # 防止 API 限流
                if task_number < total_tasks:
                    print("   ⏸️  Waiting 2 seconds before next request...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                fail_count += 1
                continue
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("📈 Reproduction Summary")
    print("=" * 60)
    print(f"Total tasks: {total_tasks}")
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"⏭️  Skipped (existing files): {skipped_count}")
    print(f"📊 New tasks processed: {processed_count}")
    print(f"\n📁 Output directories:")
    print(f"   - Vue files: {OUTPUT_VUE_DIR}")
    print(f"   - Config files: {OUTPUT_CONFIG_DIR}")
    print("=" * 60)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reproduce Vue components from dataset_multi_turn using multiple models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all folders with all models
  python build.py
  
  # Limit to 2 folders for testing
  python build.py --limit 2
  
  # Process specific folder
  python build.py --folder Apple_1766504959
  
  # Use specific models only
  python build.py --models Qwen3-Coder-480B Qwen3-VL-235B
  
  # Reprocess all tasks (ignore existing files)
  python build.py --no-skip-processed
  
  # List available models
  python build.py --list-models
        """
    )
    
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of folders to process (for testing)')
    parser.add_argument('--folder', type=str, default=None,
                       help='Process a specific folder by name')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Use specific models only (space-separated model names)')
    parser.add_argument('--skip-processed', action='store_true', default=True,
                       help='Skip already processed tasks (default: True)')
    parser.add_argument('--no-skip-processed', action='store_false', dest='skip_processed',
                       help='Reprocess all tasks (ignore existing files)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and exit')
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list_models:
        print("=" * 60)
        print("🤖 Available Models")
        print("=" * 60)
        for i, model in enumerate(MODEL_LIST, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Model ID: {model['model_id']}")
            print(f"   Description: {model['description']}")
            print(f"   API: {model['base_url']}")
        print("=" * 60)
        return
    
    # 执行复现任务
    run_reproduction(
        limit=args.limit,
        specific_folder=args.folder,
        specific_models=args.models,
        skip_processed=args.skip_processed
    )

if __name__ == "__main__":
    main()
