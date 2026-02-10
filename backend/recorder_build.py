import os
import json
import time
import glob
import sys
from PIL import Image, ImageStat
from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

# --- 配置 ---
BASE_URL = "http://localhost:1234"
CONFIG_DIR = "./reproduced_configs"
OUTPUT_DIR = "./build_dataset_multi_turn"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 调试选项：设置为 False 可以在步骤失败时停止执行（满足任务要求）
CONTINUE_ON_ERROR = False
# 调试选项：设置为 False 可以看到浏览器界面进行调试
HEADLESS_MODE = True

def load_configs(specific_task_id=None, model_prefix=None):
    """
    加载配置文件
    Args:
        specific_task_id: 指定要执行的任务ID（如 'Apple_1766504899'），如果为None则加载所有任务
        model_prefix: 指定模型名称前缀（如 'Apple', 'Discord', 'Qwen-VL-8B'），只加载以此前缀开头的任务
    """
    tasks = []
    # 读取 CONFIG_DIR 文件夹下所有的 json
    files = glob.glob(os.path.join(CONFIG_DIR, "*.json"))
    
    for f in files:
        with open(f, 'r') as file:
            task = json.load(file)
            task_id = task.get('id', '')
            
            # 过滤逻辑：
            # 1. 如果指定了 specific_task_id，只匹配完全相等的任务ID
            # 2. 否则如果指定了 model_prefix，匹配以此前缀开头的任务ID
            # 3. 否则加载所有任务
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
    """
    检查图片是否为空白（完全是一种颜色，特别是白色）
    Args:
        image_path: 图片文件路径
    Returns:
        bool: 如果图片为空白返回True，否则返回False
    """
    try:
        # 确保文件存在且可读
        if not os.path.exists(image_path):
            print(f"  [Warning] Image file does not exist: {image_path}")
            return False
        
        # 打开图片
        img = Image.open(image_path)
        # 转换为RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 计算图片的标准差（注意：Pillow中使用的是stddev而不是stdev）
        stat = ImageStat.Stat(img)
        std_dev = stat.stddev  # 正确的属性名
        
        # 计算标准差的总和
        std_dev_sum = sum(std_dev) if isinstance(std_dev, (list, tuple)) else std_dev
        
        # 获取平均颜色
        mean_color = stat.mean
        
        print(f"  [Debug] Image analysis - std_dev: {std_dev_sum:.4f}, mean_color: {mean_color}")
        
        # 检测条件：
        # 1. 标准差很小（颜色变化少）
        # 2. 平均颜色接近白色（255, 255, 255）
        is_low_variance = std_dev_sum < 3.0  # 增加阈值以确保检测
        is_mostly_white = all(c > 240 for c in mean_color)  # 白色检测阈值
        
        is_blank = is_low_variance and is_mostly_white
        
        print(f"  [Debug] Blank detection - low_variance: {is_low_variance}, mostly_white: {is_mostly_white}, is_blank: {is_blank}")
        
        return is_blank
        
    except Exception as e:
        print(f"  [Warning] Error checking blank image: {e}")
        import traceback
        traceback.print_exc()
        return False

def has_navigation_error(task_dir):
    """
    检查任务目录中的 metadata.json 是否包含导航失败错误
    Args:
        task_dir: 任务目录路径
    Returns:
        bool: 如果存在导航失败错误返回True，否则返回False
    """
    try:
        metadata_path = os.path.join(task_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return False
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 检查是否存在 error 字段且包含 "Navigation failed:"
        error_field = metadata.get('error', '')
        if error_field and 'Navigation failed:' in str(error_field):
            print(f"  [Info] Found navigation error in metadata: {error_field}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  [Warning] Error checking navigation error: {e}")
        return False

def run_recorder(specific_task_id=None, model_prefix=None):
    """
    执行拍照程序
    Args:
        specific_task_id: 指定要执行的任务ID（如 'Apple_1766504899'）
        model_prefix: 指定模型名称前缀（如 'Apple', 'Discord', 'Qwen-VL-8B'）
        如果两个参数都为None，则执行所有任务
    """
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    tasks = load_configs(specific_task_id, model_prefix)
    
    if len(tasks) == 0:
        if specific_task_id:
            print(f"[Error] Task '{specific_task_id}' not found in {CONFIG_DIR}")
        elif model_prefix:
            print(f"[Error] No tasks found with model prefix '{model_prefix}' in {CONFIG_DIR}")
        else:
            print(f"[Warning] No tasks found in {CONFIG_DIR}")
        return
    
    if specific_task_id:
        print(f"Loaded 1 task: {specific_task_id}")
    elif model_prefix:
        print(f"Loaded {len(tasks)} tasks with model prefix: {model_prefix}")
    else:
        print(f"Loaded {len(tasks)} multi-step tasks.")

    with sync_playwright() as p:
        # 使用配置的 headless 模式
        browser = p.chromium.launch(headless=HEADLESS_MODE)
        context = browser.new_context(viewport={"width": 1000, "height": 800})
        page = context.new_page()

        for task in tasks:
            print(f"\n--- Processing: {task['id']} ---")
            
            # 1. 准备输出目录
            task_dir = os.path.join(OUTPUT_DIR, task['id'])
            
            # 检查目录是否存在
            if os.path.exists(task_dir):
                # 检查是否存在导航错误，如果存在则重新执行
                if has_navigation_error(task_dir):
                    print(f"  [Info] Directory exists but contains navigation error, reprocessing...")
                    # 清空目录以便重新截图
                    import shutil
                    try:
                        shutil.rmtree(task_dir)
                        os.makedirs(task_dir)
                        print(f"  [Info] Cleaned directory for reprocessing")
                    except Exception as e:
                        print(f"  [Error] Failed to clean directory: {e}")
                        continue
                else:
                    print(f"  Directory {task_dir} already exists and has no navigation error, skipping...")
                    continue
            else:
                os.makedirs(task_dir)

            # 2. 访问组件
            url = f"{BASE_URL}/?component={task['url_param']}"
            print(f"Navigating to {url}")
            
            try:
                page.goto(url)
            except Exception as goto_error:
                print(f"  [Error] Failed to navigate to {url}: {goto_error}")
                print(f"  [Info] Recording successful steps: 0/{len(task['steps'])}")
                
                # 即使导航失败，也尝试截图（可能显示错误信息）
                try:
                    time.sleep(1)
                    page.locator("body").screenshot(path=f"{task_dir}/step_00.png")
                    print(f"  [Debug] Saved screenshot after navigation error")
                except Exception as screenshot_error:
                    print(f"  [Warning] Failed to capture screenshot: {screenshot_error}")
                
                # 保存 Metadata（成功步数为0）
                source_code = ""
                try:
                    src_path = os.path.join(PROJECT_ROOT, task['source_file'])
                    with open(src_path, "r") as f: source_code = f.read()
                except:
                    source_code = "// Source file not found"
                
                description = task.get('description', '')
                if 'meta' in task:
                    meta = task['meta']
                    description = f"{description} | Flow: {meta.get('flow', '')} | Style: {meta.get('style', '')} | Context: {meta.get('context', '')}"
                
                final_metadata = {
                    "id": task['id'],
                    "description": description,
                    "ground_truth_code": source_code,
                    "total_planned_steps": len(task['steps']),
                    "successful_steps": 0,
                    "success_ratio": 0.0,
                    "sequence": [],
                    "error": f"Navigation failed: {str(goto_error)}"
                }
                
                if 'meta' in task:
                    final_metadata['meta'] = task['meta']
                
                with open(f"{task_dir}/metadata.json", "w") as f:
                    json.dump(final_metadata, f, indent=2)
                
                print(f"Saved {task['id']} with 0 steps (navigation failed).")
                continue  # 跳过当前任务，继续下一个
            
            # 等待组件加载完成，等待 body 元素确保页面加载
            try:
                page.wait_for_selector("body", timeout=10000)
                print("  Page loaded successfully")
            except Exception as e:
                print(f"  [Warning] Page load timeout: {e}")
            
            # 额外等待确保 Vue 组件渲染完成
            time.sleep(2.0)

            # 3. 初始状态截图 (Step 0)
            print("Capturing Step 0 (Initial State)...")
            # 这里我用了 body 截图，你也可以像之前一样指定 task['capture_selector']
            page.locator("body").screenshot(path=f"{task_dir}/step_00.png")
            
            # 确保文件已写入
            time.sleep(0.5)
            
            # 检查是否有 Vite 报错弹窗
            print("  [Check] Checking for Vite error popup...")
            vite_error_detected = False
            
            # Vite 错误弹窗通常包含特定的文本或类名
            # 尝试检测多种可能的 Vite 错误指示器
            vite_error_selectors = [
                "text=Vite",
                "text=Failed to compile",
                "text=Internal server error",
                "text=Error",
                ".vite-error",
                "[data-vite-dev-id]"
            ]
            
            for selector in vite_error_selectors:
                try:
                    # 检查页面中是否存在这些错误指示器
                    elements = page.locator(selector)
                    count = elements.count()
                    if count > 0:
                        # 检查元素是否可见
                        for i in range(count):
                            if elements.nth(i).is_visible(timeout=500):
                                print(f"  [Error] Vite error detected with selector: {selector}")
                                vite_error_detected = True
                                break
                    if vite_error_detected:
                        break
                except Exception as check_error:
                    # 检查失败继续检查其他选择器
                    pass
            
            if vite_error_detected:
                print(f"  [Error] Vite error popup detected. Terminating task.")
                print(f"  [Info] Recording successful steps: 0/{len(task['steps'])}")
                
                # 保存 Metadata（成功步数为0）
                source_code = ""
                try:
                    src_path = os.path.join(PROJECT_ROOT, task['source_file'])
                    with open(src_path, "r") as f: source_code = f.read()
                except:
                    source_code = "// Source file not found"
                
                description = task.get('description', '')
                if 'meta' in task:
                    meta = task['meta']
                    description = f"{description} | Flow: {meta.get('flow', '')} | Style: {meta.get('style', '')} | Context: {meta.get('context', '')}"
                
                final_metadata = {
                    "id": task['id'],
                    "description": description,
                    "ground_truth_code": source_code,
                    "total_planned_steps": len(task['steps']),
                    "successful_steps": 0,
                    "success_ratio": 0.0,
                    "sequence": [],
                    "error": "Vite error popup detected"
                }
                
                if 'meta' in task:
                    final_metadata['meta'] = task['meta']
                
                with open(f"{task_dir}/metadata.json", "w") as f:
                    json.dump(final_metadata, f, indent=2)
                
                print(f"Saved {task['id']} with 0 steps (Vite error detected).")
                continue  # 跳过当前任务，继续下一个
            
            # 检查初始截图是否为空白
            print("  [Check] Checking if initial screenshot is blank...")
            is_blank = is_blank_image(f"{task_dir}/step_00.png")
            
            if is_blank:
                print(f"  [Error] Initial screenshot is blank. Terminating task.")
                print(f"  [Info] Recording successful steps: 0/{len(task['steps'])}")
                
                # 保存 Metadata（成功步数为0）
                source_code = ""
                try:
                    src_path = os.path.join(PROJECT_ROOT, task['source_file'])
                    with open(src_path, "r") as f: source_code = f.read()
                except:
                    source_code = "// Source file not found"
                
                description = task.get('description', '')
                if 'meta' in task:
                    meta = task['meta']
                    description = f"{description} | Flow: {meta.get('flow', '')} | Style: {meta.get('style', '')} | Context: {meta.get('context', '')}"
                
                final_metadata = {
                    "id": task['id'],
                    "description": description,
                    "ground_truth_code": source_code,
                    "total_planned_steps": len(task['steps']),
                    "successful_steps": 0,
                    "success_ratio": 0.0,
                    "sequence": [],
                    "error": "Initial screenshot was blank"
                }
                
                if 'meta' in task:
                    final_metadata['meta'] = task['meta']
                
                with open(f"{task_dir}/metadata.json", "w") as f:
                    json.dump(final_metadata, f, indent=2)
                
                print(f"Saved {task['id']} with 0 steps (blank initial screenshot).")
                continue  # 跳过当前任务，继续下一个

            # 4. 循环执行 Steps
            executed_steps = []
            
            # 处理 steps 可能是字符串的情况（JSON 字符串需要解析）
            steps = task['steps']
            if isinstance(steps, str):
                print(f"  [Warning] 'steps' is a string, attempting to parse as JSON...")
                try:
                    # 尝试解析 JSON 字符串
                    # 移除可能的前缀（如 "playwright_steps": "）
                    steps_json = steps
                    if '"playwright_steps":' in steps:
                        # 提取 JSON 数组部分
                        start = steps.find('[')
                        end = steps.rfind(']') + 1
                        if start >= 0 and end > start:
                            steps_json = steps[start:end]
                    
                    steps = json.loads(steps_json)
                    print(f"  [Info] Successfully parsed steps, found {len(steps)} steps")
                except Exception as parse_error:
                    print(f"  [Error] Failed to parse steps as JSON: {parse_error}")
                    print(f"  [Debug] Steps content: {steps[:200]}...")  # 打印前200个字符
                    continue  # 跳过此任务
            
            for i, step in enumerate(steps):
                step_num = i + 1
                
                # 检查 step 是否为字典
                if not isinstance(step, dict):
                    print(f"  [Warning] Step {step_num} is not a dictionary, type={type(step)}")
                    print(f"  [Debug] Step content: {step}")
                    continue
                
                # 安全获取描述，如果不存在则使用默认值
                step_desc = step.get('desc', 'No description')
                print(f"  Step {step_num}: {step_desc}")
                
                try:
                    # 执行动作
                    # 兼容新旧格式：新格式用 'step'，旧格式用 'order'
                    step_field = step.get('step', step.get('order', i + 1))
                    selector = step.get('selector')  # 新格式中 wait 动作可能没有 selector
                    # 安全获取 action_type，如果不存在则跳过此步骤
                    action_type = step.get('action')
                    if not action_type:
                        print(f"    [Warning] Step {step_num} missing 'action' field, skipping...")
                        continue
                    
                    if action_type == 'click':
                        if selector:
                            print(f"    [Debug] Attempting to click: {selector}")
                            try:
                                # 等待元素可见和可点击
                                page.wait_for_selector(selector, timeout=5000, state='visible')
                                page.click(selector)
                                print(f"    [Debug] Successfully clicked: {selector}")
                            except Exception as click_error:
                                print(f"    [Error] Failed to click {selector}: {click_error}")
                                # 尝试使用 JavaScript 点击作为备选方案
                                try:
                                    element = page.query_selector(selector)
                                    if element:
                                        element.click()
                                        print(f"    [Debug] Successfully clicked using JS fallback: {selector}")
                                    else:
                                        raise Exception("Element not found")
                                except Exception as js_error:
                                    print(f"    [Error] JS fallback also failed: {js_error}")
                                    raise js_error
                        else:
                            print(f"    [Warning] Click action without selector, skipping...")
                    elif action_type == 'fill':
                        if selector:
                            print(f"    [Debug] Attempting to fill: {selector} with value: {step.get('value', '')}")
                            try:
                                page.wait_for_selector(selector, timeout=5000, state='visible')
                                page.fill(selector, step['value'])
                                print(f"    [Debug] Successfully filled: {selector}")
                            except Exception as fill_error:
                                print(f"    [Error] Failed to fill {selector}: {fill_error}")
                                raise fill_error
                        else:
                            print(f"    [Warning] Fill action without selector, skipping...")
                    elif action_type == 'type':
                        if selector:
                            print(f"    [Debug] Attempting to type: {selector} with value: {step.get('value', '')}")
                            try:
                                page.wait_for_selector(selector, timeout=5000, state='visible')
                                page.type(selector, step['value'])
                                print(f"    [Debug] Successfully typed: {selector}")
                            except Exception as type_error:
                                print(f"    [Error] Failed to type {selector}: {type_error}")
                                raise type_error
                        else:
                            print(f"    [Warning] Type action without selector, skipping...")
                    elif action_type == 'hover':
                        if selector:
                            print(f"    [Debug] Attempting to hover: {selector}")
                            try:
                                page.wait_for_selector(selector, timeout=5000, state='visible')
                                page.hover(selector)
                                print(f"    [Debug] Successfully hovered: {selector}")
                            except Exception as hover_error:
                                print(f"    [Error] Failed to hover {selector}: {hover_error}")
                                raise hover_error
                        else:
                            print(f"    [Warning] Hover action without selector, skipping...")
                    elif 'wait' in action_type:
                        # wait 动作可能没有 value 字段，使用默认值
                        wait_time = step.get('value', 1)
                        print(f"    [Debug] Waiting for {wait_time} seconds (action type: {action_type})")
                        time.sleep(1)
                    elif action_type == 'mouse_down_and_move':
                        if selector:
                            print(f"    [Debug] Attempting to mouse down and move: {selector}")
                            try:
                                page.wait_for_selector(selector, timeout=5000, state='visible')
                                # 模拟鼠标按下并移动
                                element = page.locator(selector)
                                element.hover()
                                page.mouse.down()
                                print(f"    [Debug] Successfully mouse down and moved: {selector}")
                            except Exception as error:
                                print(f"    [Error] Failed to mouse down and move {selector}: {error}")
                                raise error
                        else:
                            print(f"    [Warning] mouse_down_and_move action without selector, skipping...")
                    elif action_type == 'mouse_up':
                        if selector:
                            print(f"    [Debug] Attempting to mouse up: {selector}")
                            try:
                                page.wait_for_selector(selector, timeout=5000, state='visible')
                                # 模拟鼠标抬起
                                element = page.locator(selector)
                                element.hover()
                                page.mouse.up()
                                print(f"    [Debug] Successfully mouse up: {selector}")
                            except Exception as error:
                                print(f"    [Error] Failed to mouse up {selector}: {error}")
                                raise error
                        else:
                            print(f"    [Warning] mouse_up action without selector, skipping...")
                    elif action_type == 'dragAndDrop':
                        # dragAndDrop 需要两个选择器：source 和 target
                        source_selector = step.get('source')
                        target_selector = step.get('target')
                        if source_selector and target_selector:
                            print(f"    [Debug] Attempting to drag from {source_selector} to {target_selector}")
                            try:
                                page.wait_for_selector(source_selector, timeout=5000, state='visible')
                                page.wait_for_selector(target_selector, timeout=5000, state='visible')
                                # 使用 Playwright 的 drag_and_drop 方法
                                page.drag_and_drop(source_selector, target_selector)
                                print(f"    [Debug] Successfully dragged from {source_selector} to {target_selector}")
                            except Exception as error:
                                print(f"    [Error] Failed to drag from {source_selector} to {target_selector}: {error}")
                                raise error
                        else:
                            print(f"    [Warning] dragAndDrop action requires both 'source' and 'target' selectors, skipping...")
                    elif action_type == 'assert_visible':
                        if selector:
                            print(f"    [Debug] Asserting visible: {selector}")
                            try:
                                # 等待元素可见，如果在超时时间内不可见则抛出异常
                                page.wait_for_selector(selector, timeout=5000, state='visible')
                                print(f"    [Debug] Successfully asserted visible: {selector}")
                            except Exception as error:
                                print(f"    [Error] Assertion failed, element not visible: {selector}")
                                raise error
                        else:
                            print(f"    [Warning] assert_visible action without selector, skipping...")
                    
                    # 动作后等待一小会儿，让 Vue 反应过来 (动画/DOM更新)
                    time.sleep(1)
                    
                    # 截图当前步骤结果
                    screenshot_name = f"step_{step_num:02d}.png" # 生成 step_01.png
                    page.locator("body").screenshot(path=f"{task_dir}/{screenshot_name}")
                    
                    # 记录成功执行的步骤
                    executed_steps.append({
                        "step_index": step_num,
                        "action": step,
                        "image": screenshot_name
                    })
                    
                except (PlaywrightTimeoutError, Exception) as e:
                    error_type = type(e).__name__
                    print(f"  [Error] Step {step_num} failed: {error_type} - {e}")
                    print(f"  [Info] Recording successful steps: {len(executed_steps)}/{len(task['steps'])}")
                    
                    # 保存调试信息
                    try:
                        debug_screenshot = f"{task_dir}/debug_step_{step_num}_error.png"
                        page.locator("body").screenshot(path=debug_screenshot)
                        print(f"  [Debug] Saved error screenshot: {debug_screenshot}")
                        
                        # 保存页面 HTML 内容
                        debug_html = f"{task_dir}/debug_step_{step_num}_error.html"
                        with open(debug_html, "w", encoding="utf-8") as f:
                            f.write(page.content())
                        print(f"  [Debug] Saved page HTML: {debug_html}")
                        
                        # 尝试获取所有可能的相关元素
                        print(f"  [Debug] Looking for elements that might match...")
                        if action_type == 'click' and selector:
                            # 尝试找到部分匹配的元素
                            try:
                                # 移除一些可能过于具体的选择器部分
                                simplified_selector = selector.split(':')[0]  # 移除 :first-child 等
                                elements = page.query_selector_all(simplified_selector)
                                print(f"  [Debug] Found {len(elements)} elements with simplified selector: {simplified_selector}")
                                
                                # 尝试更宽泛的搜索
                                if 'bg-gray-900' in selector:
                                    gray_elements = page.query_selector_all('.bg-gray-900')
                                    print(f"  [Debug] Found {len(gray_elements)} elements with .bg-gray-900")
                                if 'cursor-pointer' in selector:
                                    pointer_elements = page.query_selector_all('.cursor-pointer')
                                    print(f"  [Debug] Found {len(pointer_elements)} elements with .cursor-pointer")
                            except Exception as debug_error:
                                print(f"  [Debug] Debug search failed: {debug_error}")
                                
                    except Exception as debug_error:
                        print(f"  [Debug] Failed to save debug info: {debug_error}")
                    
                    # 遇到错误直接停止当前任务，保存已完成的步骤
                    print(f"  [Info] Stopping current task and moving to next task")
                    print(f"  [Info] Successfully recorded {len(executed_steps)} steps out of {len(task['steps'])} total steps")
                    break

            # 5. 保存 Metadata
            # 读取源码
            source_code = ""
            try:
                src_path = os.path.join(PROJECT_ROOT, task['source_file'])
                with open(src_path, "r") as f: source_code = f.read()
            except:
                source_code = "// Source file not found"

            # 兼容新旧格式的描述字段
            description = task.get('description', '')
            # 如果新格式有 meta 字段，可以从中提取更多信息
            if 'meta' in task:
                meta = task['meta']
                # 可以将 meta 信息添加到描述中或单独保存
                description = f"{description} | Flow: {meta.get('flow', '')} | Style: {meta.get('style', '')} | Context: {meta.get('context', '')}"

            # 计算成功步数与总步数的比值
            # 使用解析后的 steps 变量，而不是原始的 task['steps']
            total_planned_steps = len(steps) if steps else 0
            successful_steps = len(executed_steps)
            success_ratio = successful_steps / total_planned_steps if total_planned_steps > 0 else 0
            
            final_metadata = {
                "id": task['id'],
                "description": description,
                "ground_truth_code": source_code,
                "total_planned_steps": total_planned_steps,  # 计划的总步数
                "successful_steps": successful_steps,       # 实际成功执行的步数
                "success_ratio": round(success_ratio, 4),     # 成功步数/总步数的比值（保留4位小数）
                "sequence": executed_steps
            }
            
            # 如果有 meta 字段，也保存到 metadata 中
            if 'meta' in task:
                final_metadata['meta'] = task['meta']

            with open(f"{task_dir}/metadata.json", "w") as f:
                json.dump(final_metadata, f, indent=2)
            
            print(f"Saved {task['id']} with {len(executed_steps)} steps.")

        browser.close()

if __name__ == "__main__":
    # 支持命令行参数
    # 用法 1: python recorder_build.py                    # 执行所有任务
    # 用法 2: python recorder_build.py Apple_1766504899   # 执行特定任务ID
    # 用法 3: python recorder_build.py --model Apple      # 执行以Apple开头的所有任务
    # 用法 4: python recorder_build.py --model Qwen-VL-8B  # 执行以Qwen-VL-8B开头的所有任务
    
    specific_task_id = None
    model_prefix = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--model" and len(sys.argv) > 2:
            # 按模型前缀过滤
            model_prefix = sys.argv[2]
            print(f"[Info] Running tasks with model prefix: {model_prefix}")
        else:
            # 按具体任务ID
            specific_task_id = sys.argv[1]
            print(f"[Info] Running specific task: {specific_task_id}")
    
    run_recorder(specific_task_id, model_prefix)
