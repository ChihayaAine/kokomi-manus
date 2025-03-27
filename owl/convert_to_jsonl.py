import os
import json
from pathlib import Path
from loguru import logger

def convert_to_jsonl():
    """
    将 results/truth.json 文件转换为 JSONL 格式
    使用 ground_truth 作为 model_answer
    输出格式: {"task_id": "task_id_1", "model_answer": "正确答案"}
    """
    # 检查输入文件是否存在
    input_file = "results/truth.json"
    if not os.path.exists(input_file):
        logger.error(f"输入文件 {input_file} 不存在，请先运行 extract_gaia_truth.py")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = "results/submission_template.jsonl"
    
    # 读取 truth.json 文件
    with open(input_file, "r", encoding="utf-8") as f:
        truth_data = json.load(f)
    
    # 转换为 JSONL 格式，使用 ground_truth 作为 model_answer
    with open(output_file, "w", encoding="utf-8") as f:
        for item in truth_data:
            # 创建符合要求的格式
            jsonl_item = {
                "task_id": item["task_id"],
                "model_answer": item["ground_truth"]  # 使用 ground_truth 作为 model_answer
            }
            # 写入一行
            f.write(json.dumps(jsonl_item, ensure_ascii=False) + "\n")
    
    logger.success(f"成功将 {len(truth_data)} 条数据转换为 JSONL 格式，保存到 {output_file}")
    logger.info("JSONL 文件格式示例 (使用 ground_truth 作为 model_answer):")
    
    # 打印前几行作为示例
    with open(output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 3:  # 只打印前 3 行
                logger.info(line.strip())
            else:
                break
    
    logger.info(f"已将 ground_truth 作为 model_answer 保存到 {output_file}")

if __name__ == "__main__":
    convert_to_jsonl() 