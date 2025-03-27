import os
import json
from pathlib import Path
from loguru import logger
from utils import GAIABenchmark

def extract_gaia_truth():
    """
    提取 GAIA 验证集的查询和标准答案，并保存到 results/truth.json 文件中
    """
    # 创建 results 目录（如果不存在）
    results_dir = "results/"
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化 GAIA Benchmark
    local_gaia_path = "/Users/weilei/Desktop/llmeva/gaia/owl/owl/huggingface.co/datasets/gaia-benchmark/GAIA"
    benchmark = GAIABenchmark(
        data_dir="data/gaia",
        save_to="results/result.json"
    )
    
    # 从本地路径加载数据集
    benchmark.load(local_path=local_gaia_path)
    
    # 提取验证集的查询和标准答案
    truth_data = []
    for idx, task in enumerate(benchmark.valid):
        truth_item = {
            "idx": idx,
            "task_id": task["task_id"],
            "question": task["Question"],
            "level": task["Level"],
            "ground_truth": task["Final answer"],
            "tools": task["Annotator Metadata"].get("Tools", []),
            "category": task["Annotator Metadata"].get("Category", "")
        }
        truth_data.append(truth_item)
    
    # 保存到 results/truth.json 文件
    with open("results/truth.json", "w", encoding="utf-8") as f:
        json.dump(truth_data, f, indent=4, ensure_ascii=False)
    
    logger.success(f"成功提取 {len(truth_data)} 条验证集数据到 results/truth.json")
    logger.info(f"验证集包含 Level 1: {sum(1 for item in truth_data if item['level'] == 1)} 条")
    logger.info(f"验证集包含 Level 2: {sum(1 for item in truth_data if item['level'] == 2)} 条")
    logger.info(f"验证集包含 Level 3: {sum(1 for item in truth_data if item['level'] == 3)} 条")
    
    # 打印一些示例数据
    if truth_data:
        logger.info("示例数据:")
        for i in range(min(3, len(truth_data))):
            logger.info(f"问题 {i+1}: {truth_data[i]['question']}")
            logger.info(f"答案 {i+1}: {truth_data[i]['ground_truth']}")
            logger.info(f"工具 {i+1}: {truth_data[i]['tools']}")
            logger.info(f"类别 {i+1}: {truth_data[i]['category']}")
            logger.info("---")

if __name__ == "__main__":
    extract_gaia_truth() 