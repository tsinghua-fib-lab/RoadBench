from pathlib import Path
from roadnetbenchmark.concurrent_jsonl_writer import load_jsonl
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
)

# 任务1：车道数识别的评估结果计算
# 评估指标：多分类指标、RMSE
data_dir = Path("data/task_1_2")
jsonl_files = sorted(list(data_dir.glob("task1_*.jsonl")))

# 存储所有结果
all_results = []

for jsonl_file in jsonl_files:
    model_name = jsonl_file.stem
    print(f"\n=== Model: {model_name} ===")

    results = load_jsonl(jsonl_file)
    error_cnt = 0
    squared_errors = []  # 计算RMSE
    for one in results:
        if (
            one["llm_value"] is None
            or one["llm_value"] == "None"
            or one["error"] is not None
        ):
            one["llm_value"] = 0
            error_cnt += 1
        if one["llm_value"] < 0:
            error_cnt += 1
            one["llm_value"] = 0
        squared_errors.append((one["llm_value"] - one["ground_truth"]) ** 2)

    results = pd.DataFrame(results)

    print(f"error_cnt: {error_cnt}")
    r = classification_report(
        results["ground_truth"].to_list(),
        results["llm_value"].to_list(),
        zero_division=0.0,
        output_dict=True,
    )

    # 计算RMSE
    rmse = np.sqrt(np.mean(squared_errors))

    print(f"Weighted precision: {r['weighted avg']['precision']}")
    print(f"Weighted recall: {r['weighted avg']['recall']}")
    print(f"Weighted f1: {r['weighted avg']['f1-score']}")
    print(f"RMSE: {rmse}")

    # 存储结果
    all_results.append(
        {
            "model": model_name,
            "weighted_precision": r["weighted avg"]["precision"],
            "weighted_recall": r["weighted avg"]["recall"],
            "weighted_f1": r["weighted avg"]["f1-score"],
            "rmse": rmse,
        }
    )

# 添加虚拟结果
# 1. 全选2车道
if len(all_results) > 0:
    # 获取第一个文件来计算虚拟结果
    first_jsonl = jsonl_files[0]
    results = load_jsonl(first_jsonl)
    ground_truths = [one["ground_truth"] for one in results]

    # 全选2车道
    all_2_predictions = [2] * len(ground_truths)
    squared_errors_2 = [
        (pred - gt) ** 2 for pred, gt in zip(all_2_predictions, ground_truths)
    ]
    rmse_2 = np.sqrt(np.mean(squared_errors_2))

    r_2 = classification_report(
        ground_truths, all_2_predictions, zero_division=0.0, output_dict=True
    )

    all_results.append(
        {
            "model": "baseline_all_2_lanes",
            "weighted_precision": r_2["weighted avg"]["precision"],
            "weighted_recall": r_2["weighted avg"]["recall"],
            "weighted_f1": r_2["weighted avg"]["f1-score"],
            "rmse": rmse_2,
        }
    )

    # 2. 随机选择
    random_predictions = [
        np.random.choice([2, 3, 4]) for _ in range(len(ground_truths))
    ]
    squared_errors_random = [
        (pred - gt) ** 2 for pred, gt in zip(random_predictions, ground_truths)
    ]
    rmse_random = np.sqrt(np.mean(squared_errors_random))

    r_avg = classification_report(
        ground_truths, random_predictions, zero_division=0.0, output_dict=True
    )

    all_results.append(
        {
            "model": "baseline_average_lanes",
            "weighted_precision": r_avg["weighted avg"]["precision"],
            "weighted_recall": r_avg["weighted avg"]["recall"],
            "weighted_f1": r_avg["weighted avg"]["f1-score"],
            "rmse": rmse_random,
        }
    )

# 保存结果到CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("result/task1_results.csv", index=False)
print("\n结果已保存到 task1_results.csv")
