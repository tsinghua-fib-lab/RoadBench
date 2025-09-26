from pathlib import Path
from roadnetbenchmark.concurrent_jsonl_writer import load_jsonl
import pandas as pd
from sklearn.metrics import (
    classification_report,
)

# 任务4：PV车道数识别的评估结果计算
# 评估指标：多分类指标、车道数与实际车道数的绝对平均偏差

# 关于测试用例的额外标签
extra_label_file = Path("data/task_4_5/dataset/extra_labels.jsonl")
extra_labels = load_jsonl(extra_label_file)
extra_labels = {label["id"]: label["extra_labels"] for label in extra_labels}

data_dir = Path("data/task_4_5")
jsonl_files = sorted(list(data_dir.glob("task4_*.jsonl")))

all_task4_results = []

for jsonl_file in jsonl_files:
    model_name = jsonl_file.stem
    print(f"\n=== Model: {model_name} ===")

    results = load_jsonl(jsonl_file)
    error_cnt = 0
    sum_squared_errors = 0  # 计算RMSE
    # 处理特殊数据，例如地面标识遮盖
    # results = [r for r in results if "地面标识遮盖" in extra_labels.get(r["id"], [])]
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
        sum_squared_errors += (one["llm_value"] - one["ground_truth"]) ** 2
    results_df = pd.DataFrame(results)

    print(f"error_cnt: {error_cnt}")

    # 计算分类指标
    report = classification_report(
        results_df["ground_truth"].to_list(),
        results_df["llm_value"].to_list(),
        zero_division=0.0,
        output_dict=True,
    )
    print(
        classification_report(
            results_df["ground_truth"].to_list(),
            results_df["llm_value"].to_list(),
            zero_division=0.0,
        )
    )

    # 计算RMSE
    rmse = (sum_squared_errors / len(results_df)) ** 0.5
    print(f"rmse: {rmse}")

    # 保存结果
    all_task4_results.append(
        {
            "model": model_name,
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "rmse": rmse,
        }
    )

# 添加baseline 1: 始终预测2车道
baseline_results = load_jsonl(jsonl_files[0])  # 使用第一个文件的数据结构
baseline_pred_2 = []
baseline_rmse_2 = 0
for one in baseline_results:
    baseline_pred_2.append(2)
    baseline_rmse_2 += (2 - one["ground_truth"]) ** 2

baseline_rmse_2 = (baseline_rmse_2 / len(baseline_results)) ** 0.5
baseline_report_2 = classification_report(
    [one["ground_truth"] for one in baseline_results],
    baseline_pred_2,
    zero_division=0.0,
    output_dict=True,
)

all_task4_results.append(
    {
        "model": "baseline_always_2_lanes",
        "weighted_precision": baseline_report_2["weighted avg"]["precision"],
        "weighted_recall": baseline_report_2["weighted avg"]["recall"],
        "weighted_f1": baseline_report_2["weighted avg"]["f1-score"],
        "rmse": baseline_rmse_2,
    }
)

# 添加baseline 2: 随机选择2/3/4车道
import random

random.seed(42)  # 保证结果可重现
baseline_pred_random = []
baseline_rmse_random = 0
for one in baseline_results:
    pred = random.choice([2, 3, 4])
    baseline_pred_random.append(pred)
    baseline_rmse_random += (pred - one["ground_truth"]) ** 2

baseline_rmse_random = (baseline_rmse_random / len(baseline_results)) ** 0.5
baseline_report_random = classification_report(
    [one["ground_truth"] for one in baseline_results],
    baseline_pred_random,
    zero_division=0.0,
    output_dict=True,
)

all_task4_results.append(
    {
        "model": "baseline_random_lanes",
        "weighted_precision": baseline_report_random["weighted avg"]["precision"],
        "weighted_recall": baseline_report_random["weighted avg"]["recall"],
        "weighted_f1": baseline_report_random["weighted avg"]["f1-score"],
        "rmse": baseline_rmse_random,
    }
)

# 保存结果到CSV
task4_results_df = pd.DataFrame(all_task4_results)
task4_results_df.to_csv("result/task4_results.csv", index=False)
print("\n结果已保存到 task4_results.csv")
