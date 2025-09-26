from pathlib import Path
from roadnetbenchmark.concurrent_jsonl_writer import load_jsonl
import pandas as pd
import numpy as np
from sklearn.metrics import (
    hamming_loss,
)

# 任务5：FPV车道数识别的评估结果计算
# 评估指标：将每个车道视为一个多标签分类任务，如果车道数不匹配则视为完全分类失败（因为真值车道数是已知的）
data_dir = Path("data/task_4_5")
jsonl_files = sorted(list(data_dir.glob("task5_*.jsonl")))

# 关于测试用例的额外标签
extra_label_file = Path("data/task45_extra_labels.jsonl")
extra_labels = load_jsonl(extra_label_file)
extra_labels = {label["id"]: label["extra_labels"] for label in extra_labels}

all_task5_results = []

for jsonl_file in jsonl_files:
    model_name = jsonl_file.stem
    print(f"\n=== Model: {model_name} ===")
    # result: {"llm_value": ["C", "C", "C"], "ground_truth": ["B", "C", "CD"]}
    # 含义: llm_value 是模型预测的车道，ground_truth 是真值车道
    # A: 掉头 B: 左转 C: 直行 D: 右转 G: 可变车道
    results = load_jsonl(jsonl_file)
    # 处理特殊数据，例如地面标识遮盖
    # results = [r for r in results if "光照不佳" in extra_labels.get(r["id"], [])]
    error_cnt = 0

    # 多标签分类指标计算
    all_lane_predictions = []
    all_lane_ground_truths = []

    for one in results:
        if (
            one["llm_value"] is None
            or one["llm_value"] == "None"
            or one["error"] is not None
            or not isinstance(one["llm_value"], list)
        ):

            error_cnt += 1
            one["llm_value"] = [""] * len(one["ground_truth"])

        # 检查车道数是否匹配
        if len(one["llm_value"]) != len(one["ground_truth"]):
            # error_cnt += 1
            one["llm_value"] = [""] * len(one["ground_truth"])

        # 将每个车道的预测和真值添加到列表中
        for pred_lane, true_lane in zip(one["llm_value"], one["ground_truth"]):
            # 将车道方向转换为多标签格式
            pred_labels = set(pred_lane) if isinstance(pred_lane, str) else set()
            true_labels = set(true_lane) if isinstance(true_lane, str) else set()

            # 为每个可能的方向创建二进制标签向量
            pred_vector = [
                1 if direction in pred_labels else 0
                for direction in ["A", "B", "C", "D", "G"]
            ]
            true_vector = [
                1 if direction in true_labels else 0
                for direction in ["A", "B", "C", "D", "G"]
            ]

            all_lane_predictions.append(pred_vector)
            all_lane_ground_truths.append(true_vector)

    print(f"error_cnt: {error_cnt}")

    if len(all_lane_predictions) > 0:
        # 转换为numpy数组
        pred_matrix = np.array(all_lane_predictions)
        true_matrix = np.array(all_lane_ground_truths)

        # 计算汉明距离
        hamming = hamming_loss(true_matrix, pred_matrix)
        print(f"Hamming Loss: {hamming:.4f}")

        # 计算准确匹配率（所有标签都正确的比例）
        exact_match = np.mean(np.all(pred_matrix == true_matrix, axis=1))
        print(f"Exact Match Ratio: {exact_match:.4f}")

        all_task5_results.append(
            {
                "model": model_name,
                "hamming_loss": hamming,
                "exact_match_ratio": exact_match,
            }
        )
    else:
        print("No valid predictions to evaluate")
        all_task5_results.append(
            {"model": model_name, "hamming_loss": 1.0, "exact_match_ratio": 0.0}
        )

# 添加规则基线
print("\n=== Baseline Rule-based ===")
results = load_jsonl(jsonl_files[0])  # 使用第一个文件的数据结构

# 多标签分类指标计算
all_lane_predictions = []
all_lane_ground_truths = []

for one in results:
    if (
        one["llm_value"] is None
        or one["llm_value"] == "None"
        or one["error"] is not None
        or not isinstance(one["llm_value"], list)
    ):
        continue

    num_lanes = len(one["ground_truth"])

    # 基于规则的预测
    if num_lanes == 1:
        rule_predictions = ["BCD"]
    elif num_lanes == 2:
        rule_predictions = ["BC", "CD"]
    else:  # 3车道或以上
        rule_predictions = ["B"]  # 最左边
        for i in range(1, num_lanes - 1):  # 中间的车道
            rule_predictions.append("C")
        rule_predictions.append("D")  # 最右边

    # 将每个车道的预测和真值添加到列表中
    for pred_lane, true_lane in zip(rule_predictions, one["ground_truth"]):
        # 将车道方向转换为多标签格式
        pred_labels = set(pred_lane) if isinstance(pred_lane, str) else set()
        true_labels = set(true_lane) if isinstance(true_lane, str) else set()

        # 为每个可能的方向创建二进制标签向量
        pred_vector = [
            1 if direction in pred_labels else 0 for direction in ["A", "B", "C", "D", "G"]
        ]
        true_vector = [
            1 if direction in true_labels else 0 for direction in ["A", "B", "C", "D", "G"]
        ]

        all_lane_predictions.append(pred_vector)
        all_lane_ground_truths.append(true_vector)

if len(all_lane_predictions) > 0:
    # 转换为numpy数组
    pred_matrix = np.array(all_lane_predictions)
    true_matrix = np.array(all_lane_ground_truths)

    # 计算汉明距离
    hamming = hamming_loss(true_matrix, pred_matrix)
    print(f"Hamming Loss: {hamming:.4f}")

    # 计算准确匹配率（所有标签都正确的比例）
    exact_match = np.mean(np.all(pred_matrix == true_matrix, axis=1))
    print(f"Exact Match Ratio: {exact_match:.4f}")

    all_task5_results.append(
        {
            "model": "baseline_rule_based",
            "hamming_loss": hamming,
            "exact_match_ratio": exact_match,
        }
    )

# 保存结果到CSV
task5_results_df = pd.DataFrame(all_task5_results)
task5_results_df.to_csv("result/task5_results.csv", index=False)
print("\n结果已保存到 task5_results.csv")
