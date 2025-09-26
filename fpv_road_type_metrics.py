from pathlib import Path
from roadnetbenchmark.concurrent_jsonl_writer import load_jsonl
import pandas as pd
from sklearn.metrics import (
    classification_report,
)

# 任务6：FPV主辅路分类
# 评估指标：二分类指标
data_dir = Path("data/task_6")
jsonl_files = sorted(list(data_dir.glob("task6_*.jsonl")))

all_task6_results = []

for jsonl_file in jsonl_files:
    model_name = jsonl_file.stem
    print(f"\n=== Model: {model_name} ===")

    results = load_jsonl(jsonl_file)
    error_cnt = 0
    for one in results:
        if (
            one["llm_value"] is None
            or one["llm_value"] not in ["main", "service"]
            or one["error"] is not None
        ):
            one["llm_value"] = "unknown"
            error_cnt += 1
    results = pd.DataFrame(results)

    print(f"error_cnt: {error_cnt}")
    r = classification_report(
        results["ground_truth"].to_list(),
        results["llm_value"].to_list(),
        zero_division=0.0,
        output_dict=True,
    )
    print(
        classification_report(
            results["ground_truth"].to_list(),
            results["llm_value"].to_list(),
            zero_division=0.0,
        )
    )

    # 提取指标
    accuracy = r["accuracy"]
    weighted_recall = r["weighted avg"]["recall"]
    weighted_f1 = r["weighted avg"]["f1-score"]

    all_task6_results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        }
    )

# 添加随机选择基线
import random
random.seed(42)
baseline_results = []
for _, row in pd.DataFrame(load_jsonl(jsonl_files[0])).iterrows():
    baseline_results.append(random.choice(["main", "service"]))

ground_truth = [one["ground_truth"] for one in load_jsonl(jsonl_files[0])]
r_baseline = classification_report(
    ground_truth, baseline_results, zero_division=0.0, output_dict=True
)

print(f"\n=== Baseline: Random Choice ===")
print(classification_report(ground_truth, baseline_results, zero_division=0.0))

all_task6_results.append(
    {
        "model": "baseline_random_choice",
        "accuracy": r_baseline["accuracy"],
        "weighted_recall": r_baseline["weighted avg"]["recall"],
        "weighted_f1": r_baseline["weighted avg"]["f1-score"],
    }
)

# 保存结果到CSV
task6_results_df = pd.DataFrame(all_task6_results)
task6_results_df.to_csv("result/task6_results.csv", index=False)
print("\n结果已保存到 task6_results.csv")
