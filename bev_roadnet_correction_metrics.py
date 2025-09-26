# 任务3：道路网络修正的评估结果计算（归一化距离单位，折线归一化到0-1），多阈值（10%/20%/50%）

from pathlib import Path

import numpy as np
import pandas as pd
import shapely.wkt as wkt
from PIL import Image
from shapely import LineString, Point, wkt
from shapely.geometry import LineString, Point

from roadnetbenchmark.concurrent_jsonl_writer import load_jsonl
from roadnetbenchmark.metric import (
    buffer_f1_score_multi_line,
    match_junctions_by_distance,
    frechet_distance,
    match_linestrings,
)

# 构造task3的baseline
labels_file = Path("data/task_3/dataset/labels.jsonl")
labels = load_jsonl(labels_file)

baselines = []

for label in labels:
    pixel_line = wkt.loads(label["pixel_line"])
    start_point = Point(pixel_line.coords[0])
    end_point = Point(pixel_line.coords[-1])
    baseline_result = {
        "id": label["id"],
        "answer": "",
        "llm_value": {
            "junctions": [start_point.wkt, end_point.wkt],
            "lines": [pixel_line.wkt],
        },
        "ground_truth": label["ground_truth"],
        "error": None,
    }
    baselines.append(baseline_result)

# 保存baseline
with open("data/task_3/task3_donothing.jsonl", "w") as f:
    import json

    for baseline in baselines:
        f.write(json.dumps(baseline) + "\n")


def calculate_junction_metrics(
    pred_junctions, gt_junctions, distance_threshold, rmse_max_distance
):
    """
    计算Junction指标：召回率和RMSE
    """
    # 解析WKT格式的点
    pred_points = [wkt.loads(pt) for pt in pred_junctions]
    gt_points = [wkt.loads(pt) for pt in gt_junctions]

    # 匹配Junction点
    matches, unmatched_pred, unmatched_gt = match_junctions_by_distance(
        pred_points, gt_points, distance_threshold
    )

    # 计算召回率
    recall = len(matches) / len(gt_points) if gt_points else 0.0

    # 计算RMSE
    squared_errors = []

    # 匹配点的误差
    for pred_idx, gt_idx, dist in matches:
        squared_errors.append(dist**2)

    # 未匹配的预测点（假正例）误差设为最大值
    for _ in unmatched_pred:
        squared_errors.append(rmse_max_distance**2)

    # 未匹配的真值点（假负例）误差设为最大值
    for _ in unmatched_gt:
        squared_errors.append(rmse_max_distance**2)

    rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0

    return {
        "recall": recall,
        "rmse": rmse,
        "matched_pairs": len(matches),
        "total_gt": len(gt_points),
        "total_pred": len(pred_points),
        "unmatched_pred": len(unmatched_pred),
        "unmatched_gt": len(unmatched_gt),
    }


def normalize_point(pt, width, height):
    """将点归一化到[0,1]区间"""
    return Point(pt.x / width, pt.y / height)


def normalize_linestring(line, width, height):
    """将LineString归一化到[0,1]区间"""
    return LineString([(x / width, y / height) for x, y in line.coords])


def calculate_line_metrics(
    pred_lines, gt_lines, width, height, buffer_distance=0.1, frechet_max_distance=0.5
):
    """
    计算Line指标：折线归一化到[0,1]，buffer_distance和frechet_max_distance为归一化比例
    """
    if not pred_lines or not gt_lines:
        return {
            "buffer_f1": 0.0,
            "avg_frechet_distance": frechet_max_distance,
            "matched_line_pairs": 0,
            "total_gt_lines": len(gt_lines),
            "total_pred_lines": len(pred_lines),
            "unmatched_pred_lines": len(pred_lines),
            "unmatched_gt_lines": len(gt_lines),
        }

    try:
        # 解析WKT格式的线段，归一化到[0,1]
        pred_linestrings = []
        for line in pred_lines:
            geom = wkt.loads(line)
            if not isinstance(geom, LineString):
                print(f"Warning: {line} is not a LineString")
                continue
            norm_line = normalize_linestring(geom, width, height)
            pred_linestrings.append(norm_line)

        gt_linestrings = []
        for line in gt_lines:
            geom = wkt.loads(line)
            norm_line = normalize_linestring(geom, width, height)
            gt_linestrings.append(norm_line)

        # 1. 使用现有的buffer_f1_score_multi_line函数计算缓冲区F1分数
        buffer_f1 = buffer_f1_score_multi_line(
            pred_linestrings, gt_linestrings, buffer_distance
        )

        # 2. 使用现有的match_linestrings函数进行匹配
        matches, unmatched_pred, unmatched_gt = match_linestrings(
            pred_linestrings, gt_linestrings
        )

        # 3. 计算平均Frechet距离
        frechet_distances = []

        # 匹配的线段的Frechet距离
        for pred_idx, gt_idx in matches:
            try:
                dist = frechet_distance(
                    pred_linestrings[pred_idx], gt_linestrings[gt_idx]
                )
                frechet_distances.append(min(dist, frechet_max_distance))
            except Exception:
                frechet_distances.append(frechet_max_distance)

        # 未匹配的线段设为最大值
        for _ in unmatched_pred:
            frechet_distances.append(frechet_max_distance)

        for _ in unmatched_gt:
            frechet_distances.append(frechet_max_distance)

        avg_frechet = np.mean(frechet_distances) if frechet_distances else 0.0

        return {
            "buffer_f1": buffer_f1,
            "avg_frechet_distance": avg_frechet,
            "matched_line_pairs": len(matches),
            "total_gt_lines": len(gt_lines),
            "total_pred_lines": len(pred_lines),
            "unmatched_pred_lines": len(unmatched_pred),
            "unmatched_gt_lines": len(unmatched_gt),
        }

    except Exception as e:
        print(f"Error in calculate_line_metrics: {e}")
        return {
            "buffer_f1": 0.0,
            "avg_frechet_distance": frechet_max_distance,
            "matched_line_pairs": 0,
            "total_gt_lines": len(gt_lines),
            "total_pred_lines": len(pred_lines),
            "unmatched_pred_lines": len(pred_lines),
            "unmatched_gt_lines": len(gt_lines),
        }


print("Line指标计算函数已定义（归一化+拉伸）")

# 多阈值设置
junction_thresholds = [0.1, 0.2, 0.5]
buffer_thresholds = [0.1, 0.2, 0.5]
frechet_thresholds = [0.1, 0.2, 0.5]

data_dir = Path("data/task_3")
jsonl_files = sorted(list(data_dir.glob("task3_*.jsonl")))

all_task3_results = []

for jsonl_file in jsonl_files:
    model_name = jsonl_file.stem
    print(f"\n=== Model: {model_name} ===")

    results = load_jsonl(jsonl_file)
    error_cnt = 0

    # 累积指标
    # 每个阈值下都要分别统计
    junction_metrics_dict = {thresh: [] for thresh in junction_thresholds}
    line_metrics_dict = {thresh: [] for thresh in buffer_thresholds}

    for one in results:
        if (
            one["llm_value"] is None
            or one["error"] is not None
            or "lines" not in one["llm_value"]
            or "junctions" not in one["llm_value"]
        ):
            error_cnt += 1
            one["llm_value"] = {"junctions": [], "lines": []}

        try:
            # 加载图片，获取宽高
            img_path = Path("data/task_3/dataset") / f"{one['id']}.png"
            with Image.open(img_path) as img:
                width, height = img.size

            # Junction点归一化
            pred_junctions_norm = []
            for pt in one["llm_value"]["junctions"]:
                pt_obj = wkt.loads(pt)
                if not isinstance(pt_obj, Point):
                    print(f"Warning: {pt} is not a Point")
                    continue
                pt_norm = normalize_point(pt_obj, width, height)
                pred_junctions_norm.append(pt_norm.wkt)
            gt_junctions_norm = []
            for pt in one["ground_truth"]["junctions"]:
                pt_obj = wkt.loads(pt)
                pt_norm = normalize_point(pt_obj, width, height)
                gt_junctions_norm.append(pt_norm.wkt)

            # 针对每个阈值分别计算Junction和Line指标
            for thresh in junction_thresholds:
                junction_metrics = calculate_junction_metrics(
                    pred_junctions_norm,
                    gt_junctions_norm,
                    distance_threshold=thresh,
                    rmse_max_distance=thresh,
                )
                junction_metrics_dict[thresh].append(junction_metrics)

            for thresh in buffer_thresholds:
                # buffer_distance和frechet_max_distance都用同一个阈值
                line_metrics = calculate_line_metrics(
                    one["llm_value"]["lines"],
                    one["ground_truth"]["lines"],
                    width=width,
                    height=height,
                    buffer_distance=thresh,
                    frechet_max_distance=thresh,
                )
                line_metrics_dict[thresh].append(line_metrics)

        except Exception as e:
            print(f"Error processing sample {one.get('id', 'unknown')}: {e}")
            error_cnt += 1
            continue

    print(f"error_cnt: {error_cnt}")

    # 汇总每个阈值下的平均指标
    result_row = {
        "model": model_name,
        "error_count": error_cnt,
        "total_samples": len(results),
    }

    for thresh in junction_thresholds:
        metrics_list = junction_metrics_dict[thresh]
        if metrics_list:
            avg_recall = np.mean([m["recall"] for m in metrics_list])
            avg_rmse = np.mean([m["rmse"] for m in metrics_list])
        else:
            avg_recall = 0.0
            avg_rmse = thresh  # 归一化最大值
        # 例如：junction_recall_10p, junction_rmse_10p
        result_row[f"junction_recall_{int(thresh*100)}p"] = avg_recall
        result_row[f"junction_rmse_{int(thresh*100)}p"] = avg_rmse

    for thresh in buffer_thresholds:
        metrics_list = line_metrics_dict[thresh]
        if metrics_list:
            avg_buffer_f1 = np.mean([m["buffer_f1"] for m in metrics_list])
            avg_frechet_distance = np.mean(
                [m["avg_frechet_distance"] for m in metrics_list]
            )
        else:
            avg_buffer_f1 = 0.0
            avg_frechet_distance = thresh  # 归一化最大值
        # 例如：buffer_f1_10p, avg_frechet_distance_10p
        result_row[f"buffer_f1_{int(thresh*100)}p"] = avg_buffer_f1
        result_row[f"avg_frechet_distance_{int(thresh*100)}p"] = avg_frechet_distance

    # 打印主要指标
    for thresh in junction_thresholds:
        print(
            f"Junction Recall (归一化{int(thresh*100)}%): {result_row[f'junction_recall_{int(thresh*100)}p']:.4f}"
        )
        print(
            f"Junction RMSE (归一化{int(thresh*100)}%): {result_row[f'junction_rmse_{int(thresh*100)}p']:.4f}"
        )
    for thresh in buffer_thresholds:
        print(
            f"Buffer F1 (归一化{int(thresh*100)}%): {result_row[f'buffer_f1_{int(thresh*100)}p']:.4f}"
        )
        print(
            f"Average Frechet Distance (归一化{int(thresh*100)}%): {result_row[f'avg_frechet_distance_{int(thresh*100)}p']:.4f}"
        )

    all_task3_results.append(result_row)

print("\n任务3评估完成（归一化距离单位+折线归一化到0-1，多阈值）")

# 保存结果到CSV
task3_results_df = pd.DataFrame(all_task3_results)
task3_results_df.to_csv("result/task3_results.csv", index=False)
print("\n结果已保存到 task3_results.csv")
