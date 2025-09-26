from typing import List, Literal

import numpy as np
import similaritymeasures
from scipy.optimize import linear_sum_assignment
from shapely import LineString, MultiLineString, Point

__all__ = [
    "buffer_f1_score",
    "buffer_f1_score_multi_line",
    "endpoint_distance",
    "endpoint_distance_multi_line",
    "frechet_distance",
    "match_linestrings",
]


def euclidean_distance(point1, point2):
    """计算两点之间的欧几里得距离"""
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def buffer_f1_score(
    line1_xy: LineString,
    line2_xy: LineString,
    threshold_meters: float,
) -> float:
    """计算两个折线之间的缓冲区F1分数"""

    buffer1 = line1_xy.buffer(threshold_meters)
    buffer2 = line2_xy.buffer(threshold_meters)

    # 计算交集和并集
    intersection = buffer1.intersection(buffer2)
    union = buffer1.union(buffer2)

    if union.area == 0:
        return 0.0

    # 计算IoU
    iou = intersection.area / union.area

    # 转换为F1分数（IoU = F1/(2-F1)）
    if iou == 0:
        return 0.0
    f1 = 2 * iou / (1 + iou)

    return f1


def buffer_f1_score_multi_line(
    lines1_xy: List[LineString],
    lines2_xy: List[LineString],
    threshold_meters: float,
) -> float:
    """计算多个折线之间的缓冲区F1分数"""

    multi_line1_xy = MultiLineString(lines1_xy)
    multi_line2_xy = MultiLineString(lines2_xy)

    buffer1 = multi_line1_xy.buffer(threshold_meters)
    buffer2 = multi_line2_xy.buffer(threshold_meters)

    intersection = buffer1.intersection(buffer2)
    union = buffer1.union(buffer2)

    if union.area == 0:
        return 0.0

    iou = intersection.area / union.area

    if iou == 0:
        return 0.0
    f1 = 2 * iou / (1 + iou)

    return f1


def endpoint_distance(
    line1_xy: LineString,
    line2_xy: LineString,
    max_distance: float,
    reduction: Literal["sum", "mean"] = "mean",
) -> float:
    """
    计算两个折线起终点之间的距离（米）

    Args:
        line1_xy: 预测的折线
        line2_xy: 真实的折线
        reduction: 距离计算方式，"sum"表示求和，"mean"表示求平均
        max_distance: 点的距离上限

    Returns:
        距离
    """
    start1 = Point(line1_xy.coords[0])
    end1 = Point(line1_xy.coords[-1])
    start2 = Point(line2_xy.coords[0])
    end2 = Point(line2_xy.coords[-1])

    start_distance = min(start1.distance(start2), max_distance)
    end_distance = min(end1.distance(end2), max_distance)

    if reduction == "sum":
        return start_distance + end_distance
    elif reduction == "mean":
        return (start_distance + end_distance) / 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def endpoint_distance_multi_line(
    lines1_xy: List[LineString],
    lines2_xy: List[LineString],
    max_distance: float,
    reduction: Literal["sum", "mean"] = "mean",
) -> float:
    """
    计算多个折线起终点之间的距离（米），未匹配的点距离为max_distance，过远的点距离为max_distance

    Args:
        lines1_xy: 预测的折线列表
        lines2_xy: 真实的折线列表
        reduction: 距离计算方式，"sum"表示求和，"mean"表示求平均
        max_distance: 点的距离上限

    Returns:
        距离
    """
    matches, unmatched_pred, unmatched_gt = match_linestrings(lines1_xy, lines2_xy)

    total_distance = 0
    for pred_index, gt_index in matches:
        total_distance += endpoint_distance(
            lines1_xy[pred_index], lines2_xy[gt_index], max_distance, "sum"
        )

    for pred_index in unmatched_pred:
        total_distance += max_distance * 2

    for gt_index in unmatched_gt:
        total_distance += max_distance * 2

    if reduction == "sum":
        return total_distance
    elif reduction == "mean":
        return total_distance / max(len(lines1_xy), len(lines2_xy)) / 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def match_junctions_by_distance(pred_junctions, gt_junctions, max_distance):
    """
    基于最近距离匹配Junction点
    返回匹配对和未匹配的点
    """
    if not pred_junctions or not gt_junctions:
        return [], pred_junctions, gt_junctions

    # 计算距离矩阵
    distances = []
    for pred_pt in pred_junctions:
        row = []
        for gt_pt in gt_junctions:
            row.append(euclidean_distance(pred_pt, gt_pt))
        distances.append(row)

    distances = np.array(distances)

    # 贪心匹配：每次找最近的点对
    matches = []
    unmatched_pred = list(range(len(pred_junctions)))
    unmatched_gt = list(range(len(gt_junctions)))

    while unmatched_pred and unmatched_gt:
        # 找到当前最小距离
        min_dist = float("inf")
        best_pred_idx = None
        best_gt_idx = None

        for pred_idx in unmatched_pred:
            for gt_idx in unmatched_gt:
                if distances[pred_idx][gt_idx] < min_dist:
                    min_dist = distances[pred_idx][gt_idx]
                    best_pred_idx = pred_idx
                    best_gt_idx = gt_idx

        # 如果最小距离在阈值内，则匹配
        if (
            min_dist <= max_distance
            and best_pred_idx is not None
            and best_gt_idx is not None
        ):
            matches.append((best_pred_idx, best_gt_idx, min_dist))
            unmatched_pred.remove(best_pred_idx)
            unmatched_gt.remove(best_gt_idx)
        else:
            break

    # 返回未匹配的点
    unmatched_pred_points = [pred_junctions[i] for i in unmatched_pred]
    unmatched_gt_points = [gt_junctions[i] for i in unmatched_gt]

    return matches, unmatched_pred_points, unmatched_gt_points


def frechet_distance(line1_xy: LineString, line2_xy: LineString) -> float:
    """计算两个折线之间的Frechet距离"""
    line1_xy_np = _linestring_to_numpy(line1_xy)
    line2_xy_np = _linestring_to_numpy(line2_xy)
    return similaritymeasures.frechet_dist(line1_xy_np, line2_xy_np)


def match_linestrings(
    line_pred: List[LineString],
    line_groundtruth: List[LineString],
):
    """
    将预测的折线与groundtruth的折线进行一对一匹配，采用frechet_distance作为距离度量，
    使用最优分配算法进行二分图匹配以保证总距离最小。

    Args:
        line_pred: 预测的折线列表
        line_groundtruth: 真实的折线列表

    Returns:
        tuple: 包含三个元素的元组:
            - matches: 匹配对列表，每个元素是(pred_index, gt_index)的元组
            - unmatched_pred: 未匹配的预测折线索引列表
            - unmatched_gt: 未匹配的真实折线索引列表
    """

    # 如果其中一方为空，直接返回空匹配
    if len(line_pred) == 0 or len(line_groundtruth) == 0:
        return [], list(range(len(line_pred))), list(range(len(line_groundtruth)))

    # 计算距离矩阵，行是预测，列是真实
    distance_matrix = np.zeros((len(line_pred), len(line_groundtruth)))
    for i, pred_line in enumerate(line_pred):
        for j, gt_line in enumerate(line_groundtruth):
            distance_matrix[i, j] = frechet_distance(pred_line, gt_line)

    # 使用匈牙利算法求解最优分配
    pred_indices, gt_indices = linear_sum_assignment(distance_matrix)

    # 构建匹配结果
    matches = list(zip(pred_indices, gt_indices))

    # 找出未匹配的预测和真实折线
    matched_pred = set(pred_indices)
    matched_gt = set(gt_indices)
    unmatched_pred = [i for i in range(len(line_pred)) if i not in matched_pred]
    unmatched_gt = [j for j in range(len(line_groundtruth)) if j not in matched_gt]

    return matches, unmatched_pred, unmatched_gt


def _linestring_to_numpy(line: LineString) -> np.ndarray:
    return np.array(line.coords)


if __name__ == "__main__":
    line1_xy = LineString([(0, 0), (1, 0), (1, 1)])
    line2_xy = LineString([(0, 0), (0, 1), (1, 1)])
    line3_xy = LineString([(0, 0), (1, 1)])
    print(buffer_f1_score(line1_xy, line2_xy, 0.1))
    print(buffer_f1_score_multi_line([line1_xy], [line2_xy], 0.1))
    print(buffer_f1_score_multi_line([line1_xy], [line3_xy], 0.1))
    print(frechet_distance(line1_xy, line3_xy))
