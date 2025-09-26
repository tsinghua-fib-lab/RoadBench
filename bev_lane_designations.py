import asyncio
import os
from pathlib import Path
from typing import Any, Dict

import dotenv
from shapely import LineString
import yaml
from PIL import Image
from shapely.wkt import loads
from tqdm.asyncio import tqdm

from roadnetbenchmark import (
    ConcurrentJSONLWriter,
    VLMClient,
    VLMConfig,
    draw_linestring_on_image,
    load_jsonl,
)
from roadnetbenchmark.image import (
    ReferenceLineMode,
    DirectionAnnotationMode,
    generate_reference_line_description,
)
from roadnetbenchmark.concurrent_jsonl_writer import dump_jsonl
from roadnetbenchmark.vlm_client import unwrap_yaml

dotenv.load_dotenv()


def convert_lane_designations_to_abcd(lane_designations):
    """
    将车道导向种类转换为ABCD格式

    Args:
        lane_designations: 车道导向种类列表，每个元素是一个车道的导向种类列表

    Returns:
        转换后的ABCD格式列表，每个元素是一个车道的ABCD字符串
    """
    direction_map = {
        "u-turn": "A",
        "left-turn": "B",
        "straight": "C",
        "right-turn": "D",
    }

    result = []
    for lane_directions in lane_designations:
        abcd_chars = []
        for direction in lane_directions:
            if isinstance(direction, str) and direction.lower() in direction_map:
                abcd_chars.append(direction_map[direction.lower()])
        result.append("".join(sorted(abcd_chars)))  # 排序保证一致性

    return result


async def process_single_item(
    img: Image.Image,
    label: Dict[str, Any],
    vlm_client: VLMClient,
    writer: ConcurrentJSONLWriter,
    reference_line_mode: ReferenceLineMode = ReferenceLineMode.BOTH,
    direction_annotation_mode: DirectionAnnotationMode = DirectionAnnotationMode.ARROWS,
) -> Dict[str, Any]:
    """
    处理单个数据项

    Args:
        img: PIL图像对象
        label: 标签数据
        vlm_client: VLM客户端
        writer: 并发JSONL写入器
        reference_line_mode: 参考线呈现模式
        direction_annotation_mode: 方向标注模式

    Returns:
        处理结果字典
    """
    lid = label["id"]
    pixel_line = loads(label["pixel_line"])
    assert isinstance(pixel_line, LineString)

    # 根据参考线模式决定是否在图片中绘制参考线
    if reference_line_mode in [ReferenceLineMode.IMAGE_ONLY, ReferenceLineMode.BOTH]:
        image = draw_linestring_on_image(
            img,
            pixel_line,
            None,
            color=(255, 0, 0),
            width=5,
            road_label="",
            show_start_label=False,
            show_end_label=False,
            direction_annotation_mode=direction_annotation_mode,
        )
    else:
        image = img.copy()

    def compress_xy(xy):
        for k, v in xy.items():
            if isinstance(v, float):
                xy[k] = round(v, 2)
        return xy

    xys = {
        "coordinates": [
            compress_xy({"x": xy[0], "y": xy[1]}) for xy in pixel_line.coords
        ]
    }

    # 根据参考线模式生成不同的问题描述
    reference_line_prompt = ""
    if reference_line_mode in [ReferenceLineMode.PROMPT_ONLY, ReferenceLineMode.BOTH]:
        # 包含参考线信息
        reference_line_prompt = generate_reference_line_description(pixel_line, direction_annotation_mode)

    question = f"""**Task: Lane Designations Recognition for Road Network Analysis**

**Image Description:**
This is a satellite image of a road network in China, where vehicles drive on the right side of the road. {reference_line_prompt}

**Data:**
```yaml
# The number of lanes
num_lanes: {label["num_lane"]}
# The image size
image:
  width: {img.width}
  height: {img.height}
{"" if reference_line_mode not in [ReferenceLineMode.PROMPT_ONLY, ReferenceLineMode.BOTH] else f"# The pixel coordinates of the reference centerline\n{yaml.dump(xys).strip()}"}
```

**Question:**
Analyze the marked road segment and determine: What are the lane designations in the direction of the {'arrow' if direction_annotation_mode.value==DirectionAnnotationMode.ARROWS.value else 'line'}?

**Lane Direction Types:**
The available lane direction types include:
(1) U-turn
(2) left-turn
(3) straight
(4) right-turn

**YAML Output Requirements:**
- Add ONLY one-line YAML comment explaining your visual analysis and reasoning
- Return `lane_designations` as a list where each item represents the lane direction types for each lane from left to right
  - Make sure the number of lane designations is equal to the number of lanes (which is {label["num_lane"]})
- Each lane can have multiple direction types, so each item in the list should also be a list
- Use the English terms: "U-turn", "left-turn", "straight", "right-turn"

**Output Format:**
```yaml
# Explanation: [Describe what you see that led to your determination]
lane_designations: [["direction1", "direction2"], ["direction3"], ...]
```

Your YAML output:
"""

    try:
        raw_answer = await vlm_client.chat_with_images_with_yaml_and_retry(
            [image], question
        )
        error = None
        try:
            answer = unwrap_yaml(raw_answer)
            lane_designations = answer["lane_designations"]
            # 转换为ABCD格式
            abcd_designations = convert_lane_designations_to_abcd(lane_designations)
        except Exception as e:
            error = str(e)
            abcd_designations = []

        result = {
            "id": lid,
            "answer": raw_answer,
            "llm_value": abcd_designations,
            "ground_truth": label["laneinfo"],
            "error": error,
        }

        # 异步写入结果
        await writer.write_async(result)

        return result

    except Exception as e:
        error_result = {
            "id": lid,
            "answer": f"Error: {str(e)}",
            "llm_value": [],
            "ground_truth": label["laneinfo"],
            "error": str(e),
        }
        await writer.write_async(error_result)
        return error_result


class Task2Dataset:
    def __init__(self, folder: str, img_suffix: str = ".png"):
        self._folder = folder
        self._img_suffix = img_suffix
        self.labels = load_jsonl(Path(folder) / "labels.jsonl")
        # 过滤掉laneinfo为空的数据，过滤掉laneinfo中的非常规数据
        # 常规数据：A(掉头) B(左转) C(直行) D(右转)
        ok_labels = []
        for label in self.labels:
            laneinfo = label["laneinfo"]
            if len(laneinfo) == 0 or len(laneinfo) != label["num_lane"]:
                continue
            bad = False
            for one_laneinfo in laneinfo:
                # 检查每个字母
                bad = False
                for ch in one_laneinfo:
                    if ch not in ["A", "B", "C", "D"]:
                        bad = True
                        break
                if bad:
                    break
            if not bad:
                ok_labels.append(label)
        self.labels = ok_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image_path = Path(self._folder) / f"{label['id']}{self._img_suffix}"
        image = Image.open(image_path)
        return image, label


async def main(
    base_url: str,
    model: str,
    concurrency: int,
    api_key: str,
    reference_line_mode: ReferenceLineMode = ReferenceLineMode.BOTH,
    direction_annotation_mode: DirectionAnnotationMode = DirectionAnnotationMode.ARROWS,
):
    """
    主函数，支持并发处理

    Args:
        concurrency: 并发数量
    """
    folder = Path("data/task_1_2")

    # 创建VLM配置
    vlm_config = VLMConfig(
        api_base=base_url,
        api_key=api_key,
        model=model,
    )

    # 生成包含参考线设置的文件名后缀
    ref_mode_suffix = reference_line_mode.value
    dir_mode_suffix = direction_annotation_mode.value
    settings_suffix = f"{ref_mode_suffix}_{dir_mode_suffix}"

    result_path = (
        folder / f"task2_{settings_suffix}_{vlm_config.model.replace('/', '_')}.jsonl"
    )

    # 加载现有结果，获取已处理的ID
    existing_results = load_jsonl(result_path)
    # 移除现有结果中llm_value为空列表或error!=None的结果
    old_len = len(existing_results)
    existing_results = [
        result
        for result in existing_results
        if result["llm_value"] != [] and result["error"] is None
    ]
    new_len = len(existing_results)
    print(f"Removed {old_len - new_len} results")
    # 重新覆盖写入文件
    dump_jsonl(result_path, existing_results)
    print(f"Overwritten {new_len} results")
    existing_ids = set([result["id"] for result in existing_results])

    # 创建并发JSONL写入器
    writer = ConcurrentJSONLWriter(result_path)

    # 创建数据集
    dataset = Task2Dataset(
        folder=str(folder / "dataset"),
        img_suffix=".png",
    )

    # 创建VLM客户端
    vlm_client = VLMClient(vlm_config)

    # 准备待处理的数据
    pending_items = []
    for img, label in dataset:
        lid = label["id"]
        if lid not in existing_ids:
            pending_items.append((img, label))

    print(
        f"Found {len(pending_items)} items to process (skipping {len(existing_ids)} already completed)"
    )
    print(f"Using concurrency level: {concurrency}")

    # 创建信号量限制并发数
    semaphore = asyncio.Semaphore(concurrency)

    async def process_with_semaphore(img, label):
        """带信号量限制的处理函数"""
        async with semaphore:
            return await process_single_item(
                img,
                label,
                vlm_client,
                writer,
                reference_line_mode,
                direction_annotation_mode,
            )

    # 并发处理所有待处理项目
    tasks = [process_with_semaphore(img, label) for img, label in pending_items]

    # 使用tqdm显示进度
    results = []
    for coro in tqdm.as_completed(tasks, desc="Processing items"):
        result = await coro
        results.append(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process lane designations with concurrent VLM calls"
    )
    parser.add_argument(
        "-b",
        "--base_url",
        type=str,
        default="https://cloud.infini-ai.com/maas/v1",
        help="Base URL of the VLM API",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen2.5-vl-72b-instruct",
        help="Model name",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "-k",
        "--api_key",
        type=str,
        default=os.getenv("API_KEY"),
        help="API key",
    )
    parser.add_argument(
        "--reference_line_mode",
        type=str,
        choices=["prompt_only", "image_only", "both"],
        default="both",
        help="Reference line presentation mode: prompt_only, image_only, or both",
    )
    parser.add_argument(
        "--direction_annotation_mode",
        type=str,
        choices=["arrows", "colors"],
        default="arrows",
        help="Direction annotation mode: arrows or colors (green start, blue end)",
    )

    args = parser.parse_args()

    # 转换字符串参数为枚举类型
    reference_line_mode = ReferenceLineMode(args.reference_line_mode)
    direction_annotation_mode = DirectionAnnotationMode(args.direction_annotation_mode)

    print(f"Starting lane designations recognition with concurrency level: {args.concurrency}")
    print(f"Reference line mode: {reference_line_mode.value}")
    print(f"Direction annotation mode: {direction_annotation_mode.value}")

    asyncio.run(
        main(
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            api_key=args.api_key,
            reference_line_mode=reference_line_mode,
            direction_annotation_mode=direction_annotation_mode,
        )
    )
