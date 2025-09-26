import asyncio
import os
from pathlib import Path
from typing import Any, Dict

import dotenv
from PIL import Image
from tqdm.asyncio import tqdm

from roadnetbenchmark import (
    ConcurrentJSONLWriter,
    VLMClient,
    VLMConfig,
    load_jsonl,
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
        "variable": "G",
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
) -> Dict[str, Any]:
    """
    处理单个数据项

    Args:
        img: PIL图像对象
        label: 标签数据
        vlm_client: VLM客户端
        writer: 并发JSONL写入器

    Returns:
        处理结果字典
    """
    lid = label["id"]

    question = f"""**Task: Lane Designations Recognition for Road Network Analysis**

**Image Description:**
This is a personal viewpoint image of a road in China, where vehicles drive on the right side of the road.

**Data:**
```yaml
# The number of lanes
num_lanes: {label["num_lane"]}
```

**Question:**
Analyze the image and determine: What are the lane designations in the direction of the road?

**Lane Direction Types:**
The available lane direction types include:
(1) U-turn
(2) left-turn
(3) straight
(4) right-turn
(5) variable

**YAML Output Requirements:**
- Add ONLY one-line YAML comment explaining your visual analysis and reasoning
- Return `lane_designations` as a list where each item represents the lane direction types for each lane from left to right
  - Make sure the number of lane designations is equal to the number of lanes (which is {label["num_lane"]})
- Each lane can have multiple direction types, so each item in the list should also be a list
- Use the English terms: "U-turn", "left-turn", "straight", "right-turn", "variable"

**Output Format:**
```yaml
# Explanation: [Describe what you see that led to your determination]
lane_designations: [["direction1", "direction2"], ["direction3"], ...]
```

Your YAML output:
"""

    try:
        raw_answer = await vlm_client.chat_with_images_with_yaml_and_retry(
            [img], question
        )
        error = None
        try:
            answer = unwrap_yaml(raw_answer)
            lane_designations = answer["lane_designations"]
            # 转换为ABCDG格式
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


class Task5Dataset:
    def __init__(self, folder: str, img_suffix: str = ".png"):
        self._folder = folder
        self._img_suffix = img_suffix
        self.labels = load_jsonl(Path(folder) / "labels.jsonl")
        # 过滤掉laneinfo为空的数据，过滤掉laneinfo中的非常规数据
        # 常规数据：A(掉头) B(左转) C(直行) D(右转) G(可变车道)
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
                    if ch not in ["A", "B", "C", "D", "G"]:
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


async def main(base_url: str, model: str, concurrency: int, api_key: str):
    """
    主函数，支持并发处理

    Args:
        concurrency: 并发数量
    """
    folder = Path("data/task_4_5")

    # 创建VLM配置
    vlm_config = VLMConfig(
        api_base=base_url,
        api_key=api_key,
        model=model,
    )

    result_path = folder / f"task5_{vlm_config.model.replace("/", "_")}.jsonl"

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
    dataset = Task5Dataset(
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
            return await process_single_item(img, label, vlm_client, writer)

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

    args = parser.parse_args()

    print(f"Starting lane counting with concurrency level: {args.concurrency}")
    asyncio.run(
        main(
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            api_key=args.api_key,
        )
    )
