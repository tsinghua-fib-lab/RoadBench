import asyncio
import os
from pathlib import Path
from typing import Any, Dict

import dotenv
import yaml
from PIL import Image
import shapely.wkt as wkt
from shapely import LineString
from tqdm.asyncio import tqdm

from roadnetbenchmark import (
    ConcurrentJSONLWriter,
    VLMClient,
    VLMConfig,
    draw_linestring_on_image,
    load_jsonl,
)
from roadnetbenchmark.concurrent_jsonl_writer import dump_jsonl
from roadnetbenchmark.vlm_client import unwrap_yaml

dotenv.load_dotenv()


async def process_single_item(
    img: Image.Image,
    label: Dict[str, Any],
    vlm_client: VLMClient,
    writer: ConcurrentJSONLWriter,
) -> Dict[str, Any]:
    """
    处理单个道路网络修正数据项

    Args:
        img: PIL图像对象
        label: 标签数据，包含参考线和真值道路网络
        vlm_client: VLM客户端
        writer: 并发JSONL写入器

    Returns:
        处理结果字典，包含识别的路口和线段
    """
    lid = label["id"]
    pixel_line = wkt.loads(label["pixel_line"])
    assert isinstance(pixel_line, LineString)
    image = draw_linestring_on_image(
        img,
        pixel_line,
        None,
        color=(255, 0, 0),
        width=5,
        road_label="",
        show_start_label=False,
        show_end_label=False,
    )

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

    question = f"""**Task: Road Network Modification and Junction Identification**

**Image Description:**
This is a satellite image of a road network in China, where vehicles drive on the right side of the road. A reference line has been marked as a red directed polyline to indicate the path and direction of travel along a road.

**Data:**
```yaml
# The image size
image:
  width: {img.width}
  height: {img.height}
# The pixel coordinates of the reference centerline
{yaml.dump(xys).strip()}
```

**Task Description:**
The given reference line may have missed important junctions such as intersections, highway on/off ramps, U-turn locations, dedicated right-turn lanes, etc., and it may not always be accurately aligned with the true centerline of the road. Your task is to analyze the image and identify:

1. **Junctions**: Important road intersections, merging points, or decision points that should be marked. Junctions are represented by WKT POINT geometries.
2. **Line Segments (with Directionality)**: Road segments between junctions that represent the centerlines of the roads, with attention to the direction of travel (from start to end, following the reference line's direction). Line segments are represented by WKT LINESTRING geometries, and their order and orientation should reflect the actual direction of traffic flow along the road.

**Analysis Requirements:**
- Carefully examine the directed reference line and the actual road network in the image
- Identify any missing junctions where the directed reference line should be split (intersections, ramps, etc.)
- Return corrected directed line segments that represent the centerlines between identified junctions
- Ensure junctions are placed at the center points of intersections/decision points
- Ensure line segments follow the actual road centerlines
- Do not return junctions or line segments that are not related to the directed reference line

**YAML Output Requirements:**
- Add ONLY one-line YAML comment explaining your analysis and corrections
- Return `junctions` as a list of WKT POINT geometries representing junction center points
- Return `lines` as a list of directed WKT LINESTRING geometries representing road centerlines between junctions
- Use pixel coordinates in the format: "POINT (x y)" and "LINESTRING (x1 y1, x2 y2, ...)"

**Output Format:**
```yaml
# Explanation: [Describe the junctions and line segments you identified and any corrections made to the directed reference line]
junctions: ["POINT (x1 y1)", "POINT (x2 y2)", ...]
lines: ["LINESTRING (x1 y1, x2 y2)", "LINESTRING (x2 y2, x3 y3)", ...]
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
            junctions = answer["junctions"]
            # 检查是否是合法的WKT POINT
            for junction in junctions:
                wkt.loads(junction)
            lines = answer["lines"]
            for line in lines:
                wkt.loads(line)
            llm_result = {"junctions": junctions, "lines": lines}
        except Exception as e:
            error = str(e)
            llm_result = {"junctions": [], "lines": []}

        result = {
            "id": lid,
            "answer": raw_answer,
            "llm_value": llm_result,
            "ground_truth": label["ground_truth"],
            "error": error,
        }

        # 异步写入结果
        await writer.write_async(result)

        return result

    except Exception as e:
        error_result = {
            "id": lid,
            "answer": f"Error: {str(e)}",
            "llm_value": {"junctions": [], "lines": []},
            "ground_truth": label["ground_truth"],
            "error": str(e),
        }
        await writer.write_async(error_result)
        return error_result


class Task3Dataset:
    def __init__(self, folder: str, img_suffix: str = ".png"):
        self._folder = folder
        self._img_suffix = img_suffix
        self.labels = load_jsonl(Path(folder) / "labels.jsonl")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image_path = Path(self._folder) / f"{label['id']}{self._img_suffix}"
        image = Image.open(image_path)
        return image, label


async def main(base_url: str, model: str, concurrency: int, api_key: str):
    """
    道路网络修正任务主函数，支持并发处理

    Args:
        base_url: VLM API的基础URL
        model: 使用的模型名称
        concurrency: 并发数量
    """
    folder = Path("data/task_3")

    # 创建VLM配置
    vlm_config = VLMConfig(
        api_base=base_url,
        api_key=api_key,
        model=model,
    )

    result_path = folder / f"task3_{vlm_config.model.replace("/", "_")}.jsonl"

    # 加载现有结果，获取已处理的ID
    existing_results = load_jsonl(result_path)
    # 移除现有结果中llm_value为空或error!=None的结果
    old_len = len(existing_results)
    existing_results = [
        result
        for result in existing_results
        if (
            result.get("llm_value", {}).get("junctions")
            or result.get("llm_value", {}).get("lines")
        )
        and result["error"] is None
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
    dataset = Task3Dataset(
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
        description="Process road network modification with concurrent VLM calls"
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

    print(
        f"Starting road network modification with concurrency level: {args.concurrency}"
    )
    asyncio.run(
        main(
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            api_key=args.api_key,
        )
    )
