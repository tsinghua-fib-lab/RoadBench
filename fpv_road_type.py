import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

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


async def process_single_item(
    img: Image.Image,
    label: Dict[str, Any],
    vlm_client: VLMClient,
    writer: ConcurrentJSONLWriter,
) -> Dict[str, Any]:
    """
    处理单个数据项

    Args:
        imgs: PIL图像对象列表
        label: 标签数据
        vlm_client: VLM客户端
        writer: 并发JSONL写入器

    Returns:
        处理结果字典
    """
    lid = label["id"]

    question = """**Task: Road Type Classification**

**Image Description:**
This is a personal viewpoint image of a road in China, where vehicles drive on the right side of the road.

**Question:**
Analyze the image and determine: Is the current vehicle traveling on a main road or a service road?

**Road Type Definitions:**
- **Main road**: Primary traffic arteries, highways, major streets with higher traffic capacity and speed limits
- **Service road**: Secondary roads, access roads, frontage roads, or auxiliary roads that typically serve adjacent properties or provide access to main roads

**YAML Output Requirements:**
- Add ONLY one-line YAML comment explaining your visual analysis and reasoning
- Extract `road_type` as either "main" or "service"

**Output Format:**
```yaml
# Explanation: [Describe what you see that led to your determination]
road_type: <"main" or "service">
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
            road_type = answer["road_type"]
            assert road_type in ["main", "service"]
        except Exception as e:
            error = str(e)
            road_type = ""

        result = {
            "id": lid,
            "answer": raw_answer,
            "llm_value": road_type,
            "ground_truth": label["road_type"],
            "error": error,
        }

        # 异步写入结果
        await writer.write_async(result)

        return result

    except Exception as e:
        error_result = {
            "id": lid,
            "answer": f"Error: {str(e)}",
            "llm_value": "",
            "ground_truth": label["road_type"],
            "error": str(e),
        }
        await writer.write_async(error_result)
        return error_result


class Task6Dataset:
    def __init__(self, folder: str, img_suffix: str = ".png"):
        self._folder = folder
        self._img_suffix = img_suffix
        self.labels = load_jsonl(Path(folder) / "labels.jsonl")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        img_name = label["img"]
        image_path = Path(self._folder) / img_name
        image = Image.open(image_path)
        return image, label


async def main(base_url: str, model: str, concurrency: int, api_key: str):
    """
    主函数，支持并发处理

    Args:
        concurrency: 并发数量，默认为5
    """
    folder = Path("data/task_6")

    # 创建VLM配置
    vlm_config = VLMConfig(
        api_base=base_url,
        api_key=api_key,
        model=model,
    )

    result_path = folder / f"task6_{vlm_config.model.replace("/", "_")}.jsonl"

    # 加载现有结果，获取已处理的ID
    existing_results = load_jsonl(result_path)
    # 移除现有结果中llm_value为0或error!=None的结果
    old_len = len(existing_results)
    existing_results = [
        result
        for result in existing_results
        if result["llm_value"] != 0 and result["error"] is None
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
    dataset = Task6Dataset(
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
        description="Process lane counting with concurrent VLM calls"
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
