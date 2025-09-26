import asyncio
import json
from pathlib import Path
from typing import Any, Dict


class ConcurrentJSONLWriter:
    """
    并发安全的JSONL写入器
    使用asyncio.Lock确保写入操作的线程安全
    """

    def __init__(self, file_path: Path):
        """
        初始化并发JSONL写入器

        Args:
            file_path: JSONL文件路径
        """
        self.file_path = Path(file_path)

        # 确保目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建asyncio锁
        self.lock = asyncio.Lock()

    async def write_async(self, data: Dict[str, Any]) -> None:
        """
        异步写入数据到JSONL文件

        Args:
            data: 要写入的数据字典
        """
        async with self.lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()  # 确保数据写入磁盘


def load_jsonl(path: Path) -> list:
    """加载JSONL文件"""
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        results = []
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error: Failed to parse line: {line}")
                    raise
                results.append(data)
        return results


def dump_jsonl(path: Path, data: list) -> None:
    """一次性写入JSONL文件"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, data: list) -> None:
    """追加写入JSONL文件"""
    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")