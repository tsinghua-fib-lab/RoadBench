#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM（Vision Language Model）客户端
支持OpenAI及其兼容接口的多模态大模型调用
"""

import asyncio
import base64
import io
import logging
import math
import os
import re
from typing import List, Optional

import httpx
import json_repair
import yaml
import yaml.scanner
from openai import AsyncOpenAI, AsyncAzureOpenAI
from PIL import Image
from pydantic import BaseModel, Field

# 禁用 OpenAI 库的 INFO 级别日志
logging.basicConfig(level=logging.WARN)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

__all__ = ["VLMConfig", "VLMClient", "unwrap_json"]


class VLMConfig(BaseModel):
    """VLM配置类"""

    # API配置
    api_base: str
    api_key: Optional[str]
    model: str
    proxy: Optional[str] = None

    # 请求配置
    timeout: int = Field(default=60, description="超时时间（秒）")

    def __init__(self, **data):
        super().__init__(**data)
        # 如果API密钥未设置，从环境变量获取
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        # 检测环境变量，自动设置proxy
        if self.proxy is None:
            if os.getenv("https_proxy") is not None:
                self.proxy = os.getenv("https_proxy")
            elif os.getenv("HTTPS_PROXY") is not None:
                self.proxy = os.getenv("HTTPS_PROXY")


class VLMClient:
    """VLM客户端类"""

    def __init__(self, config: Optional[VLMConfig] = None):
        """
        初始化VLM客户端

        Args:
            config: VLM配置，如果为None则使用默认配置
        """
        self.config = config or VLMConfig()

        # 检查API密钥
        if not self.config.api_key:
            raise ValueError("API密钥未设置，请设置OPENAI_API_KEY环境变量")

        # 根据配置创建相应的OpenAI客户端
        if self._is_azure_url(self.config.api_base):
            # 获取Azure配置
            azure_endpoint = self._get_azure_endpoint(self.config.api_base)
            api_version = self._get_api_version_from_model(self.config.model)
            
            # 创建Azure OpenAI客户端
            self.client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                timeout=self.config.timeout,
                http_client=httpx.AsyncClient(
                    proxy=self.config.proxy,
                ),
            )
        else:
            # 创建标准OpenAI客户端
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
                http_client=httpx.AsyncClient(
                    proxy=self.config.proxy,
                ),
            )

    def _is_azure_url(self, url: str) -> bool:
        """检测是否为Azure OpenAI URL"""
        return "openai.azure.com" in url

    def _get_azure_endpoint(self, api_base: str) -> str:
        """从api_base提取Azure endpoint"""
        import urllib.parse
        parsed_url = urllib.parse.urlparse(api_base)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    def _get_api_version_from_model(self, model: str) -> str:
        """从model名称映射API版本"""
        # 定义模型到API版本的映射
        model_to_version = {
            "gpt-5-nano": "2025-01-01-preview",
            "gpt-5-mini": "2025-01-01-preview",
            "gpt-4o": "2024-02-15-preview",
            "gpt-4": "2024-02-15-preview",
            # 可以根据需要添加更多映射
        }
        return model_to_version.get(model, "2024-02-15-preview")  # 默认版本

    def _get_deployment_from_model(self, model: str) -> str:
        """从model名称映射Azure部署名称"""
        # 对于简单情况，部署名称就是模型名称
        # 可以根据实际情况定制映射逻辑
        return model

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        将图片编码为base64字符串

        Args:
            image: PIL图像对象

        Returns:
            base64编码的图片字符串
        """
        # 转换为base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG", optimize=True)
        img_data = img_buffer.getvalue()
        base64_string = base64.b64encode(img_data).decode("utf-8")

        return base64_string

    async def chat_with_images(
        self,
        images: List[Image.Image],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        与图片进行对话

        Args:
            image: PIL图像对象
            question: 问题文本
            system_prompt: 系统提示词，可选

        Returns:
            API响应字典
        """
        # 编码图片
        base64_images = [self._encode_image_to_base64(image) for image in images]

        # 构建消息列表
        messages = []

        # 添加系统提示词
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 添加用户消息（包含图片和问题）
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                },
            }
            for base64_image in base64_images
        ] + [
            {"type": "text", "text": question},
        ]

        messages.append({"role": "user", "content": user_content})

        # 使用OpenAI客户端发送请求
        # 对于Azure OpenAI，使用deployment名称作为model
        model_name = self.config.model
        if self._is_azure_url(self.config.api_base):
            model_name = self._get_deployment_from_model(self.config.model)
            
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        if (
            response.choices[0].message.content is None
            or response.choices[0].message.content == ""
        ):
            raise ValueError("API响应为空")
        return response.choices[0].message.content

    async def chat_with_images_with_yaml_and_retry(
        self,
        images: List[Image.Image],
        question: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        与图片进行对话，并解析YAML
        """
        for attempt in range(max_retries + 1):
            try:
                answer = await self.chat_with_images(images, question, system_prompt)
                if answer == "":
                    raise ValueError("API响应为空")
                unwrap_yaml(answer)
                return answer
            except (ValueError, yaml.YAMLError, yaml.scanner.ScannerError) as e:
                logging.info(f"YAML parsing failed on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.info(
                        f"All {max_retries + 1} attempts failed. Using fallback."
                    )
                    return answer
        return ""


def unwrap_json(text: str) -> dict:
    """
    从文本中提取JSON对象
    """
    # 使用正则表达式提取JSON对象
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return json_repair.loads(match.group(1))  # type: ignore
    raise ValueError(f"未找到JSON对象: {text}")


def unwrap_yaml(text: str) -> dict:
    """
    从文本中提取YAML对象
    """
    yaml_pattern = r"```yaml\s*(.*?)\s*```"
    match = re.search(yaml_pattern, text, re.DOTALL)
    if match:
        return yaml.safe_load(match.group(1))  # type: ignore
    # 如果没有markdown标签，直接将所有文本视为YAML
    return yaml.safe_load(text)
    # raise ValueError(f"未找到YAML对象: {text}")
