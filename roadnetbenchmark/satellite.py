#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Earth卫星底图下载程序
专门用于下载Google Earth最高精度的卫星影像
"""

import os
import math
import asyncio
import aiohttp
from typing import Tuple, List, Optional
from dataclasses import dataclass
from PIL import Image
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """边界框类"""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def __post_init__(self):
        """验证边界框坐标"""
        if not (-90 <= self.min_lat <= 90 and -90 <= self.max_lat <= 90):
            raise ValueError("纬度必须在-90到90之间")
        if not (-180 <= self.min_lon <= 180 and -180 <= self.max_lon <= 180):
            raise ValueError("经度必须在-180到180之间")
        if self.min_lat >= self.max_lat:
            raise ValueError("最小纬度必须小于最大纬度")
        if self.min_lon >= self.max_lon:
            raise ValueError("最小经度必须小于最大经度")


class CoordinateConverter:
    """统一的坐标转换器，基于卫星图像的实际边界框进行经纬度和像素坐标转换"""

    def __init__(self, image: Image.Image, bbox: BoundingBox):
        """
        初始化坐标转换器

        Args:
            image: PIL图像对象
            bbox: 图像的实际边界框
        """
        self.image = image
        self.bbox = bbox
        self.image_width = image.width
        self.image_height = image.height

    def lat_lon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        将经纬度坐标转换为像素坐标

        Args:
            lat: 纬度
            lon: 经度

        Returns:
            (x, y) 像素坐标
        """
        # 计算经纬度在边界框中的相对位置
        lat_ratio = (lat - self.bbox.min_lat) / (self.bbox.max_lat - self.bbox.min_lat)
        lon_ratio = (lon - self.bbox.min_lon) / (self.bbox.max_lon - self.bbox.min_lon)

        # 转换为像素坐标（注意Y轴方向：图像坐标系Y轴向下，地理坐标系Y轴向上）
        x = int(lon_ratio * self.image_width)
        y = int((1 - lat_ratio) * self.image_height)  # 翻转Y轴

        return x, y

    def pixel_to_lat_lon(self, x: float, y: float) -> Tuple[float, float]:
        """
        将像素坐标转换为经纬度坐标

        Args:
            x: 像素X坐标
            y: 像素Y坐标

        Returns:
            (lat, lon) 经纬度坐标
        """
        # 计算像素在图像中的相对位置
        x_ratio = x / self.image_width
        y_ratio = y / self.image_height

        # 转换为经纬度（注意Y轴方向）
        lon = self.bbox.min_lon + x_ratio * (self.bbox.max_lon - self.bbox.min_lon)
        lat = self.bbox.max_lat - y_ratio * (
            self.bbox.max_lat - self.bbox.min_lat
        )  # 翻转Y轴

        return lat, lon

    def get_image_bounds(self) -> Tuple[int, int, int, int]:
        """
        获取图像的像素边界

        Returns:
            (min_x, min_y, max_x, max_y) 像素边界
        """
        return (0, 0, self.image_width, self.image_height)

    def is_pixel_in_bounds(self, x: int, y: int) -> bool:
        """
        检查像素坐标是否在图像范围内

        Args:
            x: 像素X坐标
            y: 像素Y坐标

        Returns:
            是否在范围内
        """
        return 0 <= x < self.image_width and 0 <= y < self.image_height

    def is_lat_lon_in_bounds(self, lat: float, lon: float) -> bool:
        """
        检查经纬度坐标是否在边界框范围内

        Args:
            lat: 纬度
            lon: 经度

        Returns:
            是否在范围内
        """
        return (
            self.bbox.min_lat <= lat <= self.bbox.max_lat
            and self.bbox.min_lon <= lon <= self.bbox.max_lon
        )
