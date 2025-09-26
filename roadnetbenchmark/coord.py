from dataclasses import dataclass
from typing import Tuple

import pyproj
from shapely import LineString

__all__ = ["lonlat_to_xy", "xy_to_lonlat", "PROJSTR"]

# 采用中国2000坐标系
PROJSTR = "+proj=tmerc +lat_0=0 +lon_0=114 +k=1 +x_0=38500000 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs"


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


def lonlat_to_xy(lon: float, lat: float) -> Tuple[float, float]:
    """
    将经纬度坐标转换为平面坐标
    """

    projector = pyproj.Proj(PROJSTR)
    return projector(lon, lat)


def xy_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    """
    将平面坐标转换为经纬度坐标
    """
    projector = pyproj.Proj(PROJSTR)
    return projector(x, y, inverse=True)


def lonlat_to_xy_linestring(line: LineString) -> LineString:
    """
    将经纬度坐标转换为平面坐标
    """
    return LineString([lonlat_to_xy(lon, lat) for lon, lat in line.coords])


def xy_to_lonlat_linestring(line: LineString) -> LineString:
    """
    将平面坐标转换为经纬度坐标
    """
    return LineString([xy_to_lonlat(x, y) for x, y in line.coords])


class CoordinateConverter:
    """统一的坐标转换器，基于卫星图像的实际边界框进行经纬度和像素坐标转换"""

    def __init__(self, image_size: dict, bbox: BoundingBox):
        """
        初始化坐标转换器

        Args:
            image_size: 图像的尺寸
            bbox: 图像的实际边界框（经纬度，按需使用WGS84或GCJ02）
        """
        self.bbox = bbox
        self.image_width = image_size["width"]
        self.image_height = image_size["height"]

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

    def lon_lat_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        将经纬度坐标转换为像素坐标
        """
        return self.lat_lon_to_pixel(lat, lon)

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

    def pixel_to_lon_lat(self, x: float, y: float) -> Tuple[float, float]:
        """
        将像素坐标转换为经纬度坐标
        """
        lat, lon = self.pixel_to_lat_lon(x, y)
        return lon, lat

    def pixel_to_lon_lat_linestring(self, line: LineString) -> LineString:
        """
        将像素坐标转换为经纬度坐标
        """
        return LineString([self.pixel_to_lon_lat(x, y) for x, y in line.coords])

    def lon_lat_to_pixel_linestring(self, line: LineString) -> LineString:
        """
        将经纬度坐标转换为像素坐标
        """
        return LineString([self.lon_lat_to_pixel(lon, lat) for lon, lat in line.coords])

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


if __name__ == "__main__":
    print(lonlat_to_xy(114.31, 30.57))
    print(xy_to_lonlat(38500000, 0))
    line = LineString([(114.31, 30.57), (114.32, 30.57)])
    print(lonlat_to_xy_linestring(line))
    print(xy_to_lonlat_linestring(lonlat_to_xy_linestring(line)))
