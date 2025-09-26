from typing import Optional, Tuple, List, Union
from PIL import Image, ImageDraw, ImageFont
from shapely import LineString, Point
import math
from enum import Enum

from .satellite import CoordinateConverter
from .coord import BoundingBox

class ReferenceLineMode(Enum):
    """参考线呈现模式"""
    PROMPT_ONLY = "prompt_only"  # 仅在Prompt中包含参考线，图片中不绘制
    IMAGE_ONLY = "image_only"  # 仅在图片中绘制参考线，Prompt中不包含
    BOTH = "both"  # 都包含


class DirectionAnnotationMode(Enum):
    """方向标注模式"""
    ARROWS = "arrows"  # 箭头标注方向
    COLORS = "colors"  # 起点终点颜色标注（绿色起点，蓝色终点）


__all__ = ["draw_linestring_on_image", "crop_image", "ReferenceLineMode", "DirectionAnnotationMode", "generate_reference_line_description"]


def _draw_arrow(draw, start, end, color, width):
    """
    在两点之间绘制带箭头的线段
    
    Args:
        draw: PIL ImageDraw对象
        start: 起点坐标 (x, y)
        end: 终点坐标 (x, y)
        color: 箭头颜色
        width: 箭头线宽
    """
    # 绘制主干线
    draw.line([start, end], fill=color, width=width)
    
    # 计算箭头方向
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx*dx + dy*dy)
    
    if length == 0:
        return
    
    # 单位向量
    ux = dx / length
    uy = dy / length
    
    # 箭头头部大小
    arrow_length = width * 3
    arrow_angle = math.pi / 6  # 30度
    
    # 箭头头部的两个点
    # 第一个点
    angle1 = math.atan2(uy, ux) + arrow_angle
    x1 = end[0] - arrow_length * math.cos(angle1)
    y1 = end[1] - arrow_length * math.sin(angle1)
    
    # 第二个点
    angle2 = math.atan2(uy, ux) - arrow_angle
    x2 = end[0] - arrow_length * math.cos(angle2)
    y2 = end[1] - arrow_length * math.sin(angle2)
    
    # 绘制箭头头部
    draw.line([end, (x1, y1)], fill=color, width=width)
    draw.line([end, (x2, y2)], fill=color, width=width)


def draw_linestring_on_image(
    image: Image.Image,
    linestring: LineString,
    converter: Optional[CoordinateConverter],
    color: tuple,
    width: int,
    road_label: str,
    show_start_label: bool,
    show_end_label: bool,
    copy: bool = True,
    direction_annotation_mode: DirectionAnnotationMode = DirectionAnnotationMode.ARROWS,
):
    """
    在卫星影像上绘制LineString，根据方向标注模式显示方向信息

    Args:
        image: PIL图像对象
        linestring: Shapely LineString对象
        converter: 坐标转换器
        color: 线条颜色 (R, G, B)，默认为红色
        width: 线条宽度，默认为20
        road_label: 道路标签，用于显示标签
        show_start_label: 是否显示起点标签
        show_end_label: 是否显示终点标签
        copy: 是否复制图像，默认为True
        direction_annotation_mode: 方向标注模式，支持箭头或颜色标注
    """
    if copy:
        image = image.copy()
    draw = ImageDraw.Draw(image)

    # 获取LineString的坐标点
    coords = list(linestring.coords)

    # 转换所有坐标点为像素坐标
    pixel_coords = []
    for coord in coords:
        lon, lat = coord[:2]
        x, y = converter.lat_lon_to_pixel(lat, lon) if converter else (lon, lat)
        pixel_coords.append((x, y))

    # 绘制线条
    if len(pixel_coords) >= 2:
        draw.line(pixel_coords, fill=color, width=width)

    # 在起点和终点绘制圆点，根据方向标注模式决定颜色
    if pixel_coords:
        if direction_annotation_mode == DirectionAnnotationMode.COLORS:
            # 颜色标注模式：绿色起点，蓝色终点
            start_color = (0, 255, 0)  # 绿色
            end_color = (0, 0, 255)   # 蓝色
        else:
            # 箭头标注模式：使用折线颜色
            start_color = color
            end_color = color
        
        # 起点
        start_x, start_y = pixel_coords[0]
        draw.ellipse(
            [start_x - width, start_y - width, start_x + width, start_y + width],
            fill=start_color,
        )

        # 终点
        end_x, end_y = pixel_coords[-1]
        draw.ellipse(
            [end_x - width, end_y - width, end_x + width, end_y + width],
            fill=end_color,
        )

    # 在线段的1/3和2/3处绘制箭头标记方向（仅在箭头模式下），避免与道路标签重合
    if len(pixel_coords) >= 2 and direction_annotation_mode == DirectionAnnotationMode.ARROWS:
        # 计算线段的总长度
        total_length = 0
        segment_lengths = []
        
        # 计算每段的长度
        for i in range(len(pixel_coords) - 1):
            x1, y1 = pixel_coords[i]
            x2, y2 = pixel_coords[i+1]
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            segment_lengths.append(segment_length)
            total_length += segment_length
        
        # 在1/3和2/3处绘制箭头
        for fraction in [1/3, 2/3]:
            target_length = total_length * fraction
            current_length = 0
            segment_start = None
            segment_end = None
            
            # 找到目标点所在的段
            for i in range(len(segment_lengths)):
                if current_length <= target_length <= current_length + segment_lengths[i]:
                    segment_start = pixel_coords[i]
                    segment_end = pixel_coords[i+1]
                    break
                current_length += segment_lengths[i]
            
            # 如果找到了目标点所在的段，则绘制箭头
            if segment_start and segment_end:
                # 计算目标点位置
                segment_length = segment_lengths[i]
                segment_progress = (target_length - current_length) / segment_length if segment_length > 0 else 0
                
                point_x = segment_start[0] + segment_progress * (segment_end[0] - segment_start[0])
                point_y = segment_start[1] + segment_progress * (segment_end[1] - segment_start[1])
                
                # 计算箭头方向（使用该段的方向）
                dx = segment_end[0] - segment_start[0]
                dy = segment_end[1] - segment_start[1]
                
                # 箭头长度
                arrow_length = width * 3
                
                # 归一化方向向量
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    ux = dx / length
                    uy = dy / length
                    
                    # 计算箭头终点
                    arrow_end_x = point_x + arrow_length * ux
                    arrow_end_y = point_y + arrow_length * uy
                    
                    # 绘制箭头，颜色与折线颜色保持一致
                    _draw_arrow(draw, (point_x, point_y), (arrow_end_x, arrow_end_y), color, width)

    # 如果有道路ID，添加标签
    if pixel_coords:
        # 根据图片尺寸动态计算字体大小
        img_width, img_height = image.size
        # 字体大小设为图片较小边长的1/30，最小16，最大64
        base_font_size = min(max(min(img_width, img_height) // 30, 16), 64)

        # 尝试加载字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                base_font_size,
            )
        except Exception:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Arial.ttf", base_font_size
                )
            except Exception:
                font = ImageFont.load_default()

        # 准备标签文本
        label_text = road_label
        start_text = f"start {label_text}".strip()
        end_text = f"end {label_text}".strip()

        # 计算各种文本的尺寸
        label_bbox = draw.textbbox((0, 0), label_text, font=font)
        start_bbox = draw.textbbox((0, 0), start_text, font=font)
        end_bbox = draw.textbbox((0, 0), end_text, font=font)

        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        start_width = start_bbox[2] - start_bbox[0]
        start_height = start_bbox[3] - start_bbox[1]
        end_width = end_bbox[2] - end_bbox[0]
        end_height = end_bbox[3] - end_bbox[1]

        # 在起点添加标签
        start_x, start_y = pixel_coords[0]
        start_label_x = start_x - start_width // 2
        start_label_y = start_y - start_height - width - 5

        # 根据字体大小调整背景边距
        padding = max(base_font_size // 16, 2)

        if show_start_label and start_text:
            # 绘制起点标签背景（黑色）
            draw.rectangle(
                [
                    start_label_x - padding,
                    start_label_y - padding,
                    start_label_x + start_width + padding,
                    start_label_y + start_height + padding,
                ],
                fill=(0, 0, 0),
            )
            # 绘制起点标签文字（白色）
            draw.text(
                (start_label_x, start_label_y),
                text=start_text,
                fill=(255, 255, 255),
                font=font,
            )
        if show_end_label and end_text:
            # 在终点添加标签
            end_x, end_y = pixel_coords[-1]
            end_label_x = end_x - end_width // 2
            end_label_y = end_y + width + 5

            # 绘制终点标签背景（黑色）
            draw.rectangle(
                [
                    end_label_x - padding,
                    end_label_y - padding,
                    end_label_x + end_width + padding,
                    end_label_y + end_height + padding,
                ],
                fill=(0, 0, 0),
            )
            # 绘制终点标签文字（白色）
            draw.text(
                (end_label_x, end_label_y),
                end_text,
                fill=(255, 255, 255),
                font=font,
            )

        if label_text:
            # 在道路线中间添加标签
            # 使用Shapely计算道路线的中点
            mid_point = linestring.interpolate(0.5, normalized=True)
            mid_lon, mid_lat = mid_point.coords[0][:2]
            mid_x, mid_y = (
                converter.lat_lon_to_pixel(mid_lat, mid_lon)
                if converter
                else (mid_lon, mid_lat)
            )

            mid_label_x = mid_x - label_width // 2
            mid_label_y = mid_y - label_height // 2

            # 绘制中间标签背景（黑色）
            draw.rectangle(
                [
                    mid_label_x - padding,
                    mid_label_y - padding,
                    mid_label_x + label_width + padding,
                    mid_label_y + label_height + padding,
                ],
                fill=(0, 0, 0),
            )
            # 绘制中间标签文字（白色）
            draw.text(
                (mid_label_x, mid_label_y), label_text, fill=(255, 255, 255), font=font
            )
    return image


def crop_image(
    image: Image.Image,
    shapely_objects: List[Union[LineString, Point]],
    converter: Optional[CoordinateConverter],
    padding: int = 50,
) -> Tuple[Image.Image, Optional[CoordinateConverter], List[Union[LineString, Point]]]:
    """
    根据shapely对象列表的bounding box截取图像区域，并更新相应的converter和shapely对象坐标
    
    Args:
        image: PIL图像对象
        shapely_objects: Shapely对象列表（LineString或Point）
        converter: 坐标转换器，可以为None
        padding: 截取区域的边距（像素），默认为50
        
    Returns:
        tuple: (截取的图像, 更新后的converter, 更新后的shapely对象列表)
    """
    # 收集所有shapely对象的像素坐标
    all_pixel_coords = []
    
    for obj in shapely_objects:
        if converter is None:
            # 对象已经是像素坐标
            if isinstance(obj, LineString):
                all_pixel_coords.extend(list(obj.coords))
            elif isinstance(obj, Point):
                all_pixel_coords.append(obj.coords[0])
        else:
            # 将经纬度坐标转换为像素坐标
            if isinstance(obj, LineString):
                for coord in obj.coords:
                    lon, lat = coord[:2]
                    x, y = converter.lat_lon_to_pixel(lat, lon)
                    all_pixel_coords.append((x, y))
            elif isinstance(obj, Point):
                lon, lat = obj.coords[0][:2]
                x, y = converter.lat_lon_to_pixel(lat, lon)
                all_pixel_coords.append((x, y))
    
    # 计算所有像素坐标的bounding box
    if not all_pixel_coords:
        # 如果没有坐标点，返回原图像
        return image, converter, shapely_objects
    
    min_x = min(coord[0] for coord in all_pixel_coords)
    max_x = max(coord[0] for coord in all_pixel_coords)
    min_y = min(coord[1] for coord in all_pixel_coords)
    max_y = max(coord[1] for coord in all_pixel_coords)
    
    # 添加padding并确保在图像范围内
    crop_min_x = max(0, int(min_x - padding))
    crop_max_x = min(image.width, int(max_x + padding))
    crop_min_y = max(0, int(min_y - padding))
    crop_max_y = min(image.height, int(max_y + padding))
    
    # 截取图像
    cropped_image = image.crop((crop_min_x, crop_min_y, crop_max_x, crop_max_y))
    
    # 更新converter（如果存在）
    updated_converter = None
    if converter is not None:
        # 计算裁剪区域对应的经纬度范围
        # 左上角和右下角的经纬度
        top_left_lat, top_left_lon = converter.pixel_to_lat_lon(crop_min_x, crop_min_y)
        bottom_right_lat, bottom_right_lon = converter.pixel_to_lat_lon(crop_max_x, crop_max_y)
        
        # 创建新的bounding box
        new_bbox = BoundingBox(
            min_lat=bottom_right_lat,
            min_lon=top_left_lon,
            max_lat=top_left_lat,
            max_lon=bottom_right_lon
        )
        
        # 创建新的converter
        updated_converter = CoordinateConverter(cropped_image, new_bbox)
    
    # 更新shapely对象坐标
    updated_objects = []
    for obj in shapely_objects:
        if converter is None:
            # 如果原来就是像素坐标，需要调整相对于新图像的坐标
            if isinstance(obj, LineString):
                adjusted_coords = []
                for x, y in obj.coords:
                    new_x = x - crop_min_x
                    new_y = y - crop_min_y
                    adjusted_coords.append((new_x, new_y))
                updated_objects.append(LineString(adjusted_coords))
            elif isinstance(obj, Point):
                x, y = obj.coords[0]
                new_x = x - crop_min_x
                new_y = y - crop_min_y
                updated_objects.append(Point(new_x, new_y))
        else:
            # 如果原来是经纬度坐标，保持经纬度坐标不变
            updated_objects.append(obj)
    
    return cropped_image, updated_converter, updated_objects


def generate_reference_line_description(
    linestring: LineString,
    direction_annotation_mode: DirectionAnnotationMode = DirectionAnnotationMode.ARROWS,
) -> str:
    """
    生成参考线的文字描述，根据方向标注模式调整描述内容
    
    Args:
        linestring: 参考线的LineString对象
        direction_annotation_mode: 方向标注模式
        
    Returns:
        参考线的文字描述
    """
    if direction_annotation_mode == DirectionAnnotationMode.COLORS:
        description = """A specific road segment has been marked as a red directed polyline. The direction is indicated by colored endpoints: green dot for the start point and blue dot for the end point."""
    else:  # DirectionAnnotationMode.ARROWS
        description = """A specific road segment has been marked as a red polyline with 
arrowheads."""

    return description
