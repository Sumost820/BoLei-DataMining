# -*- coding: utf-8 -*-
"""为每个任务构造任务下发时可获得的回归特征，并从 GPS 记录计算标签。

依赖同目录或 Python 路径中的：
- LanePlanner.py
- RouteSampler.py

输出为一个标准 JSON 数组；数组中每个对象对应一个任务。

核心特征：
1. straight_line_distance_m
   起点到终点的 Haversine 直线距离。
2. planned_total_distance_m
   LanePlanner 规划结果 metrics.total_distance_m。
3. endpoint_altitude_change_m
   使用 RouteSampler 的 GPS 海拔近邻查询器分别查询起终点海拔，
   再计算终点海拔 - 起点海拔。
4. planned_slope_mean
5. planned_slope_std
   使用 RouteSampler 对规划路径离散采样，并用同一个 GPS 海拔查询器
   获取采样点海拔，再计算相邻高程差 / 路径距离差。
6. planned_cumulative_ascent_m
7. planned_cumulative_descent_m
   在平滑后的规划路径海拔序列上，将相邻点正高程差累加为累计上升，
   将负高程差的绝对值累加为累计下降。
8. vehicle_prev5_gps_speed_mean
   同一车辆最近 5 个已完成任务的全部有效 GPS speed 记录合并求均值。
9. similar_task_gps_speed_mean
   所有已完成历史任务中，起点距离 + 终点距离最小的任务的 GPS speed 均值。
10. similar_task_actual_distance_km
   被选中相似任务的真实行驶距离：最后一条有效 odometer - 第一条有效 odometer。
   SIMILAR_TASK_TOP_K > 1 时，返回所选相似任务真实行驶距离的平均值。
11. similar_task_duration_sec / similar_task_duration_min
   被选中相似任务的真实持续时间，同时输出秒和分钟。
   SIMILAR_TASK_TOP_K > 1 时，返回所选相似任务真实持续时间的平均值。
12. similar_task_energy_soc_delta_pct
   被选中相似任务的真实能耗：最早有效 SOC - 最晚有效 SOC。
   SIMILAR_TASK_TOP_K > 1 时，返回有效能耗的平均值。
10. similar_top3_duration_mean_sec
11. similar_top3_duration_median_sec
12. similar_top3_duration_std_sec
13. similar_top3_od_distance_mean_m
   按原有 OD 相似度排序，取最相似且有有效时长的前 3 条历史任务，
   计算时长均值、中位数、总体标准差，以及 OD 距离均值。
14. fleet_prev60m_speed_mean
15. fleet_prev60m_speed_min
   当前任务开始前 60 分钟内，所有车辆已经产生的有效 GPS speed
   的均值和最小值。fleet_prev30m_speed_mean 继续作为兼容字段输出。
   可以包含当时仍在执行的其他任务，但严格排除当前任务开始时刻及之后的记录。
15. vehicle_prev1_duration_sec
16. vehicle_prev5_duration_mean_sec
   同一车辆最近 1 条已完成任务的真实时长，以及最近最多 5 条已完成任务时长均值。

标签：
- task_duration_min：当前任务 GPS received_at 的最大值减最小值。
- energy_soc_delta_pct：当前任务最早有效 SOC 减最晚有效 SOC。

历史任务必须满足：历史任务 GPS 结束时间 < 当前任务 GPS 开始时间。
因此历史速度特征不会使用当前任务或未来任务的数据。
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from LanePlanner import (
    DEFAULT_SETTINGS,
    LanePlanner,
    haversine_m,
    load_lanes,
    make_point,
)
from RouteSampler import make_gps_altitude_estimator, sample_route_records


# =============================================================================
# 1. 配置区：运行前修改这里
# =============================================================================

# 原始任务 JSON 所在目录。目录中可放多辆车的 JSON 文件。
# JSON 顶层可以是任务列表，也可以是 {"tasks": [...]}。
TASK_DATA_DIR = Path(r"C:\Users\14993\PycharmProjects\BoLei-DataMining\data\GPSdata")
TASK_FILE_GLOB = "*.json"

# LanePlanner 使用的车道资源文件。
MAP_RESOURCE_FILE = Path(r"C:\Users\14993\PycharmProjects\BoLei-DataMining\data\MapResource.json")

# 输出文件：一个 JSON 数组，每个对象对应一个任务。
OUTPUT_JSON = Path(r"C:\Users\14993\PycharmProjects\BoLei-DataMining\data\任务特征和标签_20米.json")

# RouteSampler 使用的 GPS 数据目录。
# 这不是额外的海拔文件，而是直接交给 RouteSampler.make_gps_altitude_estimator()：
# 其内部会调用 load_gps_altitude_points()，读取目录下任务 JSON 的
# rt_message.longitude / latitude / altitude，并按空间近邻查询海拔。
ROUTE_SAMPLER_GPS_DATA_DIR = Path(r"C:\Users\14993\PycharmProjects\BoLei-DataMining\data\GPSdata")
GPS_ALTITUDE_MAX_DISTANCE_M = 60.0
GPS_ALTITUDE_NEIGHBORS = 8

# 规划路径离散采样设置。
ROUTE_SAMPLE_INTERVAL_M = 20.0
MAX_ROUTE_SAMPLE_POINTS = 20_000_000

# 对采样海拔做滚动中位数平滑，降低 GPS 高程噪声。
# 1 表示不平滑；建议使用奇数。
ALTITUDE_SMOOTH_WINDOW = 5

# 累计上升/下降的单步高程变化死区，单位为米。
# 平滑后绝对高程变化小于该值时视为 GPS 海拔微小抖动，不参与累计。
# 设置为 0.0 表示所有高程变化都参与累计。
ALTITUDE_CHANGE_DEADBAND_M = 0.05

# 历史特征设置。
PREVIOUS_VEHICLE_TASK_COUNT = 5

# 相似任务默认只选 OD 最相似的一条。
# 设置为 3 时，会选择最相似的 3 条，并将它们全部 GPS speed 合并求均值。
SIMILAR_TASK_TOP_K = 1

# 相似 OD 多任务统计固定使用前 3 条有效历史任务。
SIMILAR_MULTI_TASK_TOP_K = 3

# 全车队近期速度主窗口。只使用 current_start 之前已经产生的 GPS 记录。
FLEET_SPEED_WINDOW_MINUTES = 60

# 保留30分钟窗口，仅用于兼容旧版JSON/Notebook，不列入新核心特征。
FLEET_SPEED_COMPAT_WINDOW_MINUTES = 30

# 可选相似度阈值：起点距离 + 终点距离超过阈值则视为无匹配。
# None 表示不限制。
SIMILAR_TASK_MAX_OD_DISTANCE_M: Optional[float] = None

# 规划失败时是否立即停止。False 表示保留任务并将规划特征写为 null。
FAIL_FAST = False

# 是否屏蔽 LanePlanner.route_metrics 中的调试输出。
SUPPRESS_PLANNER_DEBUG_OUTPUT = True

# OD 坐标缓存精度。5 位小数约为米级，可减少重复路线规划。
ROUTE_CACHE_COORD_DECIMALS = 5


# =============================================================================
# 2. 通用工具
# =============================================================================


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        # 当前数据为无时区时间；统一去掉时区以便比较。
        return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def weighted_mean_from_sums(items: Iterable[Tuple[float, int]]) -> Optional[float]:
    total_sum = 0.0
    total_count = 0

    for value_sum, value_count in items:
        if value_count <= 0 or not math.isfinite(float(value_sum)):
            continue
        total_sum += float(value_sum)
        total_count += int(value_count)

    return total_sum / total_count if total_count > 0 else None


def to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


def extract_task_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return [item for item in data["tasks"] if isinstance(item, dict)]
    return []


def route_cache_key(task: Dict[str, Any]) -> Tuple[float, float, float, float]:
    digits = ROUTE_CACHE_COORD_DECIMALS
    return (
        round(task["start_longitude"], digits),
        round(task["start_latitude"], digits),
        round(task["end_longitude"], digits),
        round(task["end_latitude"], digits),
    )


# =============================================================================
# 3. 从当前任务 GPS 提取基础数据和标签
# =============================================================================


def sort_valid_messages(task: Dict[str, Any]) -> List[Tuple[datetime, Dict[str, Any]]]:
    messages = task.get("rt_message")
    if not isinstance(messages, list):
        return []

    parsed: List[Tuple[datetime, Dict[str, Any]]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        timestamp = parse_datetime(message.get("received_at"))
        if timestamp is not None:
            parsed.append((timestamp, message))

    parsed.sort(key=lambda item: item[0])
    return parsed


def first_last_coordinate(
    messages: Sequence[Tuple[datetime, Dict[str, Any]]],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    valid: List[Tuple[float, float]] = []

    for _, message in messages:
        lon = finite_float(message.get("longitude"))
        lat = finite_float(message.get("latitude"))
        if lon is not None and lat is not None:
            valid.append((lon, lat))

    if not valid:
        return None, None
    return valid[0], valid[-1]


def first_last_soc(
    messages: Sequence[Tuple[datetime, Dict[str, Any]]],
) -> Tuple[Optional[float], Optional[float]]:
    valid_soc = [
        soc
        for _, message in messages
        if (soc := finite_float(message.get("soc"))) is not None
    ]

    if not valid_soc:
        return None, None
    return valid_soc[0], valid_soc[-1]


def first_last_odometer(
    messages: Sequence[Tuple[datetime, Dict[str, Any]]],
) -> Tuple[Optional[float], Optional[float]]:
    """提取按时间排序后的第一条和最后一条有效 odometer。

    当前数据中的 odometer 按 km 使用，因此差值单位也是 km。
    """
    valid_odometer = [
        odometer
        for _, message in messages
        if (odometer := finite_float(message.get("odometer"))) is not None
    ]

    if not valid_odometer:
        return None, None
    return valid_odometer[0], valid_odometer[-1]


def datetime_to_epoch_us(value: datetime) -> int:
    """将无时区 datetime 转成稳定的微秒整数，避免受本机时区影响。"""
    epoch = datetime(1970, 1, 1)
    delta = value - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000
        + delta.microseconds
    )


def speed_summary(
    messages: Sequence[Tuple[datetime, Dict[str, Any]]],
) -> Tuple[float, int, Optional[float], np.ndarray, np.ndarray]:
    """汇总任务速度，并保留紧凑的时间/速度数组供 30 分钟车队窗口查询。"""
    timestamps_us: List[int] = []
    values: List[float] = []

    for timestamp, message in messages:
        speed = finite_float(message.get("speed"))
        if speed is None:
            continue
        timestamps_us.append(datetime_to_epoch_us(timestamp))
        values.append(speed)

    if not values:
        return (
            0.0,
            0,
            None,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )

    value_array = np.asarray(values, dtype=float)
    timestamp_array = np.asarray(timestamps_us, dtype=np.int64)
    value_sum = float(np.sum(value_array))
    value_count = int(value_array.size)

    return (
        value_sum,
        value_count,
        value_sum / value_count,
        timestamp_array,
        value_array,
    )


def normalize_task(
    raw_task: Dict[str, Any],
    source_file: Path,
    source_index: int,
) -> Dict[str, Any]:
    messages = sort_valid_messages(raw_task)
    start_coord, end_coord = first_last_coordinate(messages)
    start_soc, end_soc = first_last_soc(messages)
    start_odometer, end_odometer = first_last_odometer(messages)
    (
        speed_sum,
        speed_count,
        speed_mean,
        speed_times_us,
        speed_values,
    ) = speed_summary(messages)

    gps_start_time = messages[0][0] if messages else None
    gps_end_time = messages[-1][0] if messages else None

    # 优先使用 GPS 首尾坐标；若 GPS 坐标缺失，再尝试任务顶层字段。
    if start_coord is None:
        start_lon = finite_float(raw_task.get("start_longitude"))
        start_lat = finite_float(raw_task.get("start_latitude"))
        if start_lon is not None and start_lat is not None:
            start_coord = (start_lon, start_lat)

    if end_coord is None:
        end_lon = finite_float(raw_task.get("end_longitude"))
        end_lat = finite_float(raw_task.get("end_latitude"))
        if end_lon is not None and end_lat is not None:
            end_coord = (end_lon, end_lat)

    # 时间标签严格从 GPS received_at 计算。
    duration_s = None
    duration_min = None
    if gps_start_time is not None and gps_end_time is not None:
        candidate_duration_s = (gps_end_time - gps_start_time).total_seconds()
        if candidate_duration_s >= 0:
            duration_s = candidate_duration_s
            duration_min = candidate_duration_s / 60.0

    # 能耗标签严格从 GPS SOC 首尾值计算。
    energy_delta = None
    if start_soc is not None and end_soc is not None:
        energy_delta = start_soc - end_soc

    # 当前任务真实行驶距离，供后续任务构造“相似任务真实路程”特征。
    actual_distance_km = None
    if start_odometer is not None and end_odometer is not None:
        distance = end_odometer - start_odometer
        # 里程表正常情况下应单调不减；负值视为无效。
        if distance >= 0:
            actual_distance_km = distance

    return {
        "task_id": raw_task.get("task_id"),
        "transport_device_id": raw_task.get("transport_device_id"),
        "task_type_id": raw_task.get("task_type_id"),
        "sn": raw_task.get("sn"),
        "source_file": source_file.name,
        "source_task_index": source_index,
        "gps_start_time": gps_start_time,
        "gps_end_time": gps_end_time,
        "start_longitude": start_coord[0] if start_coord else None,
        "start_latitude": start_coord[1] if start_coord else None,
        "end_longitude": end_coord[0] if end_coord else None,
        "end_latitude": end_coord[1] if end_coord else None,
        "gps_speed_sum": speed_sum,
        "gps_speed_count": speed_count,
        "gps_speed_mean": speed_mean,

        # 仅供构建全车队历史速度窗口索引；写出 JSON 前会被移除。
        "_gps_speed_times_us": speed_times_us,
        "_gps_speed_values": speed_values,
        "gps_start_soc_pct": start_soc,
        "gps_end_soc_pct": end_soc,
        "gps_start_odometer_km": start_odometer,
        "gps_end_odometer_km": end_odometer,
        "gps_actual_distance_km": actual_distance_km,
        "gps_duration_sec": duration_s,
        "task_duration_min": duration_min,
        "total_energy_soc_delta_pct": energy_delta,
        "gps_message_count": len(messages),
    }


def load_all_tasks(data_dir: Path, pattern: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"目录中没有匹配的 JSON 文件：{data_dir / pattern}")

    tasks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    seen: set[Tuple[str, str]] = set()

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8-sig") as file:
                data = json.load(file)
        except Exception as exc:
            warnings.append(f"读取失败：{file_path.name}：{exc}")
            continue

        records = extract_task_list(data)
        if not records:
            warnings.append(f"未找到任务数组：{file_path.name}")
            continue

        for source_index, raw_task in enumerate(records):
            task = normalize_task(raw_task, file_path, source_index)
            key = (str(task.get("sn")), str(task.get("task_id")))

            if key in seen:
                warnings.append(f"重复任务已跳过：sn={key[0]}, task_id={key[1]}")
                continue

            seen.add(key)
            tasks.append(task)

    # 无 GPS 时间的任务排到最后，但仍保留在输出中。
    max_time = datetime.max
    tasks.sort(
        key=lambda task: (
            task["gps_start_time"] if task["gps_start_time"] is not None else max_time,
            task["gps_end_time"] if task["gps_end_time"] is not None else max_time,
        )
    )
    return tasks, warnings


# =============================================================================
# 4. 路线规划、海拔和坡度特征
# =============================================================================


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.astype(float, copy=True)

    window = int(window)
    if window % 2 == 0:
        window += 1

    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.array(
        [np.median(padded[index : index + window]) for index in range(len(values))],
        dtype=float,
    )


def calculate_slope_features(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    points: List[Tuple[float, float]] = []

    for record in records:
        distance = finite_float(record.get("distance_from_start_m"))
        altitude = finite_float(record.get("altitude_m"))
        if distance is not None and altitude is not None:
            points.append((distance, altitude))

    if len(points) < 2:
        return {
            "planned_slope_mean": None,
            "planned_slope_std": None,
            "planned_cumulative_ascent_m": None,
            "planned_cumulative_descent_m": None,
            "planned_altitude_sample_count": len(points),
        }

    points.sort(key=lambda item: item[0])

    # 删除相同距离点，避免除零。
    unique: List[Tuple[float, float]] = []
    seen_distances: set[float] = set()
    for distance, altitude in points:
        if distance not in seen_distances:
            seen_distances.add(distance)
            unique.append((distance, altitude))

    if len(unique) < 2:
        return {
            "planned_slope_mean": None,
            "planned_slope_std": None,
            "planned_cumulative_ascent_m": None,
            "planned_cumulative_descent_m": None,
            "planned_altitude_sample_count": len(unique),
        }

    distance_array = np.array([item[0] for item in unique], dtype=float)
    altitude_array = np.array([item[1] for item in unique], dtype=float)
    altitude_array = rolling_median(altitude_array, ALTITUDE_SMOOTH_WINDOW)

    delta_distance = np.diff(distance_array)
    delta_altitude = np.diff(altitude_array)
    valid = delta_distance > 1e-6

    valid_delta_altitude = delta_altitude[valid]
    slopes = valid_delta_altitude / delta_distance[valid]

    # 累计上升/下降均基于同一条平滑后的规划路径海拔序列。
    # 使用小死区忽略 GPS 海拔查询产生的厘米级抖动，避免 1 米采样时
    # 正负噪声被大量累加。
    deadband_m = max(0.0, float(ALTITUDE_CHANGE_DEADBAND_M))
    ascent_steps = valid_delta_altitude[valid_delta_altitude > deadband_m]
    descent_steps = valid_delta_altitude[valid_delta_altitude < -deadband_m]

    cumulative_ascent_m = float(np.sum(ascent_steps)) if len(ascent_steps) else 0.0
    cumulative_descent_m = (
        float(np.sum(-descent_steps)) if len(descent_steps) else 0.0
    )

    return {
        "planned_slope_mean": float(np.mean(slopes)) if len(slopes) else None,
        "planned_slope_std": float(np.std(slopes, ddof=0)) if len(slopes) else None,
        "planned_cumulative_ascent_m": cumulative_ascent_m,
        "planned_cumulative_descent_m": cumulative_descent_m,
        "planned_altitude_sample_count": len(unique),
    }


def query_endpoint_altitude(
    longitude: float,
    latitude: float,
    altitude_provider,
) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    if altitude_provider is None:
        return None, None, None

    info = altitude_provider(float(longitude), float(latitude), None)
    return (
        finite_float(info.get("altitude_m")),
        info.get("altitude_source"),
        finite_float(info.get("altitude_nearest_distance_m")),
    )


def first_last_sample_altitude(
    records: Sequence[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    valid_altitudes = [
        altitude
        for record in records
        if (altitude := finite_float(record.get("altitude_m"))) is not None
    ]
    if not valid_altitudes:
        return None, None
    return valid_altitudes[0], valid_altitudes[-1]


def construct_route_features(
    task: Dict[str, Any],
    planner: LanePlanner,
    altitude_provider,
) -> Dict[str, Any]:
    required = (
        task.get("start_longitude"),
        task.get("start_latitude"),
        task.get("end_longitude"),
        task.get("end_latitude"),
    )
    if any(value is None for value in required):
        raise ValueError("任务缺少有效起终点坐标")

    origin = make_point(task["start_longitude"], task["start_latitude"])
    destination = make_point(task["end_longitude"], task["end_latitude"])

    if SUPPRESS_PLANNER_DEBUG_OUTPUT:
        with contextlib.redirect_stdout(io.StringIO()):
            plan, tried = planner.plan(origin, destination)
    else:
        plan, tried = planner.plan(origin, destination)

    records, sample_meta = sample_route_records(
        planner=planner,
        plan=plan,
        interval_m=ROUTE_SAMPLE_INTERVAL_M,
        include_end=True,
        max_records=MAX_ROUTE_SAMPLE_POINTS,
        altitude_provider=altitude_provider,
    )

    metrics = plan.get("metrics") or {}
    slope_features = calculate_slope_features(records)

    # 起终点海拔优先按精确 OD 坐标查询。
    start_altitude, start_altitude_source, start_altitude_nearest_m = query_endpoint_altitude(
        task["start_longitude"], task["start_latitude"], altitude_provider
    )
    end_altitude, end_altitude_source, end_altitude_nearest_m = query_endpoint_altitude(
        task["end_longitude"], task["end_latitude"], altitude_provider
    )

    # 如果精确 OD 查询失败，则用规划路径首尾采样点海拔兜底。
    sample_start_altitude, sample_end_altitude = first_last_sample_altitude(records)
    if start_altitude is None:
        start_altitude = sample_start_altitude
        start_altitude_source = "planned_route_sample_fallback" if start_altitude is not None else None
    if end_altitude is None:
        end_altitude = sample_end_altitude
        end_altitude_source = "planned_route_sample_fallback" if end_altitude is not None else None

    altitude_change = None
    if start_altitude is not None and end_altitude is not None:
        altitude_change = end_altitude - start_altitude

    result = {
        # 用户确认：车道距离使用规划 metrics.total_distance_m。
        "planned_total_distance_m": finite_float(metrics.get("total_distance_m")),

        # 用户确认：爬升定义为终点查询海拔 - 起点查询海拔。
        "start_altitude_m": start_altitude,
        "end_altitude_m": end_altitude,
        "endpoint_altitude_change_m": altitude_change,

        "start_altitude_source": start_altitude_source,
        "end_altitude_source": end_altitude_source,
        "start_altitude_nearest_distance_m": start_altitude_nearest_m,
        "end_altitude_nearest_distance_m": end_altitude_nearest_m,

        # 质量检查字段，不必作为模型输入。
        "planned_lane_distance_m_debug": finite_float(metrics.get("lane_distance_m")),
        "planned_snap_distance_m": finite_float(metrics.get("snap_distance_m")),
        "planned_jump_distance_m": finite_float(metrics.get("jump_distance_m")),
        "planned_lane_count": len(plan.get("path") or []),
        "planned_lane_uids": list(plan.get("path") or []),
        "planned_route_sample_count": int(sample_meta.get("sample_count", len(records))),
        "planned_route_candidate_count": int(plan.get("tried_count", len(tried))),
        "route_planning_error": None,
    }
    result.update(slope_features)
    return result


def empty_route_features(error: str) -> Dict[str, Any]:
    return {
        "planned_total_distance_m": None,
        "start_altitude_m": None,
        "end_altitude_m": None,
        "endpoint_altitude_change_m": None,
        "planned_slope_mean": None,
        "planned_slope_std": None,
        "planned_cumulative_ascent_m": None,
        "planned_cumulative_descent_m": None,
        "planned_altitude_sample_count": 0,
        "start_altitude_source": None,
        "end_altitude_source": None,
        "start_altitude_nearest_distance_m": None,
        "end_altitude_nearest_distance_m": None,
        "planned_lane_distance_m_debug": None,
        "planned_snap_distance_m": None,
        "planned_jump_distance_m": None,
        "planned_lane_count": None,
        "planned_lane_uids": None,
        "planned_route_sample_count": 0,
        "planned_route_candidate_count": 0,
        "route_planning_error": error,
    }


# =============================================================================
# 5. 历史速度特征
# =============================================================================


def build_fleet_speed_index(
    tasks: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """建立按 GPS 时间排序的全车队速度索引。

    返回：
    - sorted_times_us：每条有效 speed 对应的 GPS 时间，升序；
    - prefix_speed_sum：速度前缀和，长度比时间数组多 1；
    - sorted_speeds：与时间数组一一对应的速度值。

    构建完成后会删除任务字典中的私有速度数组，降低后续内存占用。
    """
    time_chunks: List[np.ndarray] = []
    speed_chunks: List[np.ndarray] = []

    for task in tasks:
        times = task.pop("_gps_speed_times_us", None)
        speeds = task.pop("_gps_speed_values", None)

        if (
            isinstance(times, np.ndarray)
            and isinstance(speeds, np.ndarray)
            and times.size > 0
            and times.size == speeds.size
        ):
            time_chunks.append(times.astype(np.int64, copy=False))
            speed_chunks.append(speeds.astype(float, copy=False))

    if not time_chunks:
        return (
            np.empty(0, dtype=np.int64),
            np.zeros(1, dtype=float),
            np.empty(0, dtype=float),
        )

    all_times = np.concatenate(time_chunks)
    all_speeds = np.concatenate(speed_chunks)

    order = np.argsort(all_times, kind="stable")
    sorted_times = all_times[order]
    sorted_speeds = all_speeds[order]

    prefix_sum = np.empty(sorted_speeds.size + 1, dtype=float)
    prefix_sum[0] = 0.0
    np.cumsum(sorted_speeds, out=prefix_sum[1:])

    return sorted_times, prefix_sum, sorted_speeds


def fleet_speed_stats_before_start(
    current_start: datetime,
    fleet_speed_index: Tuple[np.ndarray, np.ndarray, np.ndarray],
    window_minutes: int,
) -> Tuple[Optional[float], Optional[float], int]:
    """计算任务开始前指定窗口内全车队 speed 的均值、最小值和记录数。

    时间范围严格为：
        [current_start - window_minutes, current_start)

    当前任务开始时刻及之后的 GPS 记录不会参与计算。
    """
    sorted_times_us, prefix_speed_sum, sorted_speeds = fleet_speed_index
    if sorted_times_us.size == 0:
        return None, None, 0

    current_us = datetime_to_epoch_us(current_start)
    window_us = int(window_minutes * 60 * 1_000_000)
    window_start_us = current_us - window_us

    left = int(np.searchsorted(sorted_times_us, window_start_us, side="left"))
    right = int(np.searchsorted(sorted_times_us, current_us, side="left"))

    count = right - left
    if count <= 0:
        return None, None, 0

    speed_sum = float(prefix_speed_sum[right] - prefix_speed_sum[left])
    speed_mean = speed_sum / count
    speed_min = float(np.min(sorted_speeds[left:right]))

    return speed_mean, speed_min, count

def od_pair_distance_m(current: Dict[str, Any], history: Dict[str, Any]) -> float:
    current_start = make_point(current["start_longitude"], current["start_latitude"])
    history_start = make_point(history["start_longitude"], history["start_latitude"])
    current_end = make_point(current["end_longitude"], current["end_latitude"])
    history_end = make_point(history["end_longitude"], history["end_latitude"])

    return float(
        haversine_m(current_start, history_start)
        + haversine_m(current_end, history_end)
    )


def construct_history_features(
    current: Dict[str, Any],
    all_tasks: Sequence[Dict[str, Any]],
    fleet_speed_index: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    current_start = current.get("gps_start_time")

    if current_start is None:
        return {
            "vehicle_prev5_gps_speed_mean": None,
            "vehicle_prev5_task_count": 0,
            "vehicle_prev5_speed_record_count": 0,
            "vehicle_prev1_duration_sec": None,
            "vehicle_prev5_duration_mean_sec": None,
            "vehicle_prev1_duration_min": None,
            "vehicle_prev5_duration_mean_min": None,

            "similar_task_gps_speed_mean": None,
            "similar_task_actual_distance_km": None,
            "similar_task_duration_sec": None,
            "similar_task_duration_min": None,
            "similar_task_energy_soc_delta_pct": None,
            "similar_task_energy_count": 0,
            "similar_top3_duration_mean_sec": None,
            "similar_top3_duration_median_sec": None,
            "similar_top3_duration_std_sec": None,
            "similar_top3_duration_mean_min": None,
            "similar_top3_duration_median_min": None,
            "similar_top3_duration_std_min": None,
            "similar_top3_od_distance_mean_m": None,
            "similar_task_actual_distance_count": 0,
            "similar_task_count": 0,
            "similar_task_speed_record_count": 0,
            "similar_task_od_distance_m": None,
            "similar_task_time_gap_min": None,
            "similar_task_source_task_ids": [],
            "similar_task_source_sns": [],
            "similar_task_candidate_count": 0,

            "fleet_prev30m_speed_mean": None,
            "fleet_prev60m_speed_mean": None,
            "fleet_prev60m_speed_min": None,
            "fleet_prev60m_speed_record_count": 0,
        }

    # 全车队近期速度：可包含当时仍在执行的其他任务，
    # 但严格只统计 current_start 之前已经产生的 GPS 记录。
    (
        fleet_prev60m_speed_mean,
        fleet_prev60m_speed_min,
        fleet_prev60m_speed_record_count,
    ) = fleet_speed_stats_before_start(
        current_start=current_start,
        fleet_speed_index=fleet_speed_index,
        window_minutes=FLEET_SPEED_WINDOW_MINUTES,
    )

    # 兼容旧版Notebook；新推荐特征使用60分钟窗口。
    (
        fleet_prev30m_speed_mean,
        _fleet_prev30m_speed_min,
        _fleet_prev30m_speed_record_count,
    ) = fleet_speed_stats_before_start(
        current_start=current_start,
        fleet_speed_index=fleet_speed_index,
        window_minutes=FLEET_SPEED_COMPAT_WINDOW_MINUTES,
    )

    # 所有在当前任务开始前已经结束的历史任务。
    completed_all = [
        task
        for task in all_tasks
        if task.get("gps_end_time") is not None
        and task["gps_end_time"] < current_start
    ]

    # 保持原有速度和相似任务逻辑：候选任务必须有有效 GPS speed。
    completed_with_speed = [
        task
        for task in completed_all
        if int(task.get("gps_speed_count", 0)) > 0
    ]

    # 同一车辆最近 5 个已完成任务：合并全部 GPS speed 记录后求均值。
    same_vehicle_speed = [
        task
        for task in completed_with_speed
        if str(task.get("transport_device_id"))
        == str(current.get("transport_device_id"))
    ]
    same_vehicle_speed.sort(
        key=lambda task: task["gps_end_time"],
        reverse=True,
    )
    previous_speed_tasks = same_vehicle_speed[:PREVIOUS_VEHICLE_TASK_COUNT]

    previous_speed_mean = weighted_mean_from_sums(
        (task["gps_speed_sum"], task["gps_speed_count"])
        for task in previous_speed_tasks
    )
    previous_speed_record_count = sum(
        int(task["gps_speed_count"]) for task in previous_speed_tasks
    )

    # 同一车辆历史时长。只要求任务已完成且时长有效，不依赖 speed 字段。
    same_vehicle_duration = [
        task
        for task in completed_all
        if str(task.get("transport_device_id"))
        == str(current.get("transport_device_id"))
        and task.get("gps_duration_sec") is not None
    ]
    same_vehicle_duration.sort(
        key=lambda task: task["gps_end_time"],
        reverse=True,
    )
    previous_duration_tasks = same_vehicle_duration[
        :PREVIOUS_VEHICLE_TASK_COUNT
    ]
    previous_durations_sec = [
        float(task["gps_duration_sec"])
        for task in previous_duration_tasks
    ]

    vehicle_prev1_duration_sec = (
        previous_durations_sec[0]
        if previous_durations_sec
        else None
    )
    vehicle_prev5_duration_mean_sec = (
        float(np.mean(previous_durations_sec))
        if previous_durations_sec
        else None
    )

    vehicle_prev1_duration_min = (
        vehicle_prev1_duration_sec / 60.0
        if vehicle_prev1_duration_sec is not None
        else None
    )
    vehicle_prev5_duration_mean_min = (
        vehicle_prev5_duration_mean_sec / 60.0
        if vehicle_prev5_duration_mean_sec is not None
        else None
    )

    # 不设时间窗口、不限制车辆；按起点距离 + 终点距离排序。
    current_has_od = all(
        current.get(field) is not None
        for field in (
            "start_longitude",
            "start_latitude",
            "end_longitude",
            "end_latitude",
        )
    )

    ranked: List[Tuple[float, float, Dict[str, Any]]] = []
    if current_has_od:
        for history in completed_with_speed:
            history_has_od = all(
                history.get(field) is not None
                for field in (
                    "start_longitude",
                    "start_latitude",
                    "end_longitude",
                    "end_latitude",
                )
            )
            if not history_has_od:
                continue

            distance = od_pair_distance_m(current, history)
            if (
                SIMILAR_TASK_MAX_OD_DISTANCE_M is not None
                and distance > SIMILAR_TASK_MAX_OD_DISTANCE_M
            ):
                continue

            time_gap_s = (
                current_start - history["gps_end_time"]
            ).total_seconds()
            ranked.append((distance, time_gap_s, history))

    # 先按 OD 距离，再按时间更近排序。
    ranked.sort(key=lambda item: (item[0], item[1]))

    # 原有单条/可配置 TOP_K 相似任务特征。
    selected = ranked[: max(1, int(SIMILAR_TASK_TOP_K))]

    similar_speed_mean = weighted_mean_from_sums(
        (item[2]["gps_speed_sum"], item[2]["gps_speed_count"])
        for item in selected
    )
    similar_speed_record_count = sum(
        int(item[2]["gps_speed_count"]) for item in selected
    )

    selected_actual_distances = [
        float(item[2]["gps_actual_distance_km"])
        for item in selected
        if item[2].get("gps_actual_distance_km") is not None
    ]
    similar_actual_distance_km = (
        float(np.mean(selected_actual_distances))
        if selected_actual_distances
        else None
    )

    selected_durations_sec = [
        float(item[2]["gps_duration_sec"])
        for item in selected
        if item[2].get("gps_duration_sec") is not None
    ]
    similar_task_duration_sec = (
        float(np.mean(selected_durations_sec))
        if selected_durations_sec
        else None
    )
    similar_task_duration_min = (
        similar_task_duration_sec / 60.0
        if similar_task_duration_sec is not None
        else None
    )

    # 相似任务真实能耗：与其他单条相似任务特征使用同一个 selected。
    # TOP_K=1 时返回该任务能耗；TOP_K>1 时返回有效能耗均值。
    selected_energy_deltas = [
        float(item[2]["total_energy_soc_delta_pct"])
        for item in selected
        if item[2].get("total_energy_soc_delta_pct") is not None
    ]
    similar_task_energy_soc_delta_pct = (
        float(np.mean(selected_energy_deltas))
        if selected_energy_deltas
        else None
    )

    # 新增：从相同 ranked 列表中取最相似且时长有效的前 3 条。
    similar_top3_items: List[Tuple[float, float, Dict[str, Any]]] = []
    for item in ranked:
        if item[2].get("gps_duration_sec") is None:
            continue
        similar_top3_items.append(item)
        if len(similar_top3_items) >= SIMILAR_MULTI_TASK_TOP_K:
            break

    similar_top3_durations = np.asarray(
        [
            float(item[2]["gps_duration_sec"])
            for item in similar_top3_items
        ],
        dtype=float,
    )

    if similar_top3_durations.size > 0:
        similar_top3_duration_mean_sec = float(
            np.mean(similar_top3_durations)
        )
        similar_top3_duration_median_sec = float(
            np.median(similar_top3_durations)
        )
        similar_top3_duration_std_sec = float(
            np.std(similar_top3_durations, ddof=0)
        )
        similar_top3_od_distance_mean_m = float(
            np.mean([item[0] for item in similar_top3_items])
        )
    else:
        similar_top3_duration_mean_sec = None
        similar_top3_duration_median_sec = None
        similar_top3_duration_std_sec = None
        similar_top3_od_distance_mean_m = None

    similar_top3_duration_mean_min = (
        similar_top3_duration_mean_sec / 60.0
        if similar_top3_duration_mean_sec is not None
        else None
    )
    similar_top3_duration_median_min = (
        similar_top3_duration_median_sec / 60.0
        if similar_top3_duration_median_sec is not None
        else None
    )
    similar_top3_duration_std_min = (
        similar_top3_duration_std_sec / 60.0
        if similar_top3_duration_std_sec is not None
        else None
    )

    return {
        "vehicle_prev5_gps_speed_mean": previous_speed_mean,
        "vehicle_prev5_task_count": len(previous_speed_tasks),
        "vehicle_prev5_speed_record_count": previous_speed_record_count,
        "vehicle_prev1_duration_sec": vehicle_prev1_duration_sec,
        "vehicle_prev5_duration_mean_sec": vehicle_prev5_duration_mean_sec,
        "vehicle_prev1_duration_min": vehicle_prev1_duration_min,
        "vehicle_prev5_duration_mean_min": vehicle_prev5_duration_mean_min,

        "similar_task_gps_speed_mean": similar_speed_mean,
        "similar_task_actual_distance_km": similar_actual_distance_km,
        "similar_task_duration_sec": similar_task_duration_sec,
        "similar_task_duration_min": similar_task_duration_min,
        "similar_task_energy_soc_delta_pct": similar_task_energy_soc_delta_pct,
        "similar_task_energy_count": len(selected_energy_deltas),
        "similar_top3_duration_mean_sec": similar_top3_duration_mean_sec,
        "similar_top3_duration_median_sec": similar_top3_duration_median_sec,
        "similar_top3_duration_std_sec": similar_top3_duration_std_sec,
        "similar_top3_duration_mean_min": similar_top3_duration_mean_min,
        "similar_top3_duration_median_min": similar_top3_duration_median_min,
        "similar_top3_duration_std_min": similar_top3_duration_std_min,
        "similar_top3_od_distance_mean_m": similar_top3_od_distance_mean_m,
        "similar_task_actual_distance_count": len(selected_actual_distances),
        "similar_task_count": len(selected),
        "similar_task_speed_record_count": similar_speed_record_count,
        "similar_task_od_distance_m": (
            float(np.mean([item[0] for item in selected]))
            if selected
            else None
        ),
        "similar_task_time_gap_min": (
            float(np.mean([item[1] / 60.0 for item in selected]))
            if selected
            else None
        ),
        "similar_task_source_task_ids": [
            item[2]["task_id"] for item in selected
        ],
        "similar_task_source_sns": [
            item[2]["sn"] for item in selected
        ],
        "similar_task_candidate_count": len(ranked),

        "fleet_prev30m_speed_mean": fleet_prev30m_speed_mean,
        "fleet_prev60m_speed_mean": fleet_prev60m_speed_mean,
        "fleet_prev60m_speed_min": fleet_prev60m_speed_min,
        "fleet_prev60m_speed_record_count": fleet_prev60m_speed_record_count,
    }


# =============================================================================
# 6. 输出记录
# =============================================================================


def straight_line_distance(task: Dict[str, Any]) -> Optional[float]:
    fields = (
        task.get("start_longitude"),
        task.get("start_latitude"),
        task.get("end_longitude"),
        task.get("end_latitude"),
    )
    if any(value is None for value in fields):
        return None

    return float(
        haversine_m(
            make_point(task["start_longitude"], task["start_latitude"]),
            make_point(task["end_longitude"], task["end_latitude"]),
        )
    )


def build_output_record(
    task: Dict[str, Any],
    route_features: Dict[str, Any],
    history_features: Dict[str, Any],
) -> Dict[str, Any]:
    output = {
        # 任务标识和数据来源。
        "task_id": task.get("task_id"),
        "transport_device_id": task.get("transport_device_id"),
        "task_type_id": task.get("task_type_id"),
        "sn": task.get("sn"),
        "source_file": task.get("source_file"),
        "source_task_index": task.get("source_task_index"),

        # GPS 首尾信息。
        "actual_start_time": task.get("gps_start_time"),
        "actual_end_time": task.get("gps_end_time"),
        "start_longitude": task.get("start_longitude"),
        "start_latitude": task.get("start_latitude"),
        "end_longitude": task.get("end_longitude"),
        "end_latitude": task.get("end_latitude"),

        # 特征 1。
        "straight_line_distance_m": straight_line_distance(task),

        # 两个监督学习标签，均由当前任务 GPS 记录计算。
        "task_duration_min": task.get("task_duration_min"),
        "total_energy_soc_delta_pct": task.get("total_energy_soc_delta_pct"),

    }

    output.update(route_features)
    output.update(history_features)
    return output


# 模型建议使用的 20 个核心特征。
MODEL_FEATURE_COLUMNS = [
    "straight_line_distance_m",
    "planned_total_distance_m",
    "endpoint_altitude_change_m",
    "planned_slope_mean",
    "planned_slope_std",
    "planned_cumulative_ascent_m",
    "planned_cumulative_descent_m",

    "vehicle_prev5_gps_speed_mean",
    "vehicle_prev1_duration_sec",
    "vehicle_prev5_duration_mean_sec",

    "similar_task_gps_speed_mean",
    "similar_task_actual_distance_km",
    "similar_task_duration_min",
    "similar_task_energy_soc_delta_pct",
    "similar_top3_duration_mean_sec",
    "similar_top3_duration_median_sec",
    "similar_top3_duration_std_sec",
    "similar_top3_od_distance_mean_m",

    "fleet_prev60m_speed_mean",
    "fleet_prev60m_speed_min",
]

LABEL_COLUMNS = [
    "task_duration_min",
    "total_energy_soc_delta_pct",
]


# =============================================================================
# 7. 主流程
# =============================================================================


def main() -> None:
    if not TASK_DATA_DIR.exists():
        raise FileNotFoundError(f"任务数据目录不存在：{TASK_DATA_DIR}")
    if not MAP_RESOURCE_FILE.exists():
        raise FileNotFoundError(f"车道资源文件不存在：{MAP_RESOURCE_FILE}")

    print("[1/5] 读取并解析全部任务 GPS……")
    tasks, warnings = load_all_tasks(TASK_DATA_DIR, TASK_FILE_GLOB)
    if not tasks:
        raise RuntimeError("未读取到任务。")
    print(f"任务数量：{len(tasks)}")

    print("构建全车队 GPS 速度时间索引……")
    fleet_speed_index = build_fleet_speed_index(tasks)
    print(
        "有效 GPS speed 记录数："
        f"{fleet_speed_index[0].size}"
    )

    print("[2/5] 初始化 LanePlanner……")
    planner_settings = deepcopy(DEFAULT_SETTINGS)
    lanes = load_lanes(MAP_RESOURCE_FILE, planner_settings)
    planner = LanePlanner(lanes, planner_settings)
    print(f"车道数量：{len(lanes)}")

    print("[3/5] 初始化海拔查询器……")
    if not ROUTE_SAMPLER_GPS_DATA_DIR.exists():
        raise FileNotFoundError(
            f"RouteSampler GPS 数据目录不存在：{ROUTE_SAMPLER_GPS_DATA_DIR}"
        )

    altitude_provider = make_gps_altitude_estimator(
        gps_data_dir=ROUTE_SAMPLER_GPS_DATA_DIR,
        max_distance_m=GPS_ALTITUDE_MAX_DISTANCE_M,
        k=GPS_ALTITUDE_NEIGHBORS,
    )
    print("海拔来源：RouteSampler GPS近邻查询；路径采样时车道z仅作兜底")

    print("[4/5] 构造路线特征、相似OD/同车历史/车队近期特征和标签……")
    outputs: List[Dict[str, Any]] = []
    route_cache: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}
    planning_failure_count = 0

    for index, task in enumerate(tasks, start=1):
        if all(
            task.get(field) is not None
            for field in (
                "start_longitude",
                "start_latitude",
                "end_longitude",
                "end_latitude",
            )
        ):
            cache_key = route_cache_key(task)
            if cache_key in route_cache:
                route_features = deepcopy(route_cache[cache_key])
            else:
                try:
                    route_features = construct_route_features(
                        task=task,
                        planner=planner,
                        altitude_provider=altitude_provider,
                    )
                    route_cache[cache_key] = deepcopy(route_features)
                except Exception as exc:
                    planning_failure_count += 1
                    route_features = empty_route_features(
                        f"{type(exc).__name__}: {exc}"
                    )
                    if FAIL_FAST:
                        raise
        else:
            planning_failure_count += 1
            route_features = empty_route_features("缺少有效起终点 GPS 坐标")

        history_features = construct_history_features(
            current=task,
            all_tasks=tasks,
            fleet_speed_index=fleet_speed_index,
        )
        outputs.append(build_output_record(task, route_features, history_features))

        if index % 50 == 0 or index == len(tasks):
            print(
                f"进度：{index}/{len(tasks)}，"
                f"规划失败：{planning_failure_count}，"
                f"路线缓存：{len(route_cache)}"
            )

    print("[5/5] 写入 JSON……")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        # 顶层直接输出任务数组，数组中每个对象对应一个任务。
        json.dump(
            to_json_safe(outputs),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"输出文件：{OUTPUT_JSON}")
    print(f"输出任务数量：{len(outputs)}")
    print(f"规划失败数量：{planning_failure_count}")
    print(f"读取警告数量：{len(warnings)}")
    print("模型核心特征：")
    for column in MODEL_FEATURE_COLUMNS:
        print(f"  - {column}")
    print("标签：")
    for column in LABEL_COLUMNS:
        print(f"  - {column}")

    if warnings:
        warning_file = OUTPUT_JSON.with_name(f"{OUTPUT_JSON.stem}_warnings.txt")
        warning_file.write_text("\n".join(warnings), encoding="utf-8")
        print(f"读取警告详情：{warning_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise