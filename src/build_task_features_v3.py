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
6. vehicle_prev5_gps_speed_mean
   同一车辆最近 5 个已完成任务的全部有效 GPS speed 记录合并求均值。
7. similar_task_gps_speed_mean
   所有已完成历史任务中，起点距离 + 终点距离最小的任务的 GPS speed 均值。

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
OUTPUT_JSON = Path(r"C:\Users\14993\PycharmProjects\BoLei-DataMining\data\任务特征和标签.json")

# RouteSampler 使用的 GPS 数据目录。
# 这不是额外的海拔文件，而是直接交给 RouteSampler.make_gps_altitude_estimator()：
# 其内部会调用 load_gps_altitude_points()，读取目录下任务 JSON 的
# rt_message.longitude / latitude / altitude，并按空间近邻查询海拔。
ROUTE_SAMPLER_GPS_DATA_DIR = TASK_DATA_DIR = Path(r"C:\Users\14993\PycharmProjects\BoLei-DataMining\data\GPSdata")
GPS_ALTITUDE_MAX_DISTANCE_M = 60.0
GPS_ALTITUDE_NEIGHBORS = 8

# 规划路径离散采样设置。
ROUTE_SAMPLE_INTERVAL_M = 20.0
MAX_ROUTE_SAMPLE_POINTS = 20_000

# 对采样海拔做滚动中位数平滑，降低 GPS 高程噪声。
# 1 表示不平滑；建议使用奇数。
ALTITUDE_SMOOTH_WINDOW = 5

# 历史特征设置。
PREVIOUS_VEHICLE_TASK_COUNT = 5

# 相似任务默认只选 OD 最相似的一条。
# 设置为 3 时，会选择最相似的 3 条，并将它们全部 GPS speed 合并求均值。
SIMILAR_TASK_TOP_K = 1

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


def speed_summary(
    messages: Sequence[Tuple[datetime, Dict[str, Any]]],
) -> Tuple[float, int, Optional[float]]:
    values = [
        speed
        for _, message in messages
        if (speed := finite_float(message.get("speed"))) is not None
    ]

    if not values:
        return 0.0, 0, None

    value_sum = float(np.sum(values))
    value_count = len(values)
    return value_sum, value_count, value_sum / value_count


def normalize_task(
    raw_task: Dict[str, Any],
    source_file: Path,
    source_index: int,
) -> Dict[str, Any]:
    messages = sort_valid_messages(raw_task)
    start_coord, end_coord = first_last_coordinate(messages)
    start_soc, end_soc = first_last_soc(messages)
    speed_sum, speed_count, speed_mean = speed_summary(messages)

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
    duration_min = None
    if gps_start_time is not None and gps_end_time is not None:
        duration_s = (gps_end_time - gps_start_time).total_seconds()
        if duration_s >= 0:
            duration_min = duration_s / 60.0

    # 能耗标签严格从 GPS SOC 首尾值计算。
    energy_delta = None
    if start_soc is not None and end_soc is not None:
        energy_delta = start_soc - end_soc

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
        "gps_start_soc_pct": start_soc,
        "gps_end_soc_pct": end_soc,
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
            "planned_altitude_sample_count": len(unique),
        }

    distance_array = np.array([item[0] for item in unique], dtype=float)
    altitude_array = np.array([item[1] for item in unique], dtype=float)
    altitude_array = rolling_median(altitude_array, ALTITUDE_SMOOTH_WINDOW)

    delta_distance = np.diff(distance_array)
    delta_altitude = np.diff(altitude_array)
    valid = delta_distance > 1e-6

    slopes = delta_altitude[valid] / delta_distance[valid]

    return {
        "planned_slope_mean": float(np.mean(slopes)) if len(slopes) else None,
        "planned_slope_std": float(np.std(slopes, ddof=0)) if len(slopes) else None,
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
) -> Dict[str, Any]:
    current_start = current.get("gps_start_time")

    if current_start is None:
        return {
            "vehicle_prev5_gps_speed_mean": None,
            "vehicle_prev5_task_count": 0,
            "vehicle_prev5_speed_record_count": 0,
            "similar_task_gps_speed_mean": None,
            "similar_task_count": 0,
            "similar_task_speed_record_count": 0,
            "similar_task_od_distance_m": None,
            "similar_task_time_gap_min": None,
            "similar_task_source_task_ids": [],
            "similar_task_source_sns": [],
            "similar_task_candidate_count": 0,
        }

    # 只能使用当前任务开始前已经结束、且有有效 speed 的历史任务。
    completed = [
        task
        for task in all_tasks
        if task.get("gps_end_time") is not None
        and task["gps_end_time"] < current_start
        and int(task.get("gps_speed_count", 0)) > 0
    ]

    # 6. 同一车辆最近 5 个任务：合并这 5 个任务全部 GPS speed 记录后求均值。
    same_vehicle = [
        task
        for task in completed
        if str(task.get("transport_device_id"))
        == str(current.get("transport_device_id"))
    ]
    same_vehicle.sort(key=lambda task: task["gps_end_time"], reverse=True)
    previous_tasks = same_vehicle[:PREVIOUS_VEHICLE_TASK_COUNT]

    previous_speed_mean = weighted_mean_from_sums(
        (task["gps_speed_sum"], task["gps_speed_count"])
        for task in previous_tasks
    )
    previous_speed_record_count = sum(
        int(task["gps_speed_count"]) for task in previous_tasks
    )

    # 7. 不设时间窗口、不限制车辆；仅按起终点经纬度相似度排序。
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
        for history in completed:
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

            time_gap_s = (current_start - history["gps_end_time"]).total_seconds()
            ranked.append((distance, time_gap_s, history))

    # 先按 OD 距离，再按时间更近排序。
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = ranked[: max(1, int(SIMILAR_TASK_TOP_K))]

    similar_speed_mean = weighted_mean_from_sums(
        (item[2]["gps_speed_sum"], item[2]["gps_speed_count"])
        for item in selected
    )
    similar_speed_record_count = sum(
        int(item[2]["gps_speed_count"]) for item in selected
    )

    return {
        "vehicle_prev5_gps_speed_mean": previous_speed_mean,
        "vehicle_prev5_task_count": len(previous_tasks),
        "vehicle_prev5_speed_record_count": previous_speed_record_count,

        "similar_task_gps_speed_mean": similar_speed_mean,
        "similar_task_count": len(selected),
        "similar_task_speed_record_count": similar_speed_record_count,
        "similar_task_od_distance_m": (
            float(np.mean([item[0] for item in selected])) if selected else None
        ),
        "similar_task_time_gap_min": (
            float(np.mean([item[1] / 60.0 for item in selected])) if selected else None
        ),
        "similar_task_source_task_ids": [item[2]["task_id"] for item in selected],
        "similar_task_source_sns": [item[2]["sn"] for item in selected],
        "similar_task_candidate_count": len(ranked),
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


# 模型应使用的 7 个核心特征。
MODEL_FEATURE_COLUMNS = [
    "straight_line_distance_m",
    "planned_total_distance_m",
    "endpoint_altitude_change_m",
    "planned_slope_mean",
    "planned_slope_std",
    "vehicle_prev5_gps_speed_mean",
    "similar_task_gps_speed_mean",
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

    print("[4/5] 构造路线特征、历史速度特征和标签……")
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

        history_features = construct_history_features(task, tasks)
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
