"""任务时间/能耗预测特征构造（v6：仅时间/能耗双目标，速度仅作为时间历史特征）。

设计目标：
1. 速度只保留一个概念：task_speed_mps = odometer 实际里程 / 去暂停后的有效时长；
2. GPS `speed` 仅用于识别暂停区间（speed < PAUSE_SPEED_THRESHOLD），不再构造 GPS speed 均值、active speed、pace 等第二套速度特征；
3. 历史任务严格要求 gps_end_time < current_start，避免未来信息泄漏；
4. 最终监督目标只有任务时间和任务能耗；速度不作为预测目标；
5. task_speed_mps 仅在特征构建内部作为“已完成历史任务”的中间量，用于增强时间预测；
6. 时间模型使用 30 个特征，其中可包含历史速度特征；
7. 能耗模型使用独立的 20 个特征，不使用速度派生特征；
8. 所有历史特征严格使用当前任务开始前已完成任务，避免未来信息泄漏。

输出标签（仅两个）：
- task_duration_min：去暂停后的任务时长（分钟）；
- total_energy_soc_delta_pct：SOC 首尾差。

说明：
- gps_task_speed_mps 只保存在内存中的标准化任务对象里，供历史速度特征计算；
- 不把当前任务 task_speed_mps 输出为标签，也不训练速度预测模型；
- gps_pause_ratio 仅作为内部诊断量，不作为监督标签。
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
# 1. 配置
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DATA_DIR = PROJECT_ROOT / "data" / "GPSdata"
TASK_FILE_GLOB = "*.json"
MAP_RESOURCE_FILE = PROJECT_ROOT / "data" / "MapResource.json"
OUTPUT_JSON = PROJECT_ROOT / "data" / "任务特征和标签_20米_特征扩充.json"
ROUTE_SAMPLER_GPS_DATA_DIR = TASK_DATA_DIR

GPS_ALTITUDE_MAX_DISTANCE_M = 60.0
GPS_ALTITUDE_NEIGHBORS = 8
ROUTE_SAMPLE_INTERVAL_M = 20.0
MAX_ROUTE_SAMPLE_POINTS = 20_000_000
ALTITUDE_SMOOTH_WINDOW = 5
ALTITUDE_CHANGE_DEADBAND_M = 0.05

# GPS speed 只用于判定暂停，不作为预测速度口径。
PAUSE_SPEED_THRESHOLD = 0.2

# 历史窗口故意控制在少数几个，避免多尺度重复。
VEHICLE_RECENT_SHORT_K = 5
VEHICLE_RECENT_BASE_K = 10
VEHICLE_SIMILAR_TOP_K = 5
SIMILAR_TOP_K = 5
SIMILAR_WEIGHTED_TOP_K = 20
SAME_HOUR_TOLERANCE_HOURS = 2.0
SIMILAR_OD_WEIGHT_SCALE_M = 100.0
SIMILAR_RECENCY_HALF_LIFE_DAYS = 7.0
SIMILAR_TASK_MAX_OD_DISTANCE_M: Optional[float] = None

FAIL_FAST = False
SUPPRESS_PLANNER_DEBUG_OUTPUT = True
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

def active_duration_seconds(
    messages: Sequence[Tuple[datetime, Dict[str, Any]]],
    pause_speed_threshold: float = PAUSE_SPEED_THRESHOLD,
) -> Optional[float]:
    """计算扣除低速暂停区间后的任务时长。

    messages 已按 received_at 升序排列。对于相邻记录 i 和 i+1，
    使用第 i 条记录的 speed 表示区间 [time_i, time_{i+1}) 的状态：
    - speed < pause_speed_threshold：暂停，该区间不计时；
    - speed >= pause_speed_threshold：行驶，该区间计时；
    - speed 缺失或无效：不额外判定为暂停，保持原有计时口径。

    无 GPS 记录时返回 None；仅有一条记录时返回 0.0。
    """
    if not messages:
        return None

    threshold = max(0.0, float(pause_speed_threshold))
    active_seconds = 0.0

    for index in range(len(messages) - 1):
        timestamp, message = messages[index]
        next_timestamp = messages[index + 1][0]
        interval_seconds = (next_timestamp - timestamp).total_seconds()

        # 排序后仍可能存在重复时间；非正时间差不参与累计。
        if interval_seconds <= 0:
            continue

        speed = finite_float(message.get("speed"))
        if speed is not None and speed < threshold:
            continue

        active_seconds += interval_seconds

    return active_seconds

def numeric_summary(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """返回有效数值的均值、中位数、总体标准差和数量。"""
    valid: List[float] = []
    for value in values:
        number = finite_float(value)
        if number is not None:
            valid.append(number)

    if not valid:
        return None, None, None, 0

    array = np.asarray(valid, dtype=float)
    return (
        float(np.mean(array)),
        float(np.median(array)),
        float(np.std(array, ddof=0)),
        int(array.size),
    )

def weighted_numeric_mean(values: Iterable[Any], weights: Iterable[Any]) -> Optional[float]:
    """对有效数值做非负权重平均。"""
    valid_values: List[float] = []
    valid_weights: List[float] = []
    for value, weight in zip(values, weights):
        number = finite_float(value)
        weight_number = finite_float(weight)
        if number is None or weight_number is None or weight_number <= 0:
            continue
        valid_values.append(number)
        valid_weights.append(weight_number)

    if not valid_values:
        return None

    value_array = np.asarray(valid_values, dtype=float)
    weight_array = np.asarray(valid_weights, dtype=float)
    weight_sum = float(np.sum(weight_array))
    if weight_sum <= 0:
        return None
    return float(np.sum(value_array * weight_array) / weight_sum)

def circular_hour_distance(a: datetime, b: datetime) -> float:
    """返回两个时刻在 24 小时钟面上的最短小时差。"""
    hour_a = a.hour + a.minute / 60.0 + a.second / 3600.0
    hour_b = b.hour + b.minute / 60.0 + b.second / 3600.0
    diff = abs(hour_a - hour_b)
    return min(diff, 24.0 - diff)

def estimate_duration_min_from_speed(distance_m: Any, speed_mps: Any) -> Optional[float]:
    distance = finite_float(distance_m)
    speed = finite_float(speed_mps)
    if distance is None or speed is None or distance < 0 or speed <= 0:
        return None
    return float(distance / speed / 60.0)

def task_energy_intensity_pct_per_km(task: Dict[str, Any]) -> Optional[float]:
    """历史任务单位规划里程能耗：SOC下降百分点 / km。

    使用规划距离而不是当前任务真实里程，目的是让历史能耗强度与预测时
    已知的 planned_total_distance_m 保持同一距离口径。历史任务的规划距离
    在主流程中计算完成后写入 _planned_total_distance_m。
    """
    energy = finite_float(task.get("total_energy_soc_delta_pct"))
    distance_m = finite_float(task.get("_planned_total_distance_m"))
    if energy is None or distance_m is None or distance_m <= 1e-9:
        return None
    return float(energy / (distance_m / 1000.0))


def estimate_energy_pct_from_intensity(
    distance_m: Any, intensity_pct_per_km: Any
) -> Optional[float]:
    """当前规划距离 × 历史单位里程能耗强度，得到SOC消耗百分点先验。"""
    distance = finite_float(distance_m)
    intensity = finite_float(intensity_pct_per_km)
    if distance is None or intensity is None or distance < 0 or intensity < 0:
        return None
    return float((distance / 1000.0) * intensity)


def safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    a = finite_float(numerator)
    b = finite_float(denominator)
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return float(a / b)

def normalize_task(
    raw_task: Dict[str, Any],
    source_file: Path,
    source_index: int,
) -> Dict[str, Any]:
    """解析单任务并计算标签。

    统一速度 task_speed_mps 的定义：
        (end_odometer_km - start_odometer_km) * 1000 / active_duration_sec

    其中 active_duration_sec 仍通过 GPS speed < 0.2m/s 识别暂停区间。
    GPS speed 数值本身不会被平均后作为历史速度特征。
    """
    messages = sort_valid_messages(raw_task)
    start_coord, end_coord = first_last_coordinate(messages)
    start_soc, end_soc = first_last_soc(messages)
    start_odometer, end_odometer = first_last_odometer(messages)

    gps_start_time = messages[0][0] if messages else None
    gps_end_time = messages[-1][0] if messages else None

    if start_coord is None:
        lon = finite_float(raw_task.get("start_longitude"))
        lat = finite_float(raw_task.get("start_latitude"))
        if lon is not None and lat is not None:
            start_coord = (lon, lat)
    if end_coord is None:
        lon = finite_float(raw_task.get("end_longitude"))
        lat = finite_float(raw_task.get("end_latitude"))
        if lon is not None and lat is not None:
            end_coord = (lon, lat)

    active_duration_sec = active_duration_seconds(messages)
    task_duration_min = (
        active_duration_sec / 60.0 if active_duration_sec is not None else None
    )

    wall_duration_sec = None
    if gps_start_time is not None and gps_end_time is not None:
        candidate = (gps_end_time - gps_start_time).total_seconds()
        if candidate >= 0:
            wall_duration_sec = candidate

    pause_duration_sec = None
    pause_ratio = None
    if wall_duration_sec is not None and active_duration_sec is not None:
        pause_duration_sec = max(wall_duration_sec - active_duration_sec, 0.0)
        if wall_duration_sec > 0:
            pause_ratio = min(max(pause_duration_sec / wall_duration_sec, 0.0), 1.0)

    actual_distance_km = None
    if start_odometer is not None and end_odometer is not None:
        delta = end_odometer - start_odometer
        if delta >= 0:
            actual_distance_km = delta

    task_speed_mps = None
    if (
        actual_distance_km is not None
        and active_duration_sec is not None
        and active_duration_sec > 0
    ):
        task_speed_mps = actual_distance_km * 1000.0 / active_duration_sec

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
        "gps_start_odometer_km": start_odometer,
        "gps_end_odometer_km": end_odometer,
        "gps_actual_distance_km": actual_distance_km,
        "gps_duration_sec": active_duration_sec,
        "gps_wall_duration_sec": wall_duration_sec,
        "gps_pause_duration_sec": pause_duration_sec,
        "gps_pause_ratio": pause_ratio,
        "gps_task_speed_mps": task_speed_mps,
        "task_duration_min": task_duration_min,
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
# 5. 历史任务与相似任务特征（统一速度）
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
    current_planned_distance_m: Optional[float] = None,
) -> Dict[str, Any]:
    """构造精简历史特征。

    只保留一种速度：历史任务 gps_task_speed_mps。
    所有任务级历史都要求历史任务结束时间早于当前任务开始时间。
    """
    result: Dict[str, Any] = {
        "vehicle_recent5_speed_mean_mps": None,
        "vehicle_recent10_speed_mean_mps": None,
        "vehicle_recent10_speed_std_mps": None,
        "vehicle_recent10_duration_mean_sec": None,
        "vehicle_time_gap_since_prev_task_min": None,
        "vehicle_similar_top5_speed_mean_mps": None,
        "vehicle_similar_top5_duration_mean_sec": None,
        "similar_top5_speed_mean_mps": None,
        "similar_top5_speed_std_mps": None,
        "similar_top5_duration_mean_sec": None,
        "similar_top5_duration_std_sec": None,
        "similar_nearest_od_distance_m": None,
        "similar_nearest_time_gap_min": None,
        "similar_same_hour_speed_mean_mps": None,
        "similar_same_hour_duration_mean_sec": None,
        "similar_weighted_speed_mean_mps": None,
        "similar_weighted_duration_mean_sec": None,
        "estimated_duration_min_from_vehicle_similar_speed": None,
        "estimated_duration_min_from_similar_weighted_speed": None,

        # 能耗专属历史特征：SOC下降百分点及单位规划里程能耗。
        "vehicle_recent5_energy_mean_pct": None,
        "vehicle_recent10_energy_std_pct": None,
        "vehicle_recent5_energy_intensity_mean_pct_per_km": None,
        "vehicle_similar_top5_energy_mean_pct": None,
        "vehicle_similar_top5_energy_intensity_mean_pct_per_km": None,
        "similar_top5_energy_mean_pct": None,
        "similar_top5_energy_std_pct": None,
        "similar_top5_energy_intensity_mean_pct_per_km": None,
        "similar_same_hour_energy_mean_pct": None,
        "similar_weighted_energy_mean_pct": None,
        "similar_weighted_energy_intensity_mean_pct_per_km": None,
        "estimated_energy_pct_from_vehicle_similar_intensity": None,
        "estimated_energy_pct_from_similar_weighted_intensity": None,
        # 历史覆盖率；vehicle_history_task_count 同时进入时间模型。
        "vehicle_history_task_count": 0,
        "vehicle_similar_task_count": 0,
        "similar_candidate_count": 0,
        "similar_same_hour_task_count": 0,
    }

    current_start = current.get("gps_start_time")
    if current_start is None:
        return result

    history = [
        task for task in all_tasks
        if task is not current
        and task.get("gps_end_time") is not None
        and task["gps_end_time"] < current_start
    ]

    # A. 同车近期历史。
    vehicle_id = current.get("transport_device_id")
    same_vehicle = [
        task for task in history
        if vehicle_id is not None and task.get("transport_device_id") == vehicle_id
    ]
    same_vehicle.sort(key=lambda task: task["gps_end_time"], reverse=True)
    result["vehicle_history_task_count"] = len(same_vehicle)

    recent5 = same_vehicle[:VEHICLE_RECENT_SHORT_K]
    recent10 = same_vehicle[:VEHICLE_RECENT_BASE_K]
    result["vehicle_recent5_speed_mean_mps"] = numeric_summary(
        task.get("gps_task_speed_mps") for task in recent5
    )[0]
    speed10 = numeric_summary(task.get("gps_task_speed_mps") for task in recent10)
    result["vehicle_recent10_speed_mean_mps"] = speed10[0]
    result["vehicle_recent10_speed_std_mps"] = speed10[2]
    result["vehicle_recent10_duration_mean_sec"] = numeric_summary(
        task.get("gps_duration_sec") for task in recent10
    )[0]

    # 同车近期能耗：均值描述近期基线，std描述稳定性，intensity消除任务距离尺度。
    result["vehicle_recent5_energy_mean_pct"] = numeric_summary(
        task.get("total_energy_soc_delta_pct") for task in recent5
    )[0]
    result["vehicle_recent10_energy_std_pct"] = numeric_summary(
        task.get("total_energy_soc_delta_pct") for task in recent10
    )[2]
    result["vehicle_recent5_energy_intensity_mean_pct_per_km"] = numeric_summary(
        task_energy_intensity_pct_per_km(task) for task in recent5
    )[0]

    if same_vehicle:
        result["vehicle_time_gap_since_prev_task_min"] = max(
            (current_start - same_vehicle[0]["gps_end_time"]).total_seconds(), 0.0
        ) / 60.0

    # B. 对所有已完成历史任务计算 OD 相似度。
    required = (
        current.get("start_longitude"), current.get("start_latitude"),
        current.get("end_longitude"), current.get("end_latitude"),
    )
    if any(value is None for value in required):
        return result

    ranked: List[Tuple[float, float, Dict[str, Any]]] = []
    for task in history:
        if any(
            task.get(field) is None
            for field in ("start_longitude", "start_latitude", "end_longitude", "end_latitude")
        ):
            continue
        try:
            od_distance = od_pair_distance_m(current, task)
        except Exception:
            continue
        if (
            SIMILAR_TASK_MAX_OD_DISTANCE_M is not None
            and od_distance > SIMILAR_TASK_MAX_OD_DISTANCE_M
        ):
            continue
        gap_sec = max((current_start - task["gps_end_time"]).total_seconds(), 0.0)
        ranked.append((od_distance, gap_sec, task))

    ranked.sort(key=lambda item: (item[0], item[1]))
    result["similar_candidate_count"] = len(ranked)
    if not ranked:
        return result

    result["similar_nearest_od_distance_m"] = ranked[0][0]
    result["similar_nearest_time_gap_min"] = ranked[0][1] / 60.0

    top5_tasks = [item[2] for item in ranked[:SIMILAR_TOP_K]]
    speed_top5 = numeric_summary(task.get("gps_task_speed_mps") for task in top5_tasks)
    duration_top5 = numeric_summary(task.get("gps_duration_sec") for task in top5_tasks)
    result["similar_top5_speed_mean_mps"] = speed_top5[0]
    result["similar_top5_speed_std_mps"] = speed_top5[2]
    result["similar_top5_duration_mean_sec"] = duration_top5[0]
    result["similar_top5_duration_std_sec"] = duration_top5[2]

    energy_top5 = numeric_summary(
        task.get("total_energy_soc_delta_pct") for task in top5_tasks
    )
    result["similar_top5_energy_mean_pct"] = energy_top5[0]
    result["similar_top5_energy_std_pct"] = energy_top5[2]
    result["similar_top5_energy_intensity_mean_pct_per_km"] = numeric_summary(
        task_energy_intensity_pct_per_km(task) for task in top5_tasks
    )[0]

    # C. 同车 + 相似 OD。
    vehicle_ranked = [
        item for item in ranked
        if vehicle_id is not None and item[2].get("transport_device_id") == vehicle_id
    ]
    vehicle_top5 = [item[2] for item in vehicle_ranked[:VEHICLE_SIMILAR_TOP_K]]
    result["vehicle_similar_task_count"] = len(vehicle_top5)
    result["vehicle_similar_top5_speed_mean_mps"] = numeric_summary(
        task.get("gps_task_speed_mps") for task in vehicle_top5
    )[0]
    result["vehicle_similar_top5_duration_mean_sec"] = numeric_summary(
        task.get("gps_duration_sec") for task in vehicle_top5
    )[0]
    result["vehicle_similar_top5_energy_mean_pct"] = numeric_summary(
        task.get("total_energy_soc_delta_pct") for task in vehicle_top5
    )[0]
    result["vehicle_similar_top5_energy_intensity_mean_pct_per_km"] = numeric_summary(
        task_energy_intensity_pct_per_km(task) for task in vehicle_top5
    )[0]

    # D. 相似 OD + 相似时段。先看最相似 top20，再筛 ±2 小时。
    same_hour_items = [
        item for item in ranked[:SIMILAR_WEIGHTED_TOP_K]
        if item[2].get("gps_start_time") is not None
        and circular_hour_distance(current_start, item[2]["gps_start_time"])
            <= SAME_HOUR_TOLERANCE_HOURS
    ][:SIMILAR_TOP_K]
    same_hour_tasks = [item[2] for item in same_hour_items]
    result["similar_same_hour_task_count"] = len(same_hour_tasks)
    result["similar_same_hour_speed_mean_mps"] = numeric_summary(
        task.get("gps_task_speed_mps") for task in same_hour_tasks
    )[0]
    result["similar_same_hour_duration_mean_sec"] = numeric_summary(
        task.get("gps_duration_sec") for task in same_hour_tasks
    )[0]
    result["similar_same_hour_energy_mean_pct"] = numeric_summary(
        task.get("total_energy_soc_delta_pct") for task in same_hour_tasks
    )[0]

    # E. OD 相似度 × 时间新鲜度联合加权。只保留这一种加权方式，避免冗余。
    weighted_items = ranked[:SIMILAR_WEIGHTED_TOP_K]
    weights: List[float] = []
    for od_distance, gap_sec, _task in weighted_items:
        od_weight = math.exp(-od_distance / max(SIMILAR_OD_WEIGHT_SCALE_M, 1e-9))
        gap_days = gap_sec / 86400.0
        time_weight = 0.5 ** (
            gap_days / max(SIMILAR_RECENCY_HALF_LIFE_DAYS, 1e-9)
        )
        weights.append(od_weight * time_weight)

    weighted_tasks = [item[2] for item in weighted_items]
    result["similar_weighted_speed_mean_mps"] = weighted_numeric_mean(
        (task.get("gps_task_speed_mps") for task in weighted_tasks), weights
    )
    result["similar_weighted_duration_mean_sec"] = weighted_numeric_mean(
        (task.get("gps_duration_sec") for task in weighted_tasks), weights
    )
    result["similar_weighted_energy_mean_pct"] = weighted_numeric_mean(
        (task.get("total_energy_soc_delta_pct") for task in weighted_tasks), weights
    )
    result["similar_weighted_energy_intensity_mean_pct_per_km"] = weighted_numeric_mean(
        (task_energy_intensity_pct_per_km(task) for task in weighted_tasks), weights
    )

    # F. 两个强解释型组合特征：当前规划距离 / 历史统一速度。
    result["estimated_duration_min_from_vehicle_similar_speed"] = (
        estimate_duration_min_from_speed(
            current_planned_distance_m,
            result.get("vehicle_similar_top5_speed_mean_mps"),
        )
    )
    result["estimated_duration_min_from_similar_weighted_speed"] = (
        estimate_duration_min_from_speed(
            current_planned_distance_m,
            result.get("similar_weighted_speed_mean_mps"),
        )
    )

    # 能耗组合先验：当前规划距离 × 历史单位里程SOC消耗。
    result["estimated_energy_pct_from_vehicle_similar_intensity"] = (
        estimate_energy_pct_from_intensity(
            current_planned_distance_m,
            result.get("vehicle_similar_top5_energy_intensity_mean_pct_per_km"),
        )
    )
    result["estimated_energy_pct_from_similar_weighted_intensity"] = (
        estimate_energy_pct_from_intensity(
            current_planned_distance_m,
            result.get("similar_weighted_energy_intensity_mean_pct_per_km"),
        )
    )

    return result


# =============================================================================
# 6. 输出与模型特征
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
    current_start = task.get("gps_start_time")
    start_hour = None
    start_hour_sin = None
    start_hour_cos = None
    if current_start is not None:
        start_hour = (
            current_start.hour
            + current_start.minute / 60.0
            + current_start.second / 3600.0
        )
        angle = 2.0 * math.pi * start_hour / 24.0
        start_hour_sin = math.sin(angle)
        start_hour_cos = math.cos(angle)

    straight_distance = straight_line_distance(task)
    planned_distance = finite_float(route_features.get("planned_total_distance_m"))

    output = {
        "task_id": task.get("task_id"),
        "transport_device_id": task.get("transport_device_id"),
        "task_type_id": task.get("task_type_id"),
        "sn": task.get("sn"),
        "source_file": task.get("source_file"),
        "source_task_index": task.get("source_task_index"),
        "actual_start_time": current_start,
        "actual_end_time": task.get("gps_end_time"),
        "start_longitude": task.get("start_longitude"),
        "start_latitude": task.get("start_latitude"),
        "end_longitude": task.get("end_longitude"),
        "end_latitude": task.get("end_latitude"),
        "start_hour": start_hour,
        "start_hour_sin": start_hour_sin,
        "start_hour_cos": start_hour_cos,
        "straight_line_distance_m": straight_distance,
        "planned_to_straight_distance_ratio": safe_ratio(
            planned_distance, straight_distance
        ),

        # 最终监督标签只有两个；当前任务速度仅在内部用于构造未来任务的历史速度特征。
        "task_duration_min": task.get("task_duration_min"),
        "total_energy_soc_delta_pct": task.get("total_energy_soc_delta_pct"),
    }
    output.update(route_features)
    output.update(history_features)
    return output


# 时间与能耗使用不同特征集合。
# 时间模型：保持 v3 的 30 个精简特征。
TIME_FEATURE_COLUMNS = [
    # 路线 / 地形：8
    "straight_line_distance_m",
    "planned_total_distance_m",
    "planned_to_straight_distance_ratio",
    "endpoint_altitude_change_m",
    "planned_slope_mean",
    "planned_slope_std",
    "planned_cumulative_ascent_m",
    "planned_cumulative_descent_m",

    # 当前时间：2
    "start_hour_sin",
    "start_hour_cos",

    # 同车近期：6
    "vehicle_recent5_speed_mean_mps",
    "vehicle_recent10_speed_mean_mps",
    "vehicle_recent10_speed_std_mps",
    "vehicle_recent10_duration_mean_sec",
    "vehicle_time_gap_since_prev_task_min",
    "vehicle_history_task_count",

    # 同车 + 相似路线：2
    "vehicle_similar_top5_speed_mean_mps",
    "vehicle_similar_top5_duration_mean_sec",

    # 全局相似路线：10
    "similar_top5_speed_mean_mps",
    "similar_top5_speed_std_mps",
    "similar_top5_duration_mean_sec",
    "similar_top5_duration_std_sec",
    "similar_nearest_od_distance_m",
    "similar_nearest_time_gap_min",
    "similar_same_hour_speed_mean_mps",
    "similar_same_hour_duration_mean_sec",
    "similar_weighted_speed_mean_mps",
    "similar_weighted_duration_mean_sec",

    # 直接预计时长：2
    "estimated_duration_min_from_vehicle_similar_speed",
    "estimated_duration_min_from_similar_weighted_speed",
]

# 能耗模型：20 个专属特征。
# 核心原则：历史能耗预测未来能耗；历史耗时/历史速度不进入能耗模型。
# 仅保留相似 OD 距离与时间间隔作为“历史能耗统计可信度”的质量信息。
ENERGY_FEATURE_COLUMNS = [
    # 当前任务路线 / 地形负荷：5
    "planned_total_distance_m",
    "endpoint_altitude_change_m",
    "planned_slope_mean",
    "planned_cumulative_ascent_m",
    "planned_cumulative_descent_m",

    # 相似历史匹配质量：2
    "similar_nearest_od_distance_m",
    "similar_nearest_time_gap_min",

    # 历史能耗：11
    "vehicle_recent5_energy_mean_pct",
    "vehicle_recent10_energy_std_pct",
    "vehicle_recent5_energy_intensity_mean_pct_per_km",
    "vehicle_similar_top5_energy_mean_pct",
    "vehicle_similar_top5_energy_intensity_mean_pct_per_km",
    "similar_top5_energy_mean_pct",
    "similar_top5_energy_std_pct",
    "similar_top5_energy_intensity_mean_pct_per_km",
    "similar_same_hour_energy_mean_pct",
    "similar_weighted_energy_mean_pct",
    "similar_weighted_energy_intensity_mean_pct_per_km",

    # 当前距离 × 历史单位里程能耗强度：2
    "estimated_energy_pct_from_vehicle_similar_intensity",
    "estimated_energy_pct_from_similar_weighted_intensity",
]

# 兼容旧训练代码：MODEL_FEATURE_COLUMNS 继续指向时间模型特征。
MODEL_FEATURE_COLUMNS = TIME_FEATURE_COLUMNS
ALL_MODEL_FEATURE_COLUMNS = list(dict.fromkeys(TIME_FEATURE_COLUMNS + ENERGY_FEATURE_COLUMNS))


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
    altitude_provider = make_gps_altitude_estimator(
        gps_data_dir=ROUTE_SAMPLER_GPS_DATA_DIR,
        max_distance_m=GPS_ALTITUDE_MAX_DISTANCE_M,
        k=GPS_ALTITUDE_NEIGHBORS,
    )

    print("[4/5] 构造路线 + 精简历史/相似任务特征……")
    outputs: List[Dict[str, Any]] = []
    route_cache: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}
    planning_failure_count = 0

    for index, task in enumerate(tasks, start=1):
        if all(
            task.get(field) is not None
            for field in (
                "start_longitude", "start_latitude",
                "end_longitude", "end_latitude",
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

        # 让后续任务在构造历史能耗强度时可以使用该历史任务的规划距离。
        # tasks 已按开始时间排序，因此满足 history.end < current.start 的任务
        # 在被引用前已经完成路线规划并写入此字段。
        task["_planned_total_distance_m"] = route_features.get("planned_total_distance_m")

        history_features = construct_history_features(
            current=task,
            all_tasks=tasks,
            current_planned_distance_m=route_features.get("planned_total_distance_m"),
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
        json.dump(
            to_json_safe(outputs),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"输出文件：{OUTPUT_JSON}")
    print(f"输出任务数量：{len(outputs)}")
    print(f"时间模型特征数量：{len(TIME_FEATURE_COLUMNS)}")
    print(f"能耗模型特征数量：{len(ENERGY_FEATURE_COLUMNS)}")
    print(f"全部去重特征数量：{len(ALL_MODEL_FEATURE_COLUMNS)}")
    print(f"规划失败数量：{planning_failure_count}")
    print("监督目标：task_duration_min, total_energy_soc_delta_pct")
    print("历史速度仅作时间特征：gps_task_speed_mps = actual_distance / active_duration")

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