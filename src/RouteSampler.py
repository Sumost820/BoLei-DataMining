# -*- coding: utf-8 -*-
"""路线采样工具。

功能：
1. 已有规划结果 plan -> 沿规划路径生成等距采样点；
2. 起点/终点经纬度 -> 规划路径 -> 沿规划路径生成等距采样点；
3. 采样点海拔优先使用 GPS 历史点近邻估算，避免 MapResource 中部分车道 z=0 导致海拔错误；
4. 将采样点导出为 CSV 文本。
"""

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

from LanePlanner import (
    make_point,
    point_at_offset,
    polyline_length_m,
    project_point_to_polyline,
)


# =============================================================================
# GPS 海拔点读取与估算
# =============================================================================
def _load_json_records(file_path: Path) -> List[Dict[str, Any]]:
    """读取一个 JSON 文件，兼容根节点为对象或列表。"""
    with file_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        return [data]

    return []


def load_gps_altitude_points(gps_data_dir: Path) -> List[Dict[str, float]]:
    """从 data/GPSdata 中读取所有有效 GPS 高程点。"""
    gps_data_dir = Path(gps_data_dir)

    if not gps_data_dir.exists():
        raise FileNotFoundError("GPS 数据目录不存在：" + str(gps_data_dir))

    points: List[Dict[str, float]] = []

    for file_path in sorted(gps_data_dir.rglob("*.json")):
        try:
            records = _load_json_records(file_path)
        except Exception:
            continue

        for record in records:
            messages = record.get("rt_message") or []
            if not isinstance(messages, list):
                continue

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                try:
                    lon = float(msg.get("longitude"))
                    lat = float(msg.get("latitude"))
                    altitude = float(msg.get("altitude"))
                except Exception:
                    continue

                if not all(math.isfinite(v) for v in (lon, lat, altitude)):
                    continue

                # 过滤明显无效海拔。你的场景正常海拔在几百米，0 基本不是有效地形高程。
                if altitude <= 1:
                    continue

                points.append({
                    "lon": lon,
                    "lat": lat,
                    "z": altitude,
                })

    if not points:
        raise RuntimeError("没有从 GPS 数据中读取到有效 altitude 点。")

    return points


def _lonlat_to_local_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    lon0: float,
    lat0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """经纬度转局部米制坐标，用于距离计算。"""
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    x = (lon - lon0) * meters_per_deg_lon
    y = (lat - lat0) * meters_per_deg_lat
    return x, y


def make_gps_altitude_estimator(
    gps_data_dir: Path,
    max_distance_m: float = 60.0,
    k: int = 8,
) -> Callable[[float, float, Optional[float]], Dict[str, Any]]:
    """
    构造 GPS 海拔估算器。

    返回的 estimator(lon, lat, fallback_altitude) 会返回：
    {
        "altitude_m": ...,
        "altitude_source": "gps_idw" / "lane_z_fallback" / "none",
        "altitude_nearest_distance_m": ...
    }

    逻辑：
    - 优先使用 GPS 历史点；
    - 查询点附近 max_distance_m 内，用最近 k 个点做反距离加权；
    - 附近没有 GPS 点时，才使用车道中心线 z 兜底；
    - 如果车道 z 也是 0 或无效，则返回 None。
    """
    raw_points = load_gps_altitude_points(Path(gps_data_dir))

    lon = np.array([p["lon"] for p in raw_points], dtype=float)
    lat = np.array([p["lat"] for p in raw_points], dtype=float)
    z = np.array([p["z"] for p in raw_points], dtype=float)

    lon0 = float(np.mean(lon))
    lat0 = float(np.mean(lat))
    x, y = _lonlat_to_local_xy(lon, lat, lon0, lat0)
    xy = np.column_stack([x, y])

    tree = cKDTree(xy) if cKDTree is not None else None

    def estimate(
        query_lon: float,
        query_lat: float,
        fallback_altitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        qlon = np.array([float(query_lon)], dtype=float)
        qlat = np.array([float(query_lat)], dtype=float)
        qx, qy = _lonlat_to_local_xy(qlon, qlat, lon0, lat0)
        q = np.array([float(qx[0]), float(qy[0])], dtype=float)

        kk = max(1, min(int(k), len(z)))

        if tree is not None:
            dists, idx = tree.query(q, k=kk)
            dists = np.atleast_1d(dists).astype(float)
            idx = np.atleast_1d(idx).astype(int)
        else:
            all_dists = np.sqrt(np.sum((xy - q) ** 2, axis=1))
            idx = np.argsort(all_dists)[:kk]
            dists = all_dists[idx]

        nearest_distance = float(dists[0])
        use = dists <= float(max_distance_m)

        if np.any(use):
            used_dists = dists[use]
            used_z = z[idx[use]]

            if used_dists[0] < 0.2:
                altitude = float(used_z[0])
            else:
                weights = 1.0 / np.maximum(used_dists, 0.5) ** 2
                altitude = float(np.sum(weights * used_z) / np.sum(weights))

            return {
                "altitude_m": altitude,
                "altitude_source": "gps_idw",
                "altitude_nearest_distance_m": nearest_distance,
            }

        # 附近没有 GPS 点，才使用车道 z 兜底。
        if fallback_altitude is not None:
            try:
                fallback = float(fallback_altitude)
                if math.isfinite(fallback) and fallback > 1:
                    return {
                        "altitude_m": fallback,
                        "altitude_source": "lane_z_fallback",
                        "altitude_nearest_distance_m": nearest_distance,
                    }
            except Exception:
                pass

        return {
            "altitude_m": None,
            "altitude_source": "none",
            "altitude_nearest_distance_m": nearest_distance,
        }

    return estimate


# =============================================================================
# 规划路径采样
# =============================================================================
def _visible_segments(planner, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把规划路径拆成按行驶顺序排列的可见车道段。"""
    segments: List[Dict[str, Any]] = []
    path = list(plan["path"])

    for index, uid in enumerate(path):
        lane = planner.lanes[uid]
        points = planner.visible_lane_points(
            path,
            uid,
            index,
            plan["origin_projection"],
            plan["destination_projection"],
        )

        if not points:
            continue

        length_m = float(polyline_length_m(points))

        if length_m <= 0 and len(points) < 2:
            continue

        segments.append({
            "lane_order": index + 1,
            "lane_uid": uid,
            "lane_name": lane.get("name"),
            "points": points,
            "length_m": length_m,
        })

    if not segments:
        raise RuntimeError("规划路径为空，无法生成路径采样点。")

    return segments


def _build_whole_polyline(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把多个可见车道段拼成一条完整折线，用于起点投影。"""
    whole: List[Dict[str, Any]] = []

    for seg in segments:
        for point in seg["points"]:
            if (
                whole
                and abs(whole[-1]["lon"] - point["lon"]) < 1e-12
                and abs(whole[-1]["lat"] - point["lat"]) < 1e-12
            ):
                continue
            whole.append(point)

    if not whole:
        raise RuntimeError("规划路径没有有效坐标点。")

    return whole


def _locate_segment(
    segments: List[Dict[str, Any]],
    route_offset_m: float,
) -> Tuple[Dict[str, Any], float]:
    """根据整条路径 offset 找到所在车道段，以及该段内 offset。"""
    walked = 0.0

    for seg in segments:
        next_walked = walked + seg["length_m"]
        if route_offset_m <= next_walked + 1e-6:
            local_offset = max(0.0, route_offset_m - walked)
            return seg, local_offset
        walked = next_walked

    last = segments[-1]
    return last, last["length_m"]


def _sample_offsets(
    total_len_m: float,
    start_offset_m: float,
    interval_m: float,
    include_end: bool,
    max_records: int,
) -> List[float]:
    """生成采样 offset 列表。"""
    if interval_m <= 0:
        raise ValueError("interval_m 必须大于 0。")
    if max_records < 1:
        raise ValueError("max_records 必须大于等于 1。")

    total_len_m = float(total_len_m)
    start_offset_m = max(0.0, min(float(start_offset_m), total_len_m))

    offsets: List[float] = []
    current = start_offset_m

    while current <= total_len_m + 1e-6:
        offsets.append(min(current, total_len_m))
        if len(offsets) >= max_records:
            break
        current += interval_m

    if (
        include_end
        and offsets
        and offsets[-1] < total_len_m - 1e-6
        and len(offsets) < max_records
    ):
        offsets.append(total_len_m)

    if not offsets:
        offsets.append(start_offset_m)

    return offsets


def sample_route_records(
    planner,
    plan: Dict[str, Any],
    start_lon: Optional[float] = None,
    start_lat: Optional[float] = None,
    interval_m: float = 5.0,
    include_end: bool = True,
    max_records: int = 20000,
    altitude_provider: Optional[Callable[[float, float, Optional[float]], Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    沿已有规划路径按固定距离生成采样点。

    用途：
    - SN + task_id 模式下，RouteWeb.py 已经拿到了 plan；
    - 这个函数直接对 plan 采样。

    altitude_provider:
    - 如果提供，采样点海拔会优先用它估算；
    - 如果不提供，则使用车道中心线 z。
    """
    segments = _visible_segments(planner, plan)
    total_len_m = float(sum(seg["length_m"] for seg in segments))

    start_projection = None
    start_offset_m = 0.0

    if start_lon is not None or start_lat is not None:
        if start_lon is None or start_lat is None:
            raise ValueError("start_lon 和 start_lat 必须同时提供。")

        whole_polyline = _build_whole_polyline(segments)
        start_point = make_point(float(start_lon), float(start_lat))
        start_projection = project_point_to_polyline(start_point, whole_polyline)
        start_offset_m = max(
            0.0,
            min(float(start_projection["offset_m"]), total_len_m),
        )

    offsets = _sample_offsets(
        total_len_m=total_len_m,
        start_offset_m=start_offset_m,
        interval_m=float(interval_m),
        include_end=bool(include_end),
        max_records=int(max_records),
    )

    records: List[Dict[str, Any]] = []

    for sample_index, route_offset_m in enumerate(offsets):
        seg, local_offset_m = _locate_segment(segments, route_offset_m)
        point = point_at_offset(seg["points"], local_offset_m)

        distance_from_start_m = route_offset_m - start_offset_m

        lane_altitude = None
        if point.get("z") is not None:
            try:
                lane_altitude = float(point["z"])
            except Exception:
                lane_altitude = None

        altitude_m = lane_altitude
        altitude_source = "lane_z"
        altitude_nearest_distance_m = None

        if altitude_provider is not None:
            alt_info = altitude_provider(
                float(point["lon"]),
                float(point["lat"]),
                lane_altitude,
            )
            altitude_m = alt_info.get("altitude_m")
            altitude_source = alt_info.get("altitude_source")
            altitude_nearest_distance_m = alt_info.get("altitude_nearest_distance_m")

        records.append({
            "sample_index": sample_index,
            "distance_m": round(float(distance_from_start_m), 3),
            "distance_from_start_m": round(float(distance_from_start_m), 3),
            "route_offset_m": round(float(route_offset_m), 3),
            "lon": round(float(point["lon"]), 10),
            "lat": round(float(point["lat"]), 10),
            "altitude_m": (
                None
                if altitude_m is None
                else round(float(altitude_m), 3)
            ),
            "altitude_source": altitude_source,
            "altitude_nearest_distance_m": (
                None
                if altitude_nearest_distance_m is None
                else round(float(altitude_nearest_distance_m), 3)
            ),
            "lane_uid": seg["lane_uid"],
            "lane_name": seg.get("lane_name"),
            "lane_order": seg["lane_order"],
        })

    meta = {
        "sample_count": len(records),
        "interval_m": float(interval_m),
        "route_length_m": round(float(total_len_m), 3),
        "start_route_offset_m": round(float(start_offset_m), 3),
        "remaining_length_m": round(float(max(0.0, total_len_m - start_offset_m)), 3),
        "start_snap_distance_m": None,
        "start_projected_lon": None,
        "start_projected_lat": None,
    }

    if start_projection is not None:
        meta.update({
            "start_snap_distance_m": round(float(start_projection["snap_dist_m"]), 3),
            "start_projected_lon": round(float(start_projection["projected_lon"]), 10),
            "start_projected_lat": round(float(start_projection["projected_lat"]), 10),
        })

    return records, meta


def plan_and_sample_by_coords(
    planner,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    interval_m: float = 5.0,
    include_end: bool = True,
    max_points: int = 20000,
    altitude_provider: Optional[Callable[[float, float, Optional[float]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    根据起点/终点经纬度规划路径，并输出路径采样点。

    不需要 SN / task_id。
    不需要 RouteCoordinateSampler.py。
    """
    origin = make_point(float(start_lon), float(start_lat))
    destination = make_point(float(end_lon), float(end_lat))

    plan, tried = planner.plan(origin, destination)

    records, sample_meta = sample_route_records(
        planner=planner,
        plan=plan,
        start_lon=None,
        start_lat=None,
        interval_m=float(interval_m),
        include_end=bool(include_end),
        max_records=int(max_points),
        altitude_provider=altitude_provider,
    )

    metrics = plan.get("metrics") or {}
    origin_projection = plan.get("origin_projection") or {}
    destination_projection = plan.get("destination_projection") or {}

    summary = {
        "route_length_m": sample_meta["route_length_m"],
        "point_count": len(records),
        "sample_count": len(records),
        "interval_m": float(interval_m),

        "start_lon": float(start_lon),
        "start_lat": float(start_lat),
        "end_lon": float(end_lon),
        "end_lat": float(end_lat),

        "start_lane_uid": plan.get("start_uid"),
        "end_lane_uid": plan.get("end_uid"),
        "planned_lane_count": len(plan.get("path", [])),
        "planned_lane_uids": list(plan.get("path", [])),
        "tried_count": plan.get("tried_count", len(tried) if tried is not None else None),

        "planned_distance_m": metrics.get("total_distance_m"),
        "objective_m": metrics.get("objective_m"),
        "snap_distance_m": metrics.get("snap_distance_m"),
        "jump_distance_m": metrics.get("jump_distance_m"),

        "start_snap_distance_m": origin_projection.get("snap_dist_m"),
        "start_projected_lon": origin_projection.get("projected_lon"),
        "start_projected_lat": origin_projection.get("projected_lat"),

        "end_snap_distance_m": destination_projection.get("snap_dist_m"),
        "end_projected_lon": destination_projection.get("projected_lon"),
        "end_projected_lat": destination_projection.get("projected_lat"),
    }

    return {
        "points": records,
        "summary": summary,
        "plan": plan,
        "origin": origin,
        "destination": destination,
        "sample_meta": sample_meta,
    }


def records_to_csv_text(records: List[Dict[str, Any]]) -> str:
    """将采样点记录转换为 CSV 文本。"""
    if not records:
        return ""

    preferred_fields = [
        "sample_index",
        "distance_m",
        "distance_from_start_m",
        "route_offset_m",
        "lon",
        "lat",
        "altitude_m",
        "altitude_source",
        "altitude_nearest_distance_m",
        "lane_uid",
        "lane_name",
        "lane_order",
    ]

    fieldnames: List[str] = []

    for name in preferred_fields:
        if any(name in row for row in records):
            fieldnames.append(name)

    for row in records:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue()