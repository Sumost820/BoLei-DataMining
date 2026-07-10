# -*- coding: utf-8 -*-
"""
GPS 高程查询/可视化工具

默认读取项目根目录 data/GPSdata，输出到项目根目录 target。
重点用途：给定一个经纬度点，在已有 GPS 轨迹点附近估算海拔。

示例：
    python src/HeightVisualizer.py
    python src/HeightVisualizer.py --input data/GPSdata --output target
    python src/HeightVisualizer.py --query-lon 89.28031 --query-lat 44.84136
    python src/HeightVisualizer.py --query-lon 89.28031 --query-lat 44.84136 --max-distance 50
    python src/HeightVisualizer.py --make-contour
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


# ===== 路径配置 =====
SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "src" else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_ROOT / "data" / "GPSdata"
DEFAULT_OUTPUT = PROJECT_ROOT / "target"

# ===== 参数 =====
BIN_SIZE_M = 2.0               # 合并重复/漂移点的网格大小
QUERY_K = 8                   # 查询海拔时使用最近的 k 个点
DEFAULT_MAX_DISTANCE_M = 60.0  # 查询点距离轨迹超过此值时给出警告/拒绝
GRID_RESOLUTION_M = 5.0        # 可选等高线网格分辨率
TRACK_BUFFER_M = 25.0          # 可选等高线只显示轨迹附近范围
CONTOUR_LEVELS = 18


def setup_matplotlib() -> None:
    candidates = [
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            font_prop = mpl.font_manager.FontProperties(fname=font_path)
            mpl.rcParams["font.family"] = font_prop.get_name()
            break
    else:
        mpl.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False


def resolve_project_path(value: str, default_path: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default_path.resolve()
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    for candidate in (Path.cwd() / p, PROJECT_ROOT / p, SCRIPT_DIR / p):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / p).resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def ensure_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"不支持的 JSON 根类型: {type(data)}")


def iter_json_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
    elif input_path.is_dir():
        yield from sorted(input_path.glob("*.json"))
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")


def extract_gps_points(input_path: Path) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    json_files = list(iter_json_files(input_path))
    if not json_files:
        raise RuntimeError(f"没有找到 JSON 文件: {input_path}")

    print(f"发现 {len(json_files)} 个 GPS JSON 文件")
    for file in json_files:
        print(f"处理: {file.name}")
        try:
            data = ensure_list(load_json(file))
            for task_index, task in enumerate(data):
                task_id = task.get("task_id", task_index)
                messages = task.get("rt_message") or []
                for seq, msg in enumerate(messages):
                    try:
                        lon = float(msg.get("longitude"))
                        lat = float(msg.get("latitude"))
                        z = float(msg.get("altitude"))
                        if not all(math.isfinite(v) for v in (lon, lat, z)):
                            continue
                        points.append({
                            "lon": lon,
                            "lat": lat,
                            "z": z,
                            "task_id": task_id,
                            "seq": seq,
                            "time": msg.get("received_at"),
                            "odometer": msg.get("odometer"),
                        })
                    except (TypeError, ValueError):
                        continue
        except Exception as exc:
            print(f"  跳过 {file.name}: {exc}")

    if not points:
        raise RuntimeError("没有找到有效的 longitude / latitude / altitude 数据")
    return points


def lonlat_to_local_meters(lon: np.ndarray, lat: np.ndarray, lon0: float | None = None, lat0: float | None = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if lon0 is None:
        lon0 = float(np.mean(lon))
    if lat0 is None:
        lat0 = float(np.mean(lat))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    x = (lon - lon0) * meters_per_deg_lon
    y = (lat - lat0) * meters_per_deg_lat
    return x, y, lon0, lat0


def prepare_arrays(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    lon = np.array([p["lon"] for p in points], dtype=float)
    lat = np.array([p["lat"] for p in points], dtype=float)
    z = np.array([p["z"] for p in points], dtype=float)
    task_ids = np.array([str(p.get("task_id", "")) for p in points])
    seq = np.array([int(p.get("seq", 0)) for p in points])
    x, y, lon0, lat0 = lonlat_to_local_meters(lon, lat)

    # 只在同一个 task 内计算连续距离，不跨任务连线。
    dist = np.zeros_like(x)
    total = 0.0
    for tid in np.unique(task_ids):
        idx = np.where(task_ids == tid)[0]
        idx = idx[np.argsort(seq[idx])]
        if len(idx) == 0:
            continue
        steps = np.hypot(np.diff(x[idx], prepend=x[idx][0]), np.diff(y[idx], prepend=y[idx][0]))
        d = np.cumsum(steps)
        dist[idx] = total + d
        total += float(d[-1])

    return {"lon": lon, "lat": lat, "x": x, "y": y, "z": z, "dist": dist, "task_id": task_ids,
            "seq": seq, "lon0": lon0, "lat0": lat0}


def bin_points(arr: Dict[str, Any], bin_size_m: float = BIN_SIZE_M) -> Dict[str, np.ndarray]:
    """合并同一小网格中的 GPS 点，减少静止漂移和重复采样。"""
    x, y, z = arr["x"], arr["y"], arr["z"]
    bx = np.round(x / bin_size_m).astype(int)
    by = np.round(y / bin_size_m).astype(int)
    buckets: Dict[Tuple[int, int], List[Tuple[float, float, float, float, float]]] = {}
    for xi, yi, zi, loni, lati, bxi, byi in zip(x, y, z, arr["lon"], arr["lat"], bx, by):
        buckets.setdefault((int(bxi), int(byi)), []).append((float(xi), float(yi), float(zi), float(loni), float(lati)))
    merged = np.array([np.mean(vals, axis=0) for vals in buckets.values()], dtype=float)
    return {"x": merged[:, 0], "y": merged[:, 1], "z": merged[:, 2], "lon": merged[:, 3], "lat": merged[:, 4]}



def set_lonlat_axes(ax) -> None:
    """把图像坐标轴格式化为经纬度小数显示。"""
    ax.ticklabel_format(style="plain", useOffset=False, axis="both")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")


def estimate_altitude(arr: Dict[str, Any], query_lon: float, query_lat: float,
                      max_distance_m: float = DEFAULT_MAX_DISTANCE_M, k: int = QUERY_K) -> Dict[str, float]:
    """
    在已有轨迹点附近估算海拔。
    方法：先合并近重复点，再用最近 k 点的反距离加权平均。查询点离最近轨迹点越远，结果越不可靠。
    """
    merged = bin_points(arr)
    qx, qy, _, _ = lonlat_to_local_meters(
        np.array([query_lon], dtype=float),
        np.array([query_lat], dtype=float),
        lon0=float(arr["lon0"]),
        lat0=float(arr["lat0"]),
    )

    tree = cKDTree(np.column_stack([merged["x"], merged["y"]]))
    k = min(k, len(merged["x"]))
    dists, idx = tree.query([float(qx[0]), float(qy[0])], k=k)
    dists = np.atleast_1d(dists).astype(float)
    idx = np.atleast_1d(idx).astype(int)

    # 如果几乎命中轨迹点，直接返回该点海拔。
    if dists[0] < 0.2:
        altitude = float(merged["z"][idx[0]])
    else:
        # 只使用 max_distance_m 范围内的近邻；如果没有，则仍报告最近点，但标记 unreliable。
        use = dists <= max_distance_m
        if not np.any(use):
            altitude = float(merged["z"][idx[0]])
        else:
            d = dists[use]
            z = merged["z"][idx[use]]
            weights = 1.0 / np.maximum(d, 0.5) ** 2
            altitude = float(np.sum(weights * z) / np.sum(weights))

    nearest = int(idx[0])
    return {
        "query_lon": float(query_lon),
        "query_lat": float(query_lat),
        "estimated_altitude_m": altitude,
        "nearest_distance_m": float(dists[0]),
        "nearest_lon": float(merged["lon"][nearest]),
        "nearest_lat": float(merged["lat"][nearest]),
        "nearest_altitude_m": float(merged["z"][nearest]),
        "reliable": bool(dists[0] <= max_distance_m),
    }

def save_track_points_map(arr: Dict[str, Any], output_dir: Path) -> None:
    """只画散点，不连线；输出图片的 x/y 轴使用经纬度，但不强制等比例，避免图像过扁。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    sc = ax.scatter(
        arr["lon"],
        arr["lat"],
        c=arr["z"],
        s=3,
        cmap="terrain",
        linewidths=0,
        alpha=0.9,
    )

    # 关键修改：
    # 不要用 ax.set_aspect("equal")，否则经纬度跨度差异会把图压扁。
    ax.set_aspect("auto")

    set_lonlat_axes(ax)
    ax.grid(True, linestyle=":", alpha=0.3)

    # 给边界留一点空白
    lon_min, lon_max = float(np.min(arr["lon"])), float(np.max(arr["lon"]))
    lat_min, lat_max = float(np.min(arr["lat"])), float(np.max(arr["lat"]))

    lon_pad = max((lon_max - lon_min) * 0.04, 1e-6)
    lat_pad = max((lat_max - lat_min) * 0.08, 1e-6)

    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    cbar = fig.colorbar(sc, ax=ax, label="高程 / m", shrink=0.82, aspect=24)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "GPS采样点高程图.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_optional_contour(arr: Dict[str, Any], output_dir: Path) -> None:
    """可选：只画轨迹附近插值面，不叠加任何轨迹连线；输出图片坐标轴使用经纬度。"""
    merged = bin_points(arr)
    x, y, z = merged["x"], merged["y"], merged["z"]
    gx = np.arange(x.min() - TRACK_BUFFER_M, x.max() + TRACK_BUFFER_M + GRID_RESOLUTION_M, GRID_RESOLUTION_M)
    gy = np.arange(y.min() - TRACK_BUFFER_M, y.max() + TRACK_BUFFER_M + GRID_RESOLUTION_M, GRID_RESOLUTION_M)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

    tree = cKDTree(np.column_stack([x, y]))
    nearest_dist, _ = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)
    nearest_dist = nearest_dist.reshape(grid_x.shape)
    near_track = nearest_dist <= TRACK_BUFFER_M
    missing_near = np.isnan(grid_z) & near_track
    if np.any(missing_near):
        grid_z[missing_near] = griddata((x, y), z, (grid_x[missing_near], grid_y[missing_near]), method="nearest")
    grid_z[~near_track] = np.nan

    if np.all(np.isnan(grid_z)):
        print("等高线图跳过：有效插值区域为空")
        return

    # 插值仍在米制局部坐标中完成；绘图前再把网格转回经纬度，避免经纬度直接插值造成尺度失真。
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(float(arr["lat0"])))
    grid_lon = grid_x / meters_per_deg_lon + float(arr["lon0"])
    grid_lat = grid_y / meters_per_deg_lat + float(arr["lat0"])

    levels = np.linspace(float(np.nanmin(grid_z)), float(np.nanmax(grid_z)), CONTOUR_LEVELS)
    fig, ax = plt.subplots(figsize=(12, 8))
    cf = ax.contourf(grid_lon, grid_lat, grid_z, levels=levels, cmap="terrain", alpha=0.9)
    # 不画 ax.contour 线，也不画轨迹折线；只叠加很小的采样点。
    ax.scatter(arr["lon"], arr["lat"], s=1, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"轨迹附近高程插值图（仅 {TRACK_BUFFER_M:.0f} m 内，无连线）")
    set_lonlat_axes(ax)
    ax.grid(True, linestyle=":", alpha=0.25)
    fig.colorbar(cf, ax=ax, label="高程 / m")
    fig.tight_layout()
    fig.savefig(output_dir / "轨迹附近高程插值图_无连线.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def write_query_result(result: Dict[str, float], output_dir: Path) -> None:
    out_csv = output_dir / "查询点海拔结果.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    print("\n查询点海拔估计：")
    print(f"  输入坐标: lon={result['query_lon']:.8f}, lat={result['query_lat']:.8f}")
    print(f"  估算海拔: {result['estimated_altitude_m']:.2f} m")
    print(f"  最近轨迹点距离: {result['nearest_distance_m']:.2f} m")
    print(f"  最近轨迹点: lon={result['nearest_lon']:.8f}, lat={result['nearest_lat']:.8f}, alt={result['nearest_altitude_m']:.2f} m")
    if not result["reliable"]:
        print("  警告: 查询点距离已有轨迹较远，海拔估计不可靠；建议只查询轨迹附近点。")
    print(f"  已保存: {out_csv}")


def write_report(points: List[Dict[str, Any]], arr: Dict[str, Any], output_dir: Path) -> None:
    report = f"""GPS 高程处理报告
========================================
输入点数: {len(points):,}
高程范围: {arr['z'].min():.2f} ~ {arr['z'].max():.2f} m
平均高程: {arr['z'].mean():.2f} m
东西跨度: {np.ptp(arr['x']):.1f} m
南北跨度: {np.ptp(arr['y']):.1f} m

说明：
- 默认图只画 GPS 采样点，不再把不同任务/跳点强行连成线。
- 查询海拔使用轨迹附近点的反距离加权，不依赖整幅二维等高线图。
- 如果查询点离最近轨迹点超过 max-distance，结果会标记为不可靠。
========================================
"""
    (output_dir / "GPS高程处理报告.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPS 高程可视化和轨迹附近海拔查询")
    parser.add_argument("--input", "-i", default="", help=f"GPS JSON 文件或目录；默认: {DEFAULT_INPUT}")
    parser.add_argument("--output", "-o", default="", help=f"输出目录；默认: {DEFAULT_OUTPUT}")
    parser.add_argument("--query-lon", type=float, default=None, help="要查询海拔的经度")
    parser.add_argument("--query-lat", type=float, default=None, help="要查询海拔的纬度")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE_M, help="查询点允许距离最近轨迹点的最大距离，单位 m")
    parser.add_argument("--make-contour", action="store_true", help="可选生成轨迹附近插值图；默认不生成")
    args = parser.parse_args()

    setup_matplotlib()
    input_path = resolve_project_path(args.input, DEFAULT_INPUT)
    output_dir = resolve_project_path(args.output, DEFAULT_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = extract_gps_points(input_path)
    arr = prepare_arrays(points)

    save_track_points_map(arr, output_dir)
    if args.make_contour:
        save_optional_contour(arr, output_dir)

    if args.query_lon is not None or args.query_lat is not None:
        if args.query_lon is None or args.query_lat is None:
            raise ValueError("--query-lon 和 --query-lat 必须同时提供")
        result = estimate_altitude(arr, args.query_lon, args.query_lat, args.max_distance)
        write_query_result(result, output_dir)

    write_report(points, arr, output_dir)
    print(f"处理完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
