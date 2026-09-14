import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import folium


# ============================================================
# 配置：通常只需要修改这里
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TARGET_AREA = "TianChi"  # "JiangYi" or "TianChi"
SN = "TLE00900VR1450048"
DATA = "data3"

TASK_FILE = ROOT_DIR / "data" / TARGET_AREA / "CheckedData" / f"{SN}_{DATA}有效卸货记录.json"
RESOURCE_FILE = ROOT_DIR / "data" / TARGET_AREA / "MapResource.json"
OUTPUT_HTML = ROOT_DIR / "data" / TARGET_AREA / "CheckedData" / f"{SN}_{DATA}矿车路线地图.html"

# 抽稀参数：数值越大越精细，HTML 越大
MAX_POINTS_PER_TASK = 1200
MAX_POINTS_PER_LANE = 300
# 只绘制路线附近的资源要素；0.01 度约 1km 左右，视纬度而定
RESOURCE_BBOX_MARGIN = 0.01

# 是否默认显示所有任务路线；False 时打开地图后任务默认隐藏，需要手动勾选
SHOW_TASKS_BY_DEFAULT = False   


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def ensure_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f'Unsupported JSON root type: {type(data)}')


def downsample(points: List[Tuple[float, float]], max_points: int) -> List[Tuple[float, float]]:
    """按固定步长抽稀点，保留首尾。points 为 [(lat, lon), ...]。"""
    if len(points) <= max_points:
        return points
    step = math.ceil(len(points) / max_points)
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def get_task_routes(task_file: Path):
    tasks = ensure_list(load_json(task_file))
    routes = []

    for task in tasks:
        task_id = task.get('task_id', 'unknown')
        messages = task.get('rt_message') or []
        messages = sorted(messages, key=lambda m: m.get('received_at', ''))

        points = []
        speeds = []
        for m in messages:
            lon = m.get('longitude')
            lat = m.get('latitude')
            if lon is None or lat is None:
                continue
            points.append((float(lat), float(lon)))
            if m.get('speed') is not None:
                speeds.append(float(m['speed']))

        if len(points) >= 2:
            routes.append({
                'task_id': task_id,
                'start_time': task.get('actual_start_time', ''),
                'end_time': task.get('actual_end_time', ''),
                'raw_count': len(points),
                'max_speed': max(speeds) if speeds else None,
                'points': downsample(points, MAX_POINTS_PER_TASK),
            })

    return routes


def bbox_from_points(point_groups: Iterable[List[Tuple[float, float]]]):
    lats, lons = [], []
    for pts in point_groups:
        for lat, lon in pts:
            lats.append(lat)
            lons.append(lon)
    if not lats:
        raise ValueError('No points found')
    return min(lats), min(lons), max(lats), max(lons)  # south, west, north, east


def point_in_bbox(lat: float, lon: float, bbox) -> bool:
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def any_point_in_bbox(points: List[Tuple[float, float]], bbox) -> bool:
    return any(point_in_bbox(lat, lon, bbox) for lat, lon in points)


def parse_point_list(point_list: Any) -> List[Tuple[float, float]]:
    """资源文件里的点通常是 {'x': lon, 'y': lat, 'z': alt}。返回 folium 需要的 [(lat, lon), ...]。"""
    out = []
    if not isinstance(point_list, list):
        return out
    for p in point_list:
        if not isinstance(p, dict):
            continue
        x, y = p.get('x'), p.get('y')
        if x is None or y is None:
            continue
        out.append((float(y), float(x)))
    return out


def parse_resource_features(resource_file: Path, filter_bbox):
    resources = ensure_list(load_json(resource_file))
    polygons = []
    lanes = []

    for item in resources:
        raw = item.get('数据')
        if not raw:
            continue
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        name = item.get('资源名称', '')
        rtype = item.get('资源类型', '')
        rid = item.get('资源类型ID', '')
        tooltip = f'{name} | {rtype} | 类型ID={rid}'

        # 区域、多边形类资源
        poly_points = parse_point_list((data.get('polygon') or {}).get('point'))
        if len(poly_points) >= 3 and any_point_in_bbox(poly_points, filter_bbox):
            polygons.append({
                'name': name,
                'type': rtype,
                'type_id': rid,
                'tooltip': tooltip,
                'points': poly_points,
            })

        # 车道/路径中心线类资源
        central_curve = data.get('central_curve') or {}
        segments = central_curve.get('segment') or []
        line_points = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            line_seg = seg.get('line_segment') or {}
            pts = parse_point_list(line_seg.get('point'))
            if pts:
                if line_points and line_points[-1] == pts[0]:
                    line_points.extend(pts[1:])
                else:
                    line_points.extend(pts)

        if len(line_points) >= 2 and any_point_in_bbox(line_points, filter_bbox):
            lanes.append({
                'name': name,
                'type': rtype,
                'type_id': rid,
                'tooltip': tooltip,
                'raw_count': len(line_points),
                'points': downsample(line_points, MAX_POINTS_PER_LANE),
            })

    return polygons, lanes


def add_osm_standard_tile_layer(m: folium.Map):
    """
    添加与截图类似的 OpenStreetMap Standard/Mapnik 样式底图。

    关键点：overlay=False，作为底图 base layer 使用；不要把它当 overlay 叠加，
    否则和其它底图混在一起时会看起来“瓦片不对”。
    """
    folium.TileLayer(
        tiles='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='© OpenStreetMap contributors',
        name='OpenStreetMap Standard',
        overlay=False,
        control=True,
        show=True,
        max_zoom=19,
        min_zoom=0,
    ).add_to(m)


def build_map(routes, polygons, lanes, output_html: Path):
    all_route_points = [r['points'] for r in routes]
    south, west, north, east = bbox_from_points(all_route_points)
    center = [(south + north) / 2, (west + east) / 2]

    # tiles=None：手动添加 OSM Standard，避免 Folium 默认图层不清楚
    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    add_osm_standard_tile_layer(m)

    # 白色兜底背景：瓦片加载慢或失败时，路线/区域仍可见
    css = """
    <style>
      .leaflet-container { background: #f4f1e8; }
      .map-note {
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #999;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 13px;
        line-height: 1.4;
        max-width: 420px;
      }
      .leaflet-control-layers-expanded {
        max-height: 70vh;
        overflow-y: auto;
        min-width: 280px;
      }
    </style>
    """
    m.get_root().html.add_child(folium.Element(css)) # type: ignore

    fg_lanes = folium.FeatureGroup(name=f'资源车道/路径中心线（{len(lanes)}条）', show=False)
    fg_polygons = folium.FeatureGroup(name=f'资源区域/多边形（{len(polygons)}个）', show=True)

    # 资源中心线
    for lane in lanes:
        folium.PolyLine(
            locations=lane['points'],
            weight=2,
            opacity=0.35,
            color='#555555',
            tooltip=lane['tooltip'],
        ).add_to(fg_lanes)

    # 资源区域
    color_by_type = {
        'HD换电站': '#1f77b4',
        'HD装料区': '#2ca02c',
        'HD装料位': '#2ca02c',
        'HD卸料位': '#ff7f0e',
        'HD排土场': '#9467bd',
        'HD均衡碾压区': '#d62728',
        'HD限速区': '#8c564b',
        'HD上下坡区': '#e377c2',
        'HD停车位': '#17becf',
    }
    for poly in polygons:
        color = color_by_type.get(poly['type'], '#3388ff')
        folium.Polygon(
            locations=poly['points'],
            color=color,
            weight=2,
            fill=True,
            fill_opacity=0.18,
            tooltip=poly['tooltip'],
        ).add_to(fg_polygons)

    # 任务路线：每个 task 单独建一个 FeatureGroup。
    # 这样右上角图层控制器中会出现每个 task 的复选框，
    # 用户可以逐个勾选/取消显示。起点、终点也放在同一个 task 图层里。
    palette = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#a65628', '#f781bf', '#666666', '#66c2a5', '#fc8d62',
        '#8da0cb', '#e78ac3', '#a6d854', '#d9a300', '#b3b3b3',
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'
    ]
    for idx, route in enumerate(routes):
        color = palette[idx % len(palette)]
        max_speed_text = '' if route['max_speed'] is None else f"<br>最大速度：{route['max_speed']}"
        tooltip = (
            f"任务 {route['task_id']}<br>"
            f"开始：{route['start_time']}<br>"
            f"结束：{route['end_time']}<br>"
            f"轨迹点：{route['raw_count']}，显示：{len(route['points'])}"
            f"{max_speed_text}"
        )

        task_layer_name = f"Task {idx + 1}: {route['task_id']} | {route['start_time']}"
        fg_task = folium.FeatureGroup(
            name=task_layer_name,
            show=SHOW_TASKS_BY_DEFAULT,
            overlay=True,
            control=True,
        )

        folium.PolyLine(
            locations=route['points'],
            color=color,
            weight=4,
            opacity=0.9,
            tooltip=tooltip,
        ).add_to(fg_task)

        start = route['points'][0]
        end = route['points'][-1]
        folium.CircleMarker(
            location=start,
            radius=4,
            color=color,
            fill=True,
            fill_opacity=1,
            popup=f"任务 {route['task_id']} 起点<br>{route['start_time']}",
            tooltip=f"任务 {route['task_id']} 起点",
        ).add_to(fg_task)
        folium.Marker(
            location=end,
            icon=folium.Icon(color='red', icon='flag'),
            popup=f"任务 {route['task_id']} 终点<br>{route['end_time']}",
            tooltip=f"任务 {route['task_id']} 终点",
        ).add_to(fg_task)

        fg_task.add_to(m)

    fg_lanes.add_to(m)
    fg_polygons.add_to(m)

    m.fit_bounds([[south, west], [north, east]])
    folium.LayerControl(collapsed=False).add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))


def main():
    routes = get_task_routes(TASK_FILE)
    if not routes:
        raise RuntimeError('没有在任务文件中找到可绘制的轨迹点')

    route_bbox = bbox_from_points([r['points'] for r in routes])
    south, west, north, east = route_bbox
    filter_bbox = (
        south - RESOURCE_BBOX_MARGIN,
        west - RESOURCE_BBOX_MARGIN,
        north + RESOURCE_BBOX_MARGIN,
        east + RESOURCE_BBOX_MARGIN,
    )
    polygons, lanes = parse_resource_features(RESOURCE_FILE, filter_bbox)
    build_map(routes, polygons, lanes, OUTPUT_HTML)

    print(f'任务数量：{len(routes)}')
    print(f'资源区域数量：{len(polygons)}')
    print(f'资源车道/路径数量：{len(lanes)}')
    print(f'地图已生成：{OUTPUT_HTML}')


if __name__ == '__main__':
    main()
