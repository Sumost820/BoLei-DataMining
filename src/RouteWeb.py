# -*- coding: utf-8 -*-
"""
统一路线网页服务：
1. 根据 SN + task_id 获取历史路径和规划路径；
2. 根据起点/终点经纬度规划路径；
3. 两种模式都支持按固定间隔输出规划路径采样点并下载 CSV；
4. 采样点海拔由 RouteSampler 中的 GPS 历史点估算器提供。

运行：
    python src/RouteWeb.py

打开：
    http://127.0.0.1:5000
"""

import threading
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Dict, List

import folium
from folium.plugins import Fullscreen, MeasureControl, MiniMap
from flask import Flask, Response, jsonify, render_template_string, request

from LanePlanner import (
    DEFAULT_SETTINGS,
    LanePlanner,
    downsample_evenly,
    latlon,
    latlon_of,
    load_lanes,
    make_point,
    polyline_length_m,
)
from MapRenderer import build_route_map
from RouteSampler import (
    make_gps_altitude_estimator,
    plan_and_sample_by_coords,
    records_to_csv_text,
    sample_route_records,
)
from TaskData import determine_origin_destination, find_task


# =============================================================================
# 配置
# =============================================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent if THIS_FILE.parent.name.lower() == "src" else THIS_FILE.parent

APP_SETTINGS = {
    "project_root": PROJECT_ROOT,
    "gps_data_dir": PROJECT_ROOT / "data" / "GPSdata",
    "resource_file": PROJECT_ROOT / "data" / "MapResource.json",
    "host": "127.0.0.1",
    "port": 5000,
    "max_history_points": 5000,
    "max_cache_items": 100,
    "auto_open_browser": True,
}

PLANNER_SETTINGS = dict(DEFAULT_SETTINGS)


# =============================================================================
# 引擎：共用车道网络、任务缓存、GPS 高程估算器
# =============================================================================
class RouteEngine:
    def __init__(self, app_settings: Dict[str, Any], planner_settings: Dict[str, Any]):
        self.app_settings = app_settings
        self.planner_settings = planner_settings
        self.planner = None
        self.altitude_estimator = None
        self.cache: Dict[str, Dict[str, Any]] = {}

    def ensure_loaded(self) -> None:
        if self.planner is not None and self.altitude_estimator is not None:
            return

        gps_data_dir = self.app_settings["gps_data_dir"]
        resource_file = self.app_settings["resource_file"]

        if not gps_data_dir.exists():
            raise FileNotFoundError("GPS 数据目录不存在：" + str(gps_data_dir))
        if not resource_file.exists():
            raise FileNotFoundError("车道资源文件不存在：" + str(resource_file))

        lanes = load_lanes(resource_file, self.planner_settings)
        if not lanes:
            raise RuntimeError("没有从资源文件中读取到车道：" + str(resource_file))

        self.planner = LanePlanner(lanes, self.planner_settings)

        self.altitude_estimator = make_gps_altitude_estimator(
            gps_data_dir=gps_data_dir,
            max_distance_m=60.0,
            k=8,
        )

    def reload(self) -> int:
        self.planner = None
        self.altitude_estimator = None
        self.cache.clear()
        self.ensure_loaded()
        return len(self.planner.lanes)

    def get_task_route(self, sn: str, task_id: str) -> Dict[str, Any]:
        self.ensure_loaded()

        cache_key = "task|" + sn.strip().lower() + "|" + task_id.strip().lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        task = find_task(sn, task_id, self.app_settings["gps_data_dir"])
        origin, destination, od_source = determine_origin_destination(task["history_points"])
        plan, tried = self.planner.plan(origin, destination)

        bundle = {
            "task": task,
            "origin": origin,
            "destination": destination,
            "od_source": od_source,
            "plan": plan,
            "tried": tried,
        }

        self.cache[cache_key] = bundle
        if len(self.cache) > self.app_settings["max_cache_items"]:
            self.cache.clear()

        return bundle

    def get_coord_route(
        self,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
    ) -> Dict[str, Any]:
        self.ensure_loaded()

        origin = make_point(start_lon, start_lat)
        destination = make_point(end_lon, end_lat)
        plan, tried = self.planner.plan(origin, destination)

        return {
            "origin": origin,
            "destination": destination,
            "plan": plan,
            "tried": tried,
            "od_source": "手动输入起终点坐标",
        }


ENGINE = RouteEngine(APP_SETTINGS, PLANNER_SETTINGS)

app = Flask(__name__)
app.json.ensure_ascii = False


# =============================================================================
# 通用工具
# =============================================================================
def _float_body(body: Dict[str, Any], name: str, default: Any = None) -> float:
    value = body.get(name, default)
    if value is None or str(value).strip() == "":
        raise ValueError("缺少参数：" + name)
    return float(value)


def _int_body(body: Dict[str, Any], name: str, default: Any = None) -> int:
    value = body.get(name, default)
    if value is None or str(value).strip() == "":
        raise ValueError("缺少参数：" + name)
    return int(value)


def _planned_lane_names(plan: Dict[str, Any]) -> List[str]:
    """把规划路径 UID 序列转换成车道名称序列。"""
    names: List[str] = []

    for uid in plan.get("path", []):
        lane = ENGINE.planner.lanes.get(uid) if ENGINE.planner else None
        if not lane:
            names.append(str(uid))
            continue

        name = lane.get("name") or uid
        names.append(str(name))

    return names


def _planned_lane_uids(plan: Dict[str, Any]) -> List[str]:
    return [str(uid) for uid in plan.get("path", [])]


def _task_sample_points(plan: Dict[str, Any], interval_m: float, max_points: int) -> Dict[str, Any]:
    records, meta = sample_route_records(
        ENGINE.planner,
        plan,
        interval_m=float(interval_m),
        include_end=True,
        max_records=int(max_points),
        altitude_provider=ENGINE.altitude_estimator,
    )

    for row in records:
        row["distance_m"] = row.get("distance_from_start_m", row.get("route_offset_m"))

    return {
        "points": records,
        "sample_meta": meta,
    }


def _summary_for_task(sn: str, task_id: str, bundle: Dict[str, Any], map_summary: Dict[str, Any],
                      sample: Dict[str, Any], interval_m: float) -> Dict[str, Any]:
    plan = bundle["plan"]
    meta = sample["sample_meta"]

    origin_projection = plan.get("origin_projection") or {}
    destination_projection = plan.get("destination_projection") or {}

    summary = dict(map_summary)
    summary.update({
        "mode": "task",
        "sn": sn,
        "task_id": task_id,
        "interval_m": float(interval_m),
        "sample_count": meta.get("sample_count"),
        "point_count": meta.get("sample_count"),
        "route_length_m": meta.get("route_length_m"),

        # 这两个字段就是前端左侧要显示的
        "start_snap_distance_m": origin_projection.get("snap_dist_m"),
        "end_snap_distance_m": destination_projection.get("snap_dist_m"),

        # 可选：吸附点经纬度，后续需要展示时也能用
        "start_projected_lon": origin_projection.get("projected_lon"),
        "start_projected_lat": origin_projection.get("projected_lat"),
        "end_projected_lon": destination_projection.get("projected_lon"),
        "end_projected_lat": destination_projection.get("projected_lat"),

        "planned_lane_names": _planned_lane_names(plan),
        "planned_lane_uids": _planned_lane_uids(plan),
    })

    return summary


def _summary_for_coord(result: Dict[str, Any]) -> Dict[str, Any]:
    plan = result["plan"]
    summary = dict(result["summary"])
    summary.update({
        "mode": "coord",
        "planned_lane_names": _planned_lane_names(plan),
        "planned_lane_uids": _planned_lane_uids(plan),
    })
    return summary


# =============================================================================
# 坐标模式地图
# =============================================================================
def _add_lane_background(map_object, lanes) -> None:
    group = folium.FeatureGroup(name="全部车道（背景）", show=True)

    for lane in lanes:
        points = downsample_evenly(lane["points"], 100)
        folium.PolyLine(
            locations=latlon(points),
            color="#98A2B3",
            weight=1,
            opacity=0.30,
            tooltip=f"{lane.get('name')} | UID={lane.get('uid')}",
        ).add_to(group)

    group.add_to(map_object)


def _fit_bounds(map_object, lats: List[float], lons: List[float]) -> None:
    if not lats or not lons:
        return

    padding = 0.001
    map_object.fit_bounds([
        [min(lats) - padding, min(lons) - padding],
        [max(lats) + padding, max(lons) + padding],
    ])


def _build_coord_map(bundle: Dict[str, Any], show_lanes: bool, sample_points: List[Dict[str, Any]]) -> str:
    origin = bundle["origin"]
    destination = bundle["destination"]
    plan = bundle["plan"]

    planned_points = downsample_evenly(ENGINE.planner.visible_points(plan), 5000)

    lats = [p["lat"] for p in planned_points] + [origin["lat"], destination["lat"]]
    lons = [p["lon"] for p in planned_points] + [origin["lon"], destination["lon"]]

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    if show_lanes:
        _add_lane_background(map_object, ENGINE.planner.lanes.values())

    route_group = folium.FeatureGroup(name="规划路线", show=True)
    path = plan["path"]

    for index, uid in enumerate(path):
        lane = ENGINE.planner.lanes[uid]
        visible = ENGINE.planner.visible_lane_points(
            path,
            uid,
            index,
            plan["origin_projection"],
            plan["destination_projection"],
        )

        folium.PolyLine(
            locations=latlon(visible),
            color="#f5222d",
            weight=7,
            opacity=0.92,
            popup=folium.Popup(
                f"规划路线第 {index + 1} 段<br>"
                f"UID：{uid}<br>"
                f"名称：{lane.get('name')}",
                max_width=360,
            ),
            tooltip=f"规划 {index + 1}: {lane.get('name')}",
        ).add_to(route_group)

    route_group.add_to(map_object)

    if sample_points:
        sample_group = folium.FeatureGroup(name="规划路径采样点（地图降采样显示）", show=False)
        shown = downsample_evenly(sample_points, 800)

        for row in shown:
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=2,
                color="#1677ff",
                fill=True,
                fill_opacity=0.85,
                tooltip=(
                    f"#{row.get('sample_index')} | "
                    f"{row.get('distance_m')} m | "
                    f"alt={row.get('altitude_m')} | "
                    f"{row.get('altitude_source')}"
                ),
            ).add_to(sample_group)

        sample_group.add_to(map_object)

    marker_group = folium.FeatureGroup(name="起终点与吸附点", show=True)

    folium.Marker(
        latlon_of(origin),
        tooltip="输入起点",
        popup="输入起点",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(marker_group)

    folium.Marker(
        latlon_of(destination),
        tooltip="输入终点",
        popup="输入终点",
        icon=folium.Icon(color="blue", icon="flag"),
    ).add_to(marker_group)

    marker_items = [
        ("起点吸附", origin, plan["origin_projection"], "#13c2c2"),
        ("终点吸附", destination, plan["destination_projection"], "#722ed1"),
    ]

    for label, raw_point, projection, color in marker_items:
        projected = (projection["projected_lat"], projection["projected_lon"])

        folium.CircleMarker(
            projected,
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1,
            tooltip=label,
        ).add_to(marker_group)

        folium.PolyLine(
            [latlon_of(raw_point), projected],
            color=color,
            weight=3,
            dash_array="6,6",
            opacity=0.9,
        ).add_to(marker_group)

    marker_group.add_to(map_object)

    MiniMap(toggle_display=True).add_to(map_object)
    Fullscreen().add_to(map_object)
    MeasureControl(primary_length_unit="meters").add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)

    _fit_bounds(map_object, lats, lons)

    return map_object.get_root().render()


# =============================================================================
# 前端页面
# =============================================================================
INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>历史路线 / 坐标规划 / 路径采样</title>
  <style>
    * { box-sizing: border-box; }

    html, body {
      height: 100%;
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      color: #182230;
      background: #f3f5f8;
    }

    .page {
      height: 100%;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    header {
      background: #fff;
      border-bottom: 1px solid #dfe3e8;
      padding: 12px 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }

    .title {
      font-size: 17px;
      font-weight: 700;
      margin-bottom: 10px;
    }

    .toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .tabs {
      display: inline-flex;
      border: 1px solid #cbd2d9;
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }

    .tab {
      height: 36px;
      padding: 0 14px;
      border: 0;
      background: #fff;
      color: #344054;
      cursor: pointer;
      border-radius: 0;
    }

    .tab.active {
      background: #1677ff;
      color: #fff;
    }

    .field {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    label {
      font-size: 13px;
      color: #52606d;
      white-space: nowrap;
    }

    input[type="text"],
    input[type="number"] {
      height: 36px;
      border: 1px solid #cbd2d9;
      border-radius: 7px;
      padding: 0 9px;
      font-size: 13px;
      outline: none;
      background: #fff;
    }

    input[type="text"] { width: 210px; }
    input.coord { width: 120px; }
    input.small { width: 86px; }

    input:focus {
      border-color: #1677ff;
      box-shadow: 0 0 0 3px rgba(22,119,255,.12);
    }

    button {
      height: 36px;
      border: 0;
      border-radius: 7px;
      padding: 0 14px;
      background: #1677ff;
      color: #fff;
      font-size: 13px;
      cursor: pointer;
    }

    button:hover { background: #0958d9; }
    button:disabled { opacity: .55; cursor: not-allowed; }

    button.secondary {
      background: #fff;
      color: #344054;
      border: 1px solid #cbd2d9;
    }

    .check {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 13px;
      color: #52606d;
    }

    .status {
      margin-left: auto;
      font-size: 13px;
      color: #667085;
    }

    .status.error {
      color: #d92d20;
    }

    .mode-form {
      display: none;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .mode-form.active {
      display: flex;
    }

    .content {
      min-height: 0;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 12px;
      padding: 12px;
    }

    .left {
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 12px;
    }

    .panel,
    .map-wrap,
    .table-card {
      background: #fff;
      border: 1px solid #dfe3e8;
      border-radius: 9px;
      overflow: hidden;
    }

    .panel {
      padding: 14px;
      overflow: auto;
    }

    .panel h2 {
      font-size: 15px;
      margin: 0 0 12px;
    }

    .metric {
      padding: 8px 0;
      border-bottom: 1px solid #edf0f3;
    }

    .metric:last-child {
      border-bottom: 0;
    }

    .metric .name {
      color: #667085;
      font-size: 12px;
    }

    .metric .value {
      margin-top: 3px;
      font-size: 14px;
      word-break: break-word;
    }

    .lane-sequence {
      line-height: 1.65;
      white-space: normal;
      word-break: break-word;
      color: #344054;
      font-size: 13px;
    }

    .table-card {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
    }

    .table-toolbar {
      padding: 10px;
      border-bottom: 1px solid #edf0f3;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .note {
      color: #667085;
      font-size: 12px;
      line-height: 1.5;
    }

    .table-wrap {
      overflow: auto;
      min-height: 0;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    th, td {
      border-bottom: 1px solid #edf0f3;
      padding: 7px 8px;
      text-align: left;
      white-space: nowrap;
    }

    th {
      position: sticky;
      top: 0;
      background: #f8fafc;
      z-index: 1;
      color: #52606d;
      font-weight: 600;
    }

    tr:hover td {
      background: #fafcff;
    }

    .empty {
      padding: 24px;
      color: #7b8794;
      font-size: 13px;
      text-align: center;
      line-height: 1.7;
    }

    .map-wrap {
      position: relative;
      min-width: 0;
      min-height: 0;
    }

    iframe {
      width: 100%;
      height: 100%;
      border: 0;
      display: block;
    }

    .placeholder {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #7b8794;
      font-size: 14px;
      background: #fff;
      z-index: 2;
    }

    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid #d0d5dd;
      border-top-color: #1677ff;
      border-radius: 50%;
      animation: spin .8s linear infinite;
      display: inline-block;
      vertical-align: -3px;
      margin-right: 6px;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    @media (max-width: 1050px) {
      .content {
        grid-template-columns: 1fr;
        grid-template-rows: auto minmax(560px, 1fr);
      }

      .left {
        grid-template-rows: auto 320px;
      }

      .status {
        width: 100%;
        margin-left: 0;
      }
    }
  </style>
</head>

<body>
<div class="page">
  <header>
    <div class="title">历史路线 / 坐标规划 / 路径采样</div>

    <div class="toolbar">
      <div class="tabs">
        <button id="taskTab" class="tab active" type="button">SN + task_id</button>
        <button id="coordTab" class="tab" type="button">起终点坐标</button>
      </div>

      <form id="taskForm" class="mode-form active">
        <div class="field">
          <label>SN</label>
          <input id="sn" type="text" value="TLE00860CR1450020" required>
        </div>

        <div class="field">
          <label>task_id</label>
          <input id="taskId" type="text" value="8061955916906753" required>
        </div>

        <div class="field">
          <label>采样间隔 m</label>
          <input id="taskInterval" class="small" type="number" min="0.1" step="0.1" value="1">
        </div>

        <div class="field">
          <label>最大点数</label>
          <input id="taskMaxPoints" class="small" type="number" min="2" step="1" value="20000">
        </div>

        <label class="check">
          <input id="taskShowLanes" type="checkbox">
          显示全部车道
        </label>

        <button id="taskSubmitBtn" type="submit">绘制并采样</button>
      </form>

      <form id="coordForm" class="mode-form">
        <div class="field">
          <label>起点经度</label>
          <input id="startLon" class="coord" type="number" step="0.00000001" value="89.27705734" required>
        </div>

        <div class="field">
          <label>起点纬度</label>
          <input id="startLat" class="coord" type="number" step="0.00000001" value="44.83500106" required>
        </div>

        <div class="field">
          <label>终点经度</label>
          <input id="endLon" class="coord" type="number" step="0.00000001" value="89.24726732" required>
        </div>

        <div class="field">
          <label>终点纬度</label>
          <input id="endLat" class="coord" type="number" step="0.00000001" value="44.83520039" required>
        </div>

        <div class="field">
          <label>采样间隔 m</label>
          <input id="coordInterval" class="small" type="number" min="0.1" step="0.1" value="1">
        </div>

        <div class="field">
          <label>最大点数</label>
          <input id="coordMaxPoints" class="small" type="number" min="2" step="1" value="20000">
        </div>

        <label class="check">
          <input id="coordShowLanes" type="checkbox">
          显示全部车道
        </label>

        <button id="coordSubmitBtn" type="submit">规划并采样</button>
      </form>

      <button id="downloadBtn" type="button" class="secondary" disabled>下载 CSV</button>
      <button id="reloadBtn" type="button" class="secondary">重新加载数据</button>

      <div id="status" class="status"> </div>
    </div>
  </header>

  <main class="content">
    <section class="left">
      <aside class="panel">
        <h2>路线指标</h2>
        <div id="summary" class="empty">查询后，这里会显示路线长度、采样点数、吸附距离和规划车道名称序列。</div>
      </aside>

      <section class="table-card">
        <div class="table-toolbar">
          <button id="copyBtn" type="button" class="secondary" disabled>复制前 100 行</button>
          <span id="tableNote" class="note">采样点会显示在这里。</span>
        </div>
        <div id="tableWrap" class="table-wrap">
          <div class="empty">还没有采样点。</div>
        </div>
      </section>
    </section>

    <section class="map-wrap">
      <div id="placeholder" class="placeholder">路线地图将在这里显示</div>
      <iframe id="mapFrame" title="路线地图" sandbox="allow-scripts allow-same-origin allow-popups"></iframe>
    </section>
  </main>
</div>

<script>
const taskTab = document.getElementById('taskTab');
const coordTab = document.getElementById('coordTab');
const taskForm = document.getElementById('taskForm');
const coordForm = document.getElementById('coordForm');

const statusBox = document.getElementById('status');
const summaryBox = document.getElementById('summary');
const tableWrap = document.getElementById('tableWrap');
const tableNote = document.getElementById('tableNote');
const mapFrame = document.getElementById('mapFrame');
const placeholder = document.getElementById('placeholder');

const downloadBtn = document.getElementById('downloadBtn');
const reloadBtn = document.getElementById('reloadBtn');
const copyBtn = document.getElementById('copyBtn');

let currentMode = 'task';
let lastPayload = null;
let lastCsvUrl = null;
let lastPoints = [];

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function number(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : '-';
}

function setMode(mode) {
  currentMode = mode;

  const isTask = mode === 'task';

  taskTab.classList.toggle('active', isTask);
  coordTab.classList.toggle('active', !isTask);

  taskForm.classList.toggle('active', isTask);
  coordForm.classList.toggle('active', !isTask);

  statusBox.classList.remove('error');
  statusBox.textContent = isTask ? '请输入 SN 和 task_id' : '请输入起点和终点经纬度';
}

function setBusy(busy, text) {
  document.getElementById('taskSubmitBtn').disabled = busy;
  document.getElementById('coordSubmitBtn').disabled = busy;
  reloadBtn.disabled = busy;
  downloadBtn.disabled = busy || !lastPayload;
  copyBtn.disabled = busy || lastPoints.length === 0;

  statusBox.classList.remove('error');
  statusBox.innerHTML = busy
    ? `<span class="spinner"></span>${escapeHtml(text)}`
    : escapeHtml(text || '');
}

function showError(message) {
  statusBox.classList.add('error');
  statusBox.textContent = message;
  placeholder.style.display = 'flex';
  placeholder.textContent = '未能绘制路线';
}

function renderSummary(s) {
  const laneNames = s.planned_lane_names || [];
  const laneText = laneNames.length ? laneNames.join(' → ') : '-';

  const plannedDistance = s.planned_distance_m ?? s.route_length_m;
  const sampleCount = s.sample_count ?? s.point_count;

  const metrics = [
    ['规划路线长度', `${number(plannedDistance, 2)} m`],
    ['采样点数', sampleCount ?? '-'],
    ['采样间隔', `${number(s.interval_m, 1)} m`],
    ['起点吸附距离', `${number(s.start_snap_distance_m, 2)} m`],
    ['终点吸附距离', `${number(s.end_snap_distance_m, 2)} m`],
  ];

  summaryBox.className = '';
  summaryBox.innerHTML =
    metrics.map(([name, value]) =>
      `<div class="metric">
        <div class="name">${escapeHtml(name)}</div>
        <div class="value">${escapeHtml(value)}</div>
      </div>`
    ).join('') +
    `<div class="metric">
      <div class="name">规划车道名称序列</div>
      <div class="value lane-sequence">${escapeHtml(laneText)}</div>
    </div>`;
}

function renderTable(points) {
  lastPoints = points || [];

  if (!lastPoints.length) {
    tableWrap.innerHTML = '<div class="empty">没有采样点。</div>';
    tableNote.textContent = '没有可显示的采样点。';
    copyBtn.disabled = true;
    return;
  }

  const columns = [
    'sample_index',
    'distance_m',
    'route_offset_m',
    'lon',
    'lat',
    'altitude_m',
    'altitude_source',
    'altitude_nearest_distance_m',
    'lane_name',
    'lane_uid',
    'lane_order'
  ];

  const shown = lastPoints.slice(0, 1000);
  const head = columns.map(c => `<th>${escapeHtml(c)}</th>`).join('');

  const body = shown.map(row =>
    `<tr>${columns.map(c => `<td>${escapeHtml(row[c])}</td>`).join('')}</tr>`
  ).join('');

  tableWrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

  tableNote.textContent = lastPoints.length > shown.length
    ? `共 ${lastPoints.length} 个点，预览前 ${shown.length} 个；完整结果请下载 CSV。`
    : `共 ${lastPoints.length} 个点。`;

  copyBtn.disabled = false;
}

function taskPayload() {
  return {
    sn: document.getElementById('sn').value.trim(),
    task_id: document.getElementById('taskId').value.trim(),
    interval_m: Number(document.getElementById('taskInterval').value),
    max_points: Number(document.getElementById('taskMaxPoints').value),
    show_lanes: document.getElementById('taskShowLanes').checked,
  };
}

function coordPayload() {
  return {
    start_lon: Number(document.getElementById('startLon').value),
    start_lat: Number(document.getElementById('startLat').value),
    end_lon: Number(document.getElementById('endLon').value),
    end_lat: Number(document.getElementById('endLat').value),
    interval_m: Number(document.getElementById('coordInterval').value),
    max_points: Number(document.getElementById('coordMaxPoints').value),
    show_lanes: document.getElementById('coordShowLanes').checked,
    include_end: true,
  };
}

async function runRoute(url, payload, csvUrl) {
  lastPayload = null;
  lastCsvUrl = null;
  lastPoints = [];
  downloadBtn.disabled = true;
  copyBtn.disabled = true;

  placeholder.style.display = 'flex';
  placeholder.innerHTML = '<span><span class="spinner"></span>正在计算并绘制地图…</span>';
  setBusy(true, '正在规划路线并生成采样点…');

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || '处理失败');
    }

    mapFrame.srcdoc = data.map_html || '';
    placeholder.style.display = 'none';

    renderSummary(data.summary || {});
    renderTable(data.points || []);

    lastPayload = payload;
    lastCsvUrl = csvUrl;

    downloadBtn.disabled = false;
    copyBtn.disabled = lastPoints.length === 0;

    setBusy(false, `完成：${lastPoints.length} 个采样点。`);
  } catch (error) {
    showError(error.message || String(error));
    document.getElementById('taskSubmitBtn').disabled = false;
    document.getElementById('coordSubmitBtn').disabled = false;
    reloadBtn.disabled = false;
  }
}

taskTab.addEventListener('click', () => setMode('task'));
coordTab.addEventListener('click', () => setMode('coord'));

taskForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const payload = taskPayload();
  if (!payload.sn || !payload.task_id) {
    showError('SN 和 task_id 不能为空。');
    return;
  }

  await runRoute('/api/task_route', payload, '/api/task_route_samples_csv');
});

coordForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const payload = coordPayload();
  await runRoute('/api/coord_route', payload, '/api/coord_route_samples_csv');
});

downloadBtn.addEventListener('click', async () => {
  if (!lastPayload || !lastCsvUrl) return;

  setBusy(true, '正在生成 CSV…');

  try {
    const response = await fetch(lastCsvUrl, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(lastPayload),
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || 'CSV 下载失败');
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');

    a.href = url;
    a.download = currentMode === 'task' ? 'task_route_samples.csv' : 'coord_route_samples.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);
    setBusy(false, 'CSV 已下载。');
  } catch (error) {
    showError(error.message || String(error));
    document.getElementById('taskSubmitBtn').disabled = false;
    document.getElementById('coordSubmitBtn').disabled = false;
    reloadBtn.disabled = false;
    downloadBtn.disabled = false;
  }
});

reloadBtn.addEventListener('click', async () => {
  setBusy(true, '正在重新加载数据…');

  try {
    const response = await fetch('/api/reload', {method: 'POST'});
    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.error || '重新加载失败');
    }

    setBusy(false, `重新加载完成：${data.lane_count} 条车道。`);
  } catch (error) {
    showError(error.message || String(error));
    document.getElementById('taskSubmitBtn').disabled = false;
    document.getElementById('coordSubmitBtn').disabled = false;
    reloadBtn.disabled = false;
  }
});

copyBtn.addEventListener('click', async () => {
  if (!lastPoints.length) return;

  const columns = [
    'sample_index',
    'distance_m',
    'route_offset_m',
    'lon',
    'lat',
    'altitude_m',
    'altitude_source',
    'altitude_nearest_distance_m',
    'lane_name',
    'lane_uid',
    'lane_order'
  ];

  const rows = lastPoints.slice(0, 100).map(row =>
    columns.map(c => row[c] ?? '').join(',')
  );

  const text = [columns.join(','), ...rows].join('\n');
  await navigator.clipboard.writeText(text);

  statusBox.classList.remove('error');
  statusBox.textContent = '已复制前 100 行 CSV 文本。';
});
</script>
</body>
</html>
"""


# =============================================================================
# Flask API
# =============================================================================
@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/api/task_route")
def api_task_route():
    body = request.get_json(silent=True) or {}

    sn = str(body.get("sn", "")).strip()
    task_id = str(body.get("task_id", "")).strip()
    show_lanes = bool(body.get("show_lanes", False))
    interval_m = _float_body(body, "interval_m", 1.0)
    max_points = _int_body(body, "max_points", 20000)

    if not sn or not task_id:
        return jsonify(ok=False, error="SN 和 task_id 均不能为空。"), 400

    try:
        ENGINE.ensure_loaded()

        bundle = ENGINE.get_task_route(sn, task_id)

        map_html, map_summary = build_route_map(
            sn=sn,
            task_id=task_id,
            bundle=bundle,
            planner=ENGINE.planner,
            settings=APP_SETTINGS,
            show_lanes=show_lanes,
        )

        sample = _task_sample_points(bundle["plan"], interval_m, max_points)
        summary = _summary_for_task(sn, task_id, bundle, map_summary, sample, interval_m)

        return jsonify(
            ok=True,
            map_html=map_html,
            summary=summary,
            points=sample["points"],
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.post("/api/coord_route")
def api_coord_route():
    body = request.get_json(silent=True) or {}

    try:
        ENGINE.ensure_loaded()

        start_lon = _float_body(body, "start_lon")
        start_lat = _float_body(body, "start_lat")
        end_lon = _float_body(body, "end_lon")
        end_lat = _float_body(body, "end_lat")
        interval_m = _float_body(body, "interval_m", 1.0)
        max_points = _int_body(body, "max_points", 20000)
        include_end = bool(body.get("include_end", True))
        show_lanes = bool(body.get("show_lanes", False))

        result = plan_and_sample_by_coords(
            ENGINE.planner,
            start_lon=start_lon,
            start_lat=start_lat,
            end_lon=end_lon,
            end_lat=end_lat,
            interval_m=interval_m,
            include_end=include_end,
            max_points=max_points,
            altitude_provider=ENGINE.altitude_estimator,
        )

        bundle = {
            "origin": result["origin"],
            "destination": result["destination"],
            "plan": result["plan"],
            "od_source": "手动输入起终点坐标",
        }

        map_html = _build_coord_map(
            bundle=bundle,
            show_lanes=show_lanes,
            sample_points=result["points"],
        )

        summary = _summary_for_coord(result)

        return jsonify(
            ok=True,
            map_html=map_html,
            summary=summary,
            points=result["points"],
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.post("/api/task_route_samples_csv")
def api_task_route_samples_csv():
    body = request.get_json(silent=True) or {}

    sn = str(body.get("sn", "")).strip()
    task_id = str(body.get("task_id", "")).strip()
    interval_m = _float_body(body, "interval_m", 1.0)
    max_points = _int_body(body, "max_points", 20000)

    if not sn or not task_id:
        return jsonify(ok=False, error="SN 和 task_id 均不能为空。"), 400

    try:
        ENGINE.ensure_loaded()

        bundle = ENGINE.get_task_route(sn, task_id)
        sample = _task_sample_points(bundle["plan"], interval_m, max_points)
        csv_text = records_to_csv_text(sample["points"])

        return Response(
            "\ufeff" + csv_text,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=task_route_samples.csv"},
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.post("/api/coord_route_samples_csv")
def api_coord_route_samples_csv():
    body = request.get_json(silent=True) or {}

    try:
        ENGINE.ensure_loaded()

        result = plan_and_sample_by_coords(
            ENGINE.planner,
            start_lon=_float_body(body, "start_lon"),
            start_lat=_float_body(body, "start_lat"),
            end_lon=_float_body(body, "end_lon"),
            end_lat=_float_body(body, "end_lat"),
            interval_m=_float_body(body, "interval_m", 1.0),
            include_end=bool(body.get("include_end", True)),
            max_points=_int_body(body, "max_points", 20000),
            altitude_provider=ENGINE.altitude_estimator,
        )

        csv_text = records_to_csv_text(result["points"])

        return Response(
            "\ufeff" + csv_text,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=coord_route_samples.csv"},
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


# -----------------------------------------------------------------------------
# 兼容旧接口
# -----------------------------------------------------------------------------
@app.post("/api/route")
def api_route_compat():
    body = request.get_json(silent=True) or {}
    body.setdefault("interval_m", 1.0)
    body.setdefault("max_points", 20000)

    with app.test_request_context(
        "/api/task_route",
        method="POST",
        json=body,
    ):
        return api_task_route()


@app.post("/api/path_points")
def api_path_points_compat():
    body = request.get_json(silent=True) or {}
    body.setdefault("interval_m", 1.0)
    body.setdefault("max_points", 20000)

    with app.test_request_context(
        "/api/coord_route",
        method="POST",
        json=body,
    ):
        return api_coord_route()


@app.post("/api/path_points_csv")
def api_path_points_csv_compat():
    body = request.get_json(silent=True) or {}
    body.setdefault("interval_m", 1.0)
    body.setdefault("max_points", 20000)

    with app.test_request_context(
        "/api/coord_route_samples_csv",
        method="POST",
        json=body,
    ):
        return api_coord_route_samples_csv()


@app.post("/api/reload")
def api_reload():
    try:
        lane_count = ENGINE.reload()
        return jsonify(ok=True, lane_count=lane_count)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.get("/api/health")
def api_health():
    try:
        ENGINE.ensure_loaded()
        return jsonify(
            ok=True,
            project_root=str(APP_SETTINGS["project_root"]),
            gps_data_dir=str(APP_SETTINGS["gps_data_dir"]),
            resource_file=str(APP_SETTINGS["resource_file"]),
            lane_count=len(ENGINE.planner.lanes),
        )
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500


def main():
    print("=" * 72)
    print("统一路线网页服务已启动")
    print("地址：http://%s:%s" % (APP_SETTINGS["host"], APP_SETTINGS["port"]))
    print("GPS 数据：" + str(APP_SETTINGS["gps_data_dir"]))
    print("车道资源：" + str(APP_SETTINGS["resource_file"]))
    print("停止服务：点击 PyCharm 运行窗口中的红色停止按钮，或在终端按 Ctrl+C")
    print("=" * 72)

    if APP_SETTINGS["auto_open_browser"]:
        url = "http://%s:%s" % (APP_SETTINGS["host"], APP_SETTINGS["port"])
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(
        host=APP_SETTINGS["host"],
        port=APP_SETTINGS["port"],
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()