import threading
import traceback
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from LanePlanner import DEFAULT_SETTINGS, LanePlanner, load_lanes
from MapRenderer import build_route_map
from TaskData import determine_origin_destination, find_task


# =============================================================================
# 配置
# =============================================================================
THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name.lower() == "src":
    PROJECT_ROOT = THIS_FILE.parent.parent
else:
    PROJECT_ROOT = THIS_FILE.parent

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
# 引擎：缓存车道网络和查询结果
# =============================================================================
class RouteEngine:
    def __init__(self, app_settings, planner_settings):
        self.app_settings = app_settings
        self.planner_settings = planner_settings
        self.planner = None
        self.cache = {}

    def ensure_loaded(self):
        if self.planner is not None:
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

    def reload(self):
        self.planner = None
        self.cache.clear()
        self.ensure_loaded()
        return len(self.planner.lanes)

    def get_route(self, sn, task_id):
        self.ensure_loaded()

        cache_key = sn.strip().lower() + "|" + task_id.strip().lower()
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
        }

        self.cache[cache_key] = bundle
        if len(self.cache) > self.app_settings["max_cache_items"]:
            self.cache.clear()

        return bundle


ENGINE = RouteEngine(APP_SETTINGS, PLANNER_SETTINGS)


# =============================================================================
# 前端页面
# =============================================================================
INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>车辆历史路线与规划路线</title>
  <style>
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }
    body { background: #f3f5f8; color: #182230; }
    .page { height: 100%; display: grid; grid-template-rows: auto 1fr; }
    .topbar { background: #fff; border-bottom: 1px solid #dfe3e8; padding: 14px 18px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; box-shadow: 0 1px 4px rgba(0,0,0,.05); }
    .brand { font-weight: 700; font-size: 17px; margin-right: 8px; white-space: nowrap; }
    .field { display: flex; align-items: center; gap: 7px; }
    label { font-size: 13px; color: #52606d; white-space: nowrap; }
    input[type="text"] { width: 225px; height: 38px; border: 1px solid #cbd2d9; border-radius: 7px; padding: 0 11px; font-size: 14px; outline: none; }
    input[type="text"]:focus { border-color: #1677ff; box-shadow: 0 0 0 3px rgba(22,119,255,.12); }
    .check { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: #52606d; white-space: nowrap; }
    button { height: 38px; border: 0; border-radius: 7px; padding: 0 17px; background: #1677ff; color: #fff; font-size: 14px; cursor: pointer; }
    button:hover { background: #0958d9; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    button.secondary { background: #fff; color: #344054; border: 1px solid #cbd2d9; }
    button.secondary:hover { background: #f7f8fa; }
    .status { font-size: 13px; color: #667085; margin-left: auto; }
    .status.error { color: #d92d20; }
    .content { min-height: 0; display: grid; grid-template-columns: 310px 1fr; gap: 12px; padding: 12px; }
    .panel, .map-wrap { background: #fff; border: 1px solid #dfe3e8; border-radius: 9px; overflow: hidden; }
    .panel { padding: 15px; overflow: auto; }
    .panel h2 { font-size: 15px; margin: 0 0 13px; }
    .empty { color: #7b8794; font-size: 13px; line-height: 1.7; }
    .metric { padding: 9px 0; border-bottom: 1px solid #edf0f3; }
    .metric:last-child { border-bottom: 0; }
    .metric .name { color: #667085; font-size: 12px; }
    .metric .value { margin-top: 3px; font-size: 14px; word-break: break-all; }
    details { margin-top: 12px; }
    summary { cursor: pointer; font-size: 13px; color: #344054; }
    .lane-list { margin-top: 8px; font-size: 12px; line-height: 1.6; word-break: break-all; color: #52606d; }
    .map-wrap { position: relative; min-width: 0; min-height: 0; }
    iframe { width: 100%; height: 100%; border: 0; display: block; }
    .placeholder { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #7b8794; font-size: 14px; background: #fff; }
    .spinner { width: 18px; height: 18px; border: 2px solid #d0d5dd; border-top-color: #1677ff; border-radius: 50%; animation: spin .8s linear infinite; display: inline-block; vertical-align: middle; margin-right: 7px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 900px) { .content { grid-template-columns: 1fr; grid-template-rows: auto minmax(520px, 1fr); } .panel { max-height: 280px; } .status { width: 100%; margin-left: 0; } input[type="text"] { width: 190px; } }
  </style>
</head>
<body>
<div class="page">
  <form id="routeForm" class="topbar">
    <div class="brand">路线查询</div>
    <div class="field"><label for="sn">SN</label><input id="sn" name="sn" type="text" placeholder="例如 TLE00860CR1450020" autocomplete="off" required></div>
    <div class="field"><label for="taskId">task_id</label><input id="taskId" name="task_id" type="text" placeholder="输入完整 task_id" autocomplete="off" required></div>
    <label class="check"><input id="showLanes" type="checkbox"> 显示全部车道背景</label>
    <button id="submitBtn" type="submit">绘制路线</button>
    <button id="reloadBtn" class="secondary" type="button">重新加载数据</button>
    <div id="status" class="status">请输入 SN 和 task_id</div>
  </form>
  <main class="content">
    <aside class="panel"><h2>任务与路线指标</h2><div id="summary" class="empty">查询后，这里会显示历史轨迹点数、历史路线长度、规划路线长度、偏差和车道序列。</div></aside>
    <section class="map-wrap"><div id="placeholder" class="placeholder">路线地图将在这里显示</div><iframe id="mapFrame" title="路线地图" sandbox="allow-scripts allow-same-origin allow-popups"></iframe></section>
  </main>
</div>
<script>
  const form = document.getElementById('routeForm');
  const submitBtn = document.getElementById('submitBtn');
  const reloadBtn = document.getElementById('reloadBtn');
  const statusBox = document.getElementById('status');
  const summaryBox = document.getElementById('summary');
  const mapFrame = document.getElementById('mapFrame');
  const placeholder = document.getElementById('placeholder');

  function escapeHtml(value) { return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;'); }
  function number(value, digits = 1) { const n = Number(value); return Number.isFinite(n) ? n.toFixed(digits) : '-'; }
  function setBusy(busy, text = '') { submitBtn.disabled = busy; reloadBtn.disabled = busy; statusBox.classList.remove('error'); statusBox.innerHTML = busy ? `<span class="spinner"></span>${escapeHtml(text)}` : escapeHtml(text); }
  function showError(message) { statusBox.classList.add('error'); statusBox.textContent = message; summaryBox.className = 'empty'; summaryBox.textContent = message; placeholder.style.display = 'flex'; placeholder.textContent = '未能绘制路线'; }
  function renderSummary(s) {
    const metrics = [
      ['SN', s.sn], ['task_id', s.task_id], ['任务文件', s.source_file], ['OD 来源', s.od_source], ['历史轨迹点数', s.history_point_count],
      ['历史路线长度', `${number(s.history_distance_m)} m`], ['规划路线长度', `${number(s.planned_distance_m)} m`], ['长度差（历史-规划）', `${number(s.distance_difference_m)} m`],
      ['起点车道', s.start_lane_uid], ['终点车道', s.end_lane_uid], ['规划车道数', s.planned_lane_count]
    ];
    summaryBox.className = '';
    summaryBox.innerHTML = metrics.map(([name, value]) => `<div class="metric"><div class="name">${escapeHtml(name)}</div><div class="value">${escapeHtml(value)}</div></div>`).join('') + `<details><summary>查看规划车道 UID 序列</summary><div class="lane-list">${escapeHtml((s.planned_lane_uids || []).join(' → '))}</div></details>`;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const sn = document.getElementById('sn').value.trim();
    const taskId = document.getElementById('taskId').value.trim();
    const showLanes = document.getElementById('showLanes').checked;
    if (!sn || !taskId) return showError('SN 和 task_id 不能为空。');
    setBusy(true, '正在查找任务并规划路线…');
    placeholder.style.display = 'flex';
    placeholder.innerHTML = '<span><span class="spinner"></span>正在计算并绘制地图…</span>';
    try {
      const response = await fetch('/api/route', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({sn: sn, task_id: taskId, show_lanes: showLanes}) });
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || '路线绘制失败');
      mapFrame.srcdoc = data.map_html;
      placeholder.style.display = 'none';
      renderSummary(data.summary);
      setBusy(false, `绘制完成：${sn} / ${taskId}`);
    } catch (error) { submitBtn.disabled = false; reloadBtn.disabled = false; showError(error.message || String(error)); }
  });

  reloadBtn.addEventListener('click', async () => {
    setBusy(true, '正在重新加载车道与任务缓存…');
    try {
      const response = await fetch('/api/reload', {method: 'POST'});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || '重新加载失败');
      setBusy(false, `数据已重新加载，共 ${data.lane_count} 条车道`);
    } catch (error) { submitBtn.disabled = false; reloadBtn.disabled = false; showError(error.message || String(error)); }
  });
</script>
</body>
</html>
"""


# =============================================================================
# Flask API
# =============================================================================
app = Flask(__name__)
app.json.ensure_ascii = False


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/api/route")
def api_route():
    body = request.get_json(silent=True) or {}
    sn = str(body.get("sn", "")).strip()
    task_id = str(body.get("task_id", "")).strip()
    show_lanes = bool(body.get("show_lanes", False))

    if not sn or not task_id:
        return jsonify(ok=False, error="SN 和 task_id 均不能为空。"), 400

    try:
        bundle = ENGINE.get_route(sn, task_id)
        map_html, summary = build_route_map(
            sn=sn,
            task_id=task_id,
            bundle=bundle,
            planner=ENGINE.planner,
            settings=APP_SETTINGS,
            show_lanes=show_lanes,
        )
        return jsonify(ok=True, map_html=map_html, summary=summary)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


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
    print("路线查询网页已启动")
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
