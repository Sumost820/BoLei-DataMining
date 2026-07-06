import heapq
import json
import math
from pathlib import Path

# 配置
DEFAULT_SETTINGS = {
    "lane_type_ids": [101],          # 只读这些类型的车道。需要临时车道时改成 [101, 116]
    "use_temp_lane": True,
    "connect_distance_threshold_m": 100.0,  # 车道连接距离阈值
    "snap_candidate_top_k": 5,  # OD吸附车道备选数量
    "snap_max_distance_m": 100.0,  # OD吸附最大距离，超过警告
    "snap_distance_penalty_factor": 5.0,  # OD吸附惩罚因子
    "jump_penalty_factor": 10.0,  # 非前驱后继的车道连接惩罚因子
}

EARTH_RADIUS_M = 6371000.0

# ======================================基础数据=================================
# GPS采样点
def make_point(lon, lat, z=None, timestamp=None):
    return {"lon": float(lon), "lat": float(lat), "z": z, "timestamp": timestamp}


def latlon_of(point):
    return point["lat"], point["lon"]


# ====================================几何工具=================================
# 直线距离
def haversine_m(a, b):
    lon1 = math.radians(a["lon"])
    lat1 = math.radians(a["lat"])
    lon2 = math.radians(b["lon"])
    lat2 = math.radians(b["lat"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_M * 2.0 * math.asin(math.sqrt(value))

# 经纬度转换为平面坐标，单位米
def lonlat_to_xy_m(point, ref):
    x = math.radians(point["lon"] - ref["lon"]) * EARTH_RADIUS_M * math.cos(math.radians(ref["lat"]))
    y = math.radians(point["lat"] - ref["lat"]) * EARTH_RADIUS_M
    return x, y

# 计算折线距离，单位米
def polyline_length_m(points):
    total = 0.0
    for i in range(len(points) - 1):
        total += haversine_m(points[i], points[i + 1])
    return total

# 投影点到折线
def project_point_to_polyline(point, points):
    """把点投影到折线，返回吸附距离、offset、remain 和投影点坐标。"""
    if not points:
        raise ValueError("空中心线，无法投影")

    # 单点投影
    if len(points) == 1:
        snap = haversine_m(point, points[0])
        return {"snap_dist_m": snap, "offset_m": 0.0, "remain_m": 0.0,
            "projected_lon": points[0]["lon"], "projected_lat": points[0]["lat"],
            "segment_index": 0,"lane_length_m": 0.0,}

    # 多点投影
    ref = point
    px, py = lonlat_to_xy_m(point, ref)  # 原点坐标
    best = None
    walked_m = 0.0

    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]

        ax, ay = lonlat_to_xy_m(start, ref)
        bx, by = lonlat_to_xy_m(end, ref)

        abx = bx - ax
        aby = by - ay
        ab_len2 = abx * abx + aby * aby

        if ab_len2 <= 0.0:
            ratio = 0.0
        else:
            ratio = ((px - ax) * abx + (py - ay) * aby) / ab_len2
            ratio = max(0.0, min(1.0, ratio))

        proj_x = ax + ratio * abx
        proj_y = ay + ratio * aby
        dist_m = math.hypot(px - proj_x, py - proj_y)

        segment_len_m = haversine_m(start, end)
        offset_m = walked_m + segment_len_m * ratio
        projected_lon = start["lon"] + (end["lon"] - start["lon"]) * ratio
        projected_lat = start["lat"] + (end["lat"] - start["lat"]) * ratio

        if best is None or dist_m < best["snap_dist_m"]:
            best = {"snap_dist_m": dist_m, "offset_m": offset_m, "projected_lon": projected_lon,
                "projected_lat": projected_lat, "segment_index": index}

        walked_m += segment_len_m

    total_len = polyline_length_m(points)
    best["remain_m"] = max(0.0, total_len - best["offset_m"])
    best["lane_length_m"] = total_len
    return best

# 找折线上某距离的点
def point_at_offset(points, target_offset_m):
    if not points:
        raise ValueError("空中心线，无法按 offset 取点。")
    if len(points) == 1:
        return points[0]

    total_len = polyline_length_m(points)
    target = max(0.0, min(float(target_offset_m), total_len))
    walked = 0.0

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        segment_len = haversine_m(start, end)
        if segment_len <= 1e-9:
            continue
        if walked + segment_len >= target:
            ratio = (target - walked) / segment_len
            z = None
            if start.get("z") is not None and end.get("z") is not None:
                z = start["z"] + (end["z"] - start["z"]) * ratio
            return make_point(
                start["lon"] + (end["lon"] - start["lon"]) * ratio,
                start["lat"] + (end["lat"] - start["lat"]) * ratio,
                z,
            )
        walked += segment_len

    return points[-1]

# 截取折线段
def clip_polyline_by_offsets(points, start_offset_m, end_offset_m):
    if not points:
        return []

    total_len = polyline_length_m(points)
    start_offset = max(0.0, min(float(start_offset_m), total_len))
    end_offset = max(0.0, min(float(end_offset_m), total_len))

    if end_offset < start_offset:
        return []
    if abs(end_offset - start_offset) <= 1e-6:
        return [point_at_offset(points, start_offset)]

    result = [point_at_offset(points, start_offset)]
    walked = 0.0

    for i in range(len(points) - 1):
        segment_len = haversine_m(points[i], points[i + 1])
        next_walked = walked + segment_len
        if start_offset < next_walked < end_offset:
            result.append(points[i + 1])
        walked = next_walked

    end_point = point_at_offset(points, end_offset)
    last = result[-1]
    if abs(last["lon"] - end_point["lon"]) > 1e-12 or abs(last["lat"] - end_point["lat"]) > 1e-12:
        result.append(end_point)

    return result

# 降采样折线，不超过 max_items 个点
def downsample_evenly(items, max_items):
    if len(items) <= max_items:
        return list(items)
    if max_items <= 1:
        return [items[0]]

    step = (len(items) - 1) / (max_items - 1)
    result = []
    used = set()
    for i in range(max_items):
        index = round(i * step)
        if index not in used:
            result.append(items[index])
            used.add(index)
    return result


def latlon(points):
    return [latlon_of(point) for point in points]


# =============================================================================
# 车道资源读取
def load_data_field(item):
    data = item.get("数据")
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return {}
        if isinstance(data, dict):
            return data
    return {}

# 从列表提取uid列表
def extract_id_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        result = []
        for item in value:
            if item is None:
                continue

            text = str(item).strip()
            if text and text != "None":
                result.append(text)

        return result

    text = str(value).strip()
    if text and text != "None":
        return [text]

    return []

# 从车道数据中提取中心线点
def extract_central_curve_points(data):
    points = []
    central_curve = data.get("central_curve") or {}
    segments = central_curve.get("segment") or []

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        line_segment = segment.get("line_segment") or {}
        raw_points = line_segment.get("point") or []  # GPS原始点
        for raw in raw_points:
            if not isinstance(raw, dict):
                continue
            lon = raw.get("x")
            lat = raw.get("y")
            z = raw.get("z")
            if lon is None or lat is None:
                continue
            try:
                z_value = None if z is None else float(z)
                points.append(make_point(float(lon), float(lat), z_value))
            except Exception:
                continue

    return points

# 从资源文件中加载车道数据
def load_lanes(resource_file, settings=None):
    resource_file = Path(resource_file)

    with resource_file.open("r", encoding="utf-8-sig") as file:
        resources = json.load(file)

    if isinstance(resources, dict):
        resources = [resources]
    if not isinstance(resources, list):
        raise ValueError("资源文件格式应为对象或数组：" + str(resource_file))

    lanes = {}
    lane_type_ids = set(settings["lane_type_ids"])

    for item in resources:
        if not isinstance(item, dict):
            continue

        type_id = item.get("资源类型ID")
        if type_id not in lane_type_ids:  # HD车道 101
            continue
        if type_id == 116 and not settings["use_temp_lane"]:
            continue

        data = load_data_field(item)
        points = extract_central_curve_points(data)
        uid = str(data.get("uid"))

        lane = {"uid": uid, "name": item.get("资源名称"), "type_id": int(type_id), "type_name": item.get("资源类型"),
                "points": points, "length_m": polyline_length_m(points), "width": data.get("width"),
                "speed_limit": data.get("speed_limit"), "direction": data.get("direction"),
                "temp_road": data.get("temp_road"), "successor_uid": extract_id_list(data.get("successor_uid")),
                "predecessor_uid": extract_id_list(data.get("predecessor_uid")), "start_point": points[0],
                "end_point": points[-1]}
        lanes[uid] = lane

    return lanes


# =============================================================================
# 规划器：用普通 class 保存 lanes/settings/graph
class LanePlanner:
    def __init__(self, lanes, settings=None):
        self.lanes = lanes
        self.settings = settings
        self.graph = self.build_graph()

    # 寻找起点终点最近的车道候选
    def snap_candidates(self, point):
        candidates = []
        for uid, lane in self.lanes.items():
            projection = project_point_to_polyline(point, lane["points"])
            candidates.append({"uid": uid, "projection": projection, "snap_dist_m": projection["snap_dist_m"]})

        candidates.sort(key=lambda item: item["snap_dist_m"])
        return candidates[: self.settings["snap_candidate_top_k"]]

    # 构建图
    def build_graph(self):
        edges = {}
        for uid in self.lanes:
            edges[uid] = {}

        # 添加或更新边
        def add_edge(from_uid, to_uid, weight):
            old = edges[from_uid].get(to_uid)
            if old is None or weight < old:
                edges[from_uid][to_uid] = weight

        # 1. 拓扑边
        for uid, lane in self.lanes.items():
            for next_uid in lane["successor_uid"]:
                if next_uid in self.lanes:
                    add_edge(uid, next_uid, self.lanes[next_uid]["length_m"])

            for pre_uid in lane["predecessor_uid"]:
                if pre_uid in self.lanes:
                    add_edge(pre_uid, uid, lane["length_m"])

        # 2. 端点距离兜底边
        lane_items = list(self.lanes.items())
        for uid_a, lane_a in lane_items:
            for uid_b, lane_b in lane_items:
                if uid_a == uid_b:
                    continue
                distance = haversine_m(lane_a["end_point"], lane_b["start_point"])
                if distance <= self.settings["connect_distance_threshold_m"]:
                    weight = lane_b["length_m"] + distance * self.settings["jump_penalty_factor"]
                    add_edge(uid_a, uid_b, weight)

        # 转为邻接表
        graph = {}
        for uid, next_map in edges.items():
            graph[uid] = list(next_map.items())
        return graph

    # Dijkstra 算法 寻找从 start 到 end 的最短路径
    def dijkstra(self, start_uid, end_uid):
        queue = [(0.0, start_uid)]
        distances = {start_uid: 0.0}
        previous = {}
        visited = set()

        while queue:
            current_distance, uid = heapq.heappop(queue)
            if uid in visited:
                continue
            visited.add(uid)

            if uid == end_uid:
                break

            for next_uid, weight in self.graph.get(uid, []):
                new_distance = current_distance + weight
                # 更新距离和前驱节点
                if next_uid not in distances or new_distance < distances[next_uid]:
                    distances[next_uid] = new_distance
                    previous[next_uid] = uid
                    heapq.heappush(queue, (new_distance, next_uid))

        if end_uid not in distances:
            return None, math.inf

        path = [end_uid]
        while path[-1] != start_uid:
            path.append(previous[path[-1]])
        path.reverse()
        return path, distances[end_uid]

    # 计算从 from_uid 到 to_uid 的跳跃距离
    def edge_jump_distance(self, from_uid, to_uid):
        lane_a = self.lanes[from_uid]
        lane_b = self.lanes[to_uid]

        if to_uid in lane_a["successor_uid"] or from_uid in lane_b["predecessor_uid"]:
            return 0.0

        distance = haversine_m(lane_a["end_point"], lane_b["start_point"])
        if distance <= self.settings["connect_distance_threshold_m"]:
            return distance
        return math.inf

    # 计算从 origin_projection 到 destination_projection 的路由指标
    # 包括总距离、目标距离、车道距离、跳跃距离、跳跃成本、跳跃成本惩罚
    def route_metrics(self, path, origin_projection, destination_projection):
        if not path:
            return None

        snap_distance = origin_projection["snap_dist_m"] + destination_projection["snap_dist_m"]
        lane_distance = 0.0
        jump_distance = 0.0
        penalized_jump_cost = 0.0

        # 情况 1：起点和终点在同一条车道上
        if len(path) == 1:
            if destination_projection["offset_m"] < origin_projection["offset_m"]:
                return None

            lane_distance = destination_projection["offset_m"] - origin_projection["offset_m"]

            objective = lane_distance + snap_distance * self.settings["snap_distance_penalty_factor"]

            return {
                "total_distance_m": lane_distance + snap_distance,
                "objective_m": objective,
                "lane_distance_m": lane_distance,
                "jump_distance_m": 0.0,
                "penalized_jump_cost_m": 0.0,
                "snap_distance_m": snap_distance,
            }

        # 情况 2：跨多条车道
        # 起点车道只走：起点投影点 -> 起点车道终点
        lane_distance += origin_projection["remain_m"]

        for index in range(1, len(path)):
            prev_uid = path[index - 1]
            cur_uid = path[index]

            jump = self.edge_jump_distance(prev_uid, cur_uid)
            if math.isinf(jump):
                return None

            jump_distance += jump
            penalized_jump_cost += jump * self.settings["jump_penalty_factor"]

            if index == len(path) - 1:
                # 终点车道只走：终点车道起点 -> 终点投影点
                lane_distance += destination_projection["offset_m"]
            else:
                # 中间车道走完整长度
                lane_distance += self.lanes[cur_uid]["length_m"]

        objective = lane_distance + penalized_jump_cost + snap_distance * self.settings["snap_distance_penalty_factor"]

        print("====== route_metrics 调试 ======")
        print("path:", path)
        print("path长度:", len(path))
        print("origin offset:", origin_projection["offset_m"])
        print("origin remain:", origin_projection["remain_m"])
        print("destination offset:", destination_projection["offset_m"])

        return {
            "total_distance_m": lane_distance + jump_distance + snap_distance,
            "objective_m": objective,
            "lane_distance_m": lane_distance,
            "jump_distance_m": jump_distance,
            "penalized_jump_cost_m": penalized_jump_cost,
            "snap_distance_m": snap_distance,
        }

    def plan(self, origin, destination):
        origin_candidates = self.snap_candidates(origin)
        destination_candidates = self.snap_candidates(destination)

        if not origin_candidates or not destination_candidates:
            raise RuntimeError("起点或终点附近没有可吸附车道。")

        plans = []
        for start in origin_candidates:
            for end in destination_candidates:
                path, dijkstra_cost = self.dijkstra(start["uid"], end["uid"])
                if path is None:
                    continue

                metrics = self.route_metrics(path, start["projection"], end["projection"])
                if metrics is None:
                    continue

                plans.append({
                    "start_uid": start["uid"],
                    "end_uid": end["uid"],
                    "path": path,
                    "origin_projection": start["projection"],
                    "destination_projection": end["projection"],
                    "start_snap_dist_m": start["snap_dist_m"],
                    "end_snap_dist_m": end["snap_dist_m"],
                    "dijkstra_cost_m": dijkstra_cost,
                    "metrics": metrics,
                    "tried_count": 0,
                })

        if not plans:
            raise RuntimeError("起点和终点的候选车道之间没有可行路径。请检查拓扑或连接阈值。")

        best = min(plans, key=lambda item: item["metrics"]["objective_m"])
        best["tried_count"] = len(plans)
        return best, plans

    def visible_lane_points(self, path, uid, index, origin_projection, destination_projection):
        lane = self.lanes[uid]
        is_first = index == 0
        is_last = index == len(path) - 1

        if is_first and is_last:
            return clip_polyline_by_offsets(lane["points"], origin_projection["offset_m"], destination_projection["offset_m"])
        if is_first:
            return clip_polyline_by_offsets(lane["points"], origin_projection["offset_m"], lane["length_m"])
        if is_last:
            return clip_polyline_by_offsets(lane["points"], 0.0, destination_projection["offset_m"])
        return list(lane["points"])

    def visible_points(self, plan):
        result = []
        path = plan["path"]
        for index, uid in enumerate(path):
            points = self.visible_lane_points(
                path,
                uid,
                index,
                plan["origin_projection"],
                plan["destination_projection"],
            )
            result.extend(points)
        return result
