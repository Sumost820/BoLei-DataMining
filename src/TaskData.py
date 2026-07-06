import json
from pathlib import Path

from LanePlanner import make_point, haversine_m


# 自定义异常
class TaskNotFoundError(RuntimeError):
    pass

class HistoryRouteNotFoundError(RuntimeError):
    pass


# 数据格式
TASK_ID_FIELD = "task_id"
SN_FIELD = "sn"
ROUTE_FIELD = "rt_message"
LON_FIELD = "longitude"
LAT_FIELD = "latitude"
TIME_FIELD = "received_at"


# 读取 JSON 文件 - 解析任务记录
def load_json_records(file_path):
    with file_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    # 列表格式
    if isinstance(data, list):
        records = []
        for item in data:
            if isinstance(item, dict):
                records.append(item)
        return records
    # 兼容单个字典格式
    if isinstance(data, dict):
        return [data]
    return []


# ===================================点和轨迹解析=================================
# 解析 rt_message 中的一条 GPS 记录
def parse_point(item):
    lon = float(item.get(LON_FIELD))
    lat = float(item.get(LAT_FIELD))
    timestamp = str(item.get(TIME_FIELD))
    return make_point(lon, lat, timestamp=timestamp)


# 删去连续的基本不动的点
def deduplicate_consecutive(points):
    result = []
    for point in points:
        if result:
            last = result[-1]
            same_lon = abs(last["lon"] - point["lon"]) < 1e-12
            same_lat = abs(last["lat"] - point["lat"]) < 1e-12
            if same_lon and same_lat:
                continue
        result.append(point)
    return result


# 从任务记录的 rt_message 字段中提取历史轨迹
def extract_history_points(record):
    messages = record.get(ROUTE_FIELD)
    points = []
    for item in messages:
        point = parse_point(item)
        if point is not None:
            points.append(point)
    return deduplicate_consecutive(points)


# ==================================任务查找=====================================
# 判断一条任务记录是否与给定的sn和task_id匹配
def record_matches(record, sn, task_id):
    record_task_id = record.get(TASK_ID_FIELD)
    record_sn = record.get(SN_FIELD)
    if str(record_task_id).strip() != str(task_id).strip() or str(record_sn).strip() != str(sn).strip():
        return False
    return True


# 查找给定的sn和task_id匹配的任务记录
def find_task(sn, task_id, gps_data_dir):
    gps_data_dir = Path(gps_data_dir)

    if not gps_data_dir.exists():
        raise FileNotFoundError("GPS 数据目录不存在：" + str(gps_data_dir))

    files = sorted(gps_data_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError("GPS 数据目录中没有 JSON：" + str(gps_data_dir))

    parse_errors = []

    for file_path in files:
        try:
            records = load_json_records(file_path)
        except Exception as exc:
            parse_errors.append(file_path.name + ": " + str(exc))
            continue

        for record in records:
            if not record_matches(record, sn, task_id):
                continue

            history = extract_history_points(record)

            # 验证轨迹点数量
            if len(history) < 2:
                raise HistoryRouteNotFoundError(
                    "已找到任务，但无法解析出至少两个历史轨迹点。来源文件：" + str(file_path)
                )

            return {"source_file": file_path, "record": record, "history_points": history}

    # 未找到匹配的任务记录
    detail = ""
    if parse_errors:
        detail = "\n部分 JSON 解析失败：\n" + "\n".join(parse_errors[:5])
    raise TaskNotFoundError(
        "没有找到 SN=%r、task_id=%r 的任务。\n已搜索：%s%s"
        % (sn, task_id, gps_data_dir, detail)
    )


# 起终点
def determine_origin_destination(history_points):
    return history_points[0], history_points[-1], "历史轨迹首尾点"


def history_length_m(points):
    total = 0.0
    for i in range(len(points) - 1):
        total += haversine_m(points[i], points[i + 1])
    return total