import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev


# ============================================================
# 配置：通常只需要修改这里
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TARGET_AREA = "TianChi"  # "JiangYi" or "TianChi"
SN = "TLE00900VR1450048"
data = "data3"
print(ROOT_DIR)

TASK_FILE = ROOT_DIR / "data" / TARGET_AREA / "GPSdata" / f"{SN}_{data}.json"

# 是否沿用原 validDownloadTask 的规则：
# 当前任务必须是 102，并且按 actual_start_time 排序后的“下一个任务”不能是 104。
# True  = 完全沿用原 validDownloadTask 的筛选口径
# False = 只要 task_type_id == 102 就保留
EXCLUDE_102_FOLLOWED_BY_104 = True

# 3σ 异常值倍数
SIGMA_MULTIPLIER = 3.0

# 输出位置。None 表示与原始 JSON 放在同一目录。
OUTPUT_DIR = ROOT_DIR / "data" / TARGET_AREA / "CheckedData"


# ============================================================
# 时间工具
# ============================================================
def parse_time(value):
    """解析 ISO 风格时间，如 2026-06-24 17:35:21.039000。"""
    if not value:
        return None

    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def task_duration_seconds(task):
    """返回 actual_end_time - actual_start_time 的秒数；无法解析时返回 None。"""
    start = parse_time(task.get("actual_start_time"))
    end = parse_time(task.get("actual_end_time"))

    if start is None or end is None:
        return None

    return (end - start).total_seconds()


# ============================================================
# 读取与任务排序
# ============================================================
def load_tasks(path: Path):
    """兼容顶层为 list、{'tasks': [...]} 或单任务 dict 的 JSON。"""
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return data["tasks"]

    if isinstance(data, dict):
        return [data]

    raise ValueError("无法识别 JSON 结构：顶层必须是任务列表或任务字典。")


def task_sort_key(task):
    """沿用 validDownloadTask：按开始时间、结束时间、task_id 排序。"""
    return (
        task.get("actual_start_time") or "",
        task.get("actual_end_time") or "",
        str(task.get("task_id")),
    )


# ============================================================
# 任务内部 rt_message 时间排序
# ============================================================
def rt_message_sort_key(message):
    """有合法 received_at 的记录按时间排序；缺失/非法时间放到最后。"""
    dt = parse_time(message.get("received_at"))
    return (dt is None, dt or datetime.max)


def sort_task_rt_message(task):
    """
    将单个任务的 rt_message 按 received_at 升序排列。
    返回 (处理后的任务, 是否发生过顺序调整)。
    """
    gps = task.get("rt_message")

    if not isinstance(gps, list) or len(gps) < 2:
        return task, False

    sorted_gps = sorted(gps, key=rt_message_sort_key)
    changed = any(a is not b for a, b in zip(gps, sorted_gps))
    task["rt_message"] = sorted_gps
    return task, changed


# ============================================================
# 删除存在空字段的整个任务
# ============================================================
def is_empty_value(value):
    """
    判断字段值是否为空。

    以下情况都视为“空”：
    1) None（JSON 中的 null）
    2) 空字符串或只包含空白字符的字符串
    3) 空列表 []
    4) 空字典 {}

    对非空 dict/list 会继续递归检查其内部字段，
    因此嵌套结构中只要出现空值，也会判定为无效。
    """
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, dict):
        if not value:
            return True
        return any(is_empty_value(item) for item in value.values())

    if isinstance(value, list):
        if not value:
            return True
        return any(is_empty_value(item) for item in value)

    return False


def task_has_empty_field(task):
    """
    判断整个任务是否存在空字段。

    删除条件：
    1) 与 rt_message 同级的任意原始任务字段为空；
    2) rt_message 缺失、不是 list 或为空列表；
    3) rt_message 中任意 GPS 记录的任意字段为空；
    4) GPS 记录中的嵌套 dict/list 内部出现空值。

    注意：
    select_102_tasks() 后新增的 _order_index、_next_task_* 等辅助字段
    不参与空值判断，避免辅助字段为 None 时误删任务。
    """
    if not isinstance(task, dict):
        return True

    # 检查与 rt_message 同级的原始任务字段。
    for key, value in task.items():
        # 跳过程序后续添加的辅助字段。
        if str(key).startswith("_"):
            continue

        if key == "rt_message":
            continue

        if is_empty_value(value):
            return True

    # rt_message 本身必须存在、必须是非空列表。
    gps = task.get("rt_message")
    if not isinstance(gps, list) or len(gps) == 0:
        return True

    # 每条 GPS 必须是非空字典，且内部任意字段都不能为空。
    for message in gps:
        if not isinstance(message, dict) or len(message) == 0:
            return True

        if is_empty_value(message):
            return True

    return False


def remove_tasks_with_empty_fields(rows):
    """删除任务层或 rt_message 内部存在任意空字段的整个任务。"""
    kept = []
    removed = []

    for row in rows:
        if task_has_empty_field(row["task"]):
            removed.append(row)
        else:
            kept.append(row)

    return kept, removed


# ============================================================
# 第一步：提取 102 有效任务，并整理任务内部时间顺序
# ============================================================
def select_102_tasks(tasks):
    tasks_sorted = sorted(tasks, key=task_sort_key)

    selected = []
    task_102_count = 0
    rt_message_reordered_task_count = 0

    for idx, task in enumerate(tasks_sorted):
        next_task = tasks_sorted[idx + 1] if idx + 1 < len(tasks_sorted) else None

        current_type = str(task.get("task_type_id"))
        next_type = None if next_task is None else str(next_task.get("task_type_id"))

        is_102 = current_type == "102"
        if is_102:
            task_102_count += 1

        if EXCLUDE_102_FOLLOWED_BY_104:
            should_select = is_102 and (next_task is None or next_type != "104")
        else:
            should_select = is_102

        if not should_select:
            continue

        # 浅复制任务本身；rt_message 会被替换为新的排序列表，不改原任务列表顺序。
        item = dict(task)

        # 保留 validDownloadTask 原有的辅助信息。
        item["_order_index"] = idx + 1
        item["_next_task_id"] = None if next_task is None else next_task.get("task_id")
        item["_next_task_type_id"] = None if next_task is None else next_task.get("task_type_id")
        item["_next_task_start_time"] = (
            None if next_task is None else next_task.get("actual_start_time")
        )

        item, changed = sort_task_rt_message(item)
        if changed:
            rt_message_reordered_task_count += 1

        selected.append(
            {
                "task": item,
                "duration_seconds": task_duration_seconds(item),
                "rt_message_reordered": changed,
            }
        )

    return (
        tasks_sorted,
        selected,
        task_102_count,
        rt_message_reordered_task_count,
    )


# ============================================================
# 第二步：按任务时长执行 3σ 异常值删除
# ============================================================
def remove_duration_outliers(selected):
    """
    只删除“异常长”的任务：
        duration > mean + SIGMA_MULTIPLIER * std

    std 使用总体标准差 pstdev；当只有 1 个有效时长时，std = 0。
    无法计算时长的任务不会因为 3σ 规则被删除。
    """
    valid_durations = [
        row["duration_seconds"]
        for row in selected
        if row["duration_seconds"] is not None
    ]

    if not valid_durations:
        return selected, [], None, None, None

    mean_duration = fmean(valid_durations)
    std_duration = pstdev(valid_durations) if len(valid_durations) > 1 else 0.0
    upper_limit = mean_duration + SIGMA_MULTIPLIER * std_duration

    kept = []
    removed = []

    for row in selected:
        duration = row["duration_seconds"]

        if duration is not None and duration > upper_limit:
            removed.append(row)
        else:
            kept.append(row)

    return kept, removed, mean_duration, std_duration, upper_limit


# ============================================================
# 第三步：生成摘要
# ============================================================
def clean_and_build_summary(rows):
    cleaned_tasks = []
    summary_rows = []

    for row in rows:
        task = row["task"]
        duration = row["duration_seconds"]

        cleaned_tasks.append(task)

        summary_rows.append(
            {
                "order_index": task.get("_order_index"),
                "task_id": task.get("task_id"),
                "task_type_id": task.get("task_type_id"),
                "actual_start_time": task.get("actual_start_time"),
                "actual_end_time": task.get("actual_end_time"),
                "duration_seconds": None if duration is None else round(duration, 3),
                "duration_minutes": None if duration is None else round(duration / 60.0, 3),
                "sn": task.get("sn"),
                "rt_message_count": len(task.get("rt_message") or []),
                "rt_message_reordered": row["rt_message_reordered"],
                "next_task_id": task.get("_next_task_id"),
                "next_task_type_id": task.get("_next_task_type_id"),
                "next_task_start_time": task.get("_next_task_start_time"),
            }
        )

    return cleaned_tasks, summary_rows


# ============================================================
# 输出
# ============================================================
def build_output_paths(input_path: Path):
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{SN}_{data}有效卸货记录"
    return out_dir / f"{base_name}.json"


def write_json(path: Path, tasks):
    with path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if rows:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            writer.writerow(["没有筛选到最终有效卸料任务"])


# ============================================================
# 主流程
# ============================================================
def main():
    if not TASK_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件：{TASK_FILE}")

    out_json = build_output_paths(TASK_FILE)

    # 1) 原始数据 -> 提取 102（沿用原 validDownloadTask 条件）
    # 2) 任务内部 rt_message 按 received_at 排序
    tasks = load_tasks(TASK_FILE)
    (
        tasks_sorted,
        selected,
        task_102_count,
        reordered_count,
    ) = select_102_tasks(tasks)

    selected_before_null_check = len(selected)

    # 3) 任务同级字段、rt_message 本身或 GPS 内部任意字段为空，都删除整个任务
    selected, empty_field_tasks = remove_tasks_with_empty_fields(selected)

    # 4) 根据任务持续时间删除异常长任务（mean + 3σ）
    (
        kept,
        duration_outliers,
        mean_duration,
        std_duration,
        upper_limit,
    ) = remove_duration_outliers(selected)

    # 5) 生成最终任务与摘要
    cleaned_tasks, summary_rows = clean_and_build_summary(kept)

    # 6) 输出完整 JSON，形式与 validDownloadTask 类似
    write_json(out_json, cleaned_tasks)

    print("=" * 68)
    print("处理完成")
    print("=" * 68)
    print(f"原始任务数：{len(tasks_sorted)}")
    print(f"102 任务总数：{task_102_count}")
    print(f"按原 validDownloadTask 条件提取数：{selected_before_null_check}")
    print(f"rt_message 顺序发生调整的任务数：{reordered_count}")

    if mean_duration is None:
        print("可计算时长的任务数：0（未执行 3σ 删除）")
    else:
        print(f"平均时长：{mean_duration / 60.0:.3f} 分钟")
        print(f"时长总体标准差：{std_duration / 60.0:.3f} 分钟")
        print(
            f"3σ 异常长阈值：{upper_limit / 60.0:.3f} 分钟 "
            f"(mean + {SIGMA_MULTIPLIER:g} × std)"
        )

    print(f"因时长异常过长删除的任务数：{len(duration_outliers)}")
    if duration_outliers:
        print("删除的异常任务：")
        for row in duration_outliers:
            task_id = row["task"].get("task_id")
            duration = row["duration_seconds"]
            print(
                f"  task_id={task_id}, "
                f"duration={duration / 60.0:.3f} 分钟"
            )

    print(f"因任务层或 GPS 内存在空字段而删除的任务数：{len(empty_field_tasks)}")
    if empty_field_tasks:
        print("因空字段删除的任务：")
        for row in empty_field_tasks:
            print(f"  task_id={row['task'].get('task_id')}")
    print(f"最终有效任务数：{len(cleaned_tasks)}")
    print(f"完整任务 JSON：{out_json}")


if __name__ == "__main__":
    main()