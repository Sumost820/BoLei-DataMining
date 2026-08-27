import json
from pathlib import Path


# ============================================================
# 配置：修改这里
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TARGET_AREA = "JiangYi"  # "JiangYi" or "TianChi"
SN = "TLE00900CS1450457"

INPUT_JSON = ROOT_DIR / "data" / TARGET_AREA / "CheckedData" / f"{SN}最终有效卸货记录.json"
OUTPUT_JSON = ROOT_DIR / "data" / TARGET_AREA / "FinalData" / f"{SN}最终有效卸货记录-删除后.json"

def load_tasks(path: Path):
    """
    读取 JSON。

    支持：
    1) 顶层直接为任务列表 [...]
    2) 顶层为 {"tasks": [...]}

    返回：
        original_data: 原始 JSON 数据
        tasks:         实际任务列表
        mode:          "list" 或 "tasks_dict"
    """
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, data, "list"

    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return data, data["tasks"], "tasks_dict"

    raise ValueError(
        "无法识别 JSON 结构：顶层必须是任务列表，"
        "或者包含 tasks 列表的字典。"
    )


def parse_indices(text: str, max_index: int):
    """
    解析需要删除的任务序号，序号从 1 开始。

    支持：
        1
        1,3,5
        1 3 5
        1, 3, 5
        2-5
        1,3,7-10

    返回：
        已去重、已排序的 1-based 序号列表。
    """
    text = text.strip()

    if not text:
        raise ValueError("没有输入任何任务序号。")

    # 同时兼容中文逗号、英文逗号和空格
    text = text.replace("，", ",")
    parts = []

    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue

        # 如果某段中还有空格，则继续拆分
        parts.extend(piece.split())

    indices = set()

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # 支持范围，例如 3-7
        if "-" in part:
            pieces = part.split("-", 1)

            if len(pieces) != 2:
                raise ValueError(f"无法识别的序号：{part}")

            start_text, end_text = pieces

            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                raise ValueError(f"无法识别的范围：{part}")

            if start > end:
                raise ValueError(
                    f"范围起点不能大于终点：{part}"
                )

            for index in range(start, end + 1):
                indices.add(index)

        else:
            try:
                indices.add(int(part))
            except ValueError:
                raise ValueError(f"无法识别的序号：{part}")

    if not indices:
        raise ValueError("没有解析到有效任务序号。")

    invalid = [
        index
        for index in indices
        if index < 1 or index > max_index
    ]

    if invalid:
        raise IndexError(
            f"任务序号超出范围：{invalid}。"
            f"当前文件共有 {max_index} 个任务，"
            f"有效序号为 1 ~ {max_index}。"
        )

    return sorted(indices)


def delete_tasks_by_indices(tasks, indices):
    """
    按 1-based 序号删除任务。

    例如：
        indices = [1]
    表示删除 tasks[0]，也就是第 1 个任务。
    """
    delete_set = set(indices)

    removed = []
    kept = []

    for index, task in enumerate(tasks, start=1):
        if index in delete_set:
            removed.append((index, task))
        else:
            kept.append(task)

    return kept, removed


def save_result(original_data, cleaned_tasks, mode, output_path: Path):
    """
    按原始 JSON 顶层结构保存结果。
    """
    if mode == "list":
        output_data = cleaned_tasks

    elif mode == "tasks_dict":
        output_data = dict(original_data)
        output_data["tasks"] = cleaned_tasks

    else:
        raise ValueError(f"未知模式：{mode}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=2
        )


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_JSON.resolve()}"
        )

    original_data, tasks, mode = load_tasks(INPUT_JSON)

    print("=" * 60)
    print("按任务序号删除")
    print("=" * 60)
    print(f"输入文件：{INPUT_JSON}")
    print(f"当前任务总数：{len(tasks)}")
    print()
    print("序号从 1 开始。")
    print("例如：")
    print("  输入 1       -> 删除第 1 个任务")
    print("  输入 1,3,5   -> 删除第 1、3、5 个任务")
    print("  输入 2-5     -> 删除第 2 到第 5 个任务")
    print("  输入 1,3,7-9 -> 删除第 1、3、7、8、9 个任务")
    print()

    user_input = input("请输入要删除的任务序号：")

    indices = parse_indices(
        user_input,
        len(tasks)
    )

    cleaned_tasks, removed = delete_tasks_by_indices(
        tasks,
        indices
    )

    print()
    print("准备删除以下任务：")

    for index, task in removed:
        task_id = (
            task.get("task_id")
            if isinstance(task, dict)
            else None
        )

        start_time = (
            task.get("actual_start_time")
            if isinstance(task, dict)
            else None
        )

        print(
            f"  第 {index} 个任务"
            f" | task_id={task_id}"
            f" | start={start_time}"
        )

    print()
    confirm = input(
        f"确认删除以上 {len(removed)} 个任务？"
        f"输入 y 确认，其它任意键取消："
    ).strip().lower()

    if confirm not in {"y", "yes"}:
        print("已取消，没有修改或生成文件。")
        return

    save_result(
        original_data,
        cleaned_tasks,
        mode,
        OUTPUT_JSON
    )

    print()
    print("=" * 60)
    print("删除完成")
    print("=" * 60)
    print(f"原任务数：{len(tasks)}")
    print(f"删除任务数：{len(removed)}")
    print(f"剩余任务数：{len(cleaned_tasks)}")
    print(f"输出文件：{OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()