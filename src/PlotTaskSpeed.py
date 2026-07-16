#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
从 GPSdata 目录中按代码内配置的 task_id 和 sn 查找任务，
提取 rt_message 中的 received_at 与 speed，并输出速度变化图。

使用方式：
    1. 修改下方“用户配置区”中的 TASK_ID 和 SN；
    2. 直接运行：
       python PlotTaskSpeed.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 用户配置区
# =============================================================================

# 需要绘图的任务
TASK_ID = 8102543468839169
SN = "TLE00860VR1450004"

# GPS任务JSON目录
DATA_DIR = Path(
    r"C:\Users\14993\PycharmProjects\BoLei-DataMining\data\GPSdata"
)

# 图片输出目录
OUTPUT_DIR = Path(
    r"C:\Users\14993\PycharmProjects\BoLei-DataMining\target\SpeedPlots"
)

# 速度单位：
#   "mps" -> m/s
#   "kmh" -> km/h
SPEED_UNIT = "mps"

# 移动平均窗口点数：
#   15 -> 显示15点移动平均
#   1  -> 不平滑
SMOOTH_WINDOW = 15

# 横轴：
#   False -> 任务开始后的分钟数
#   True  -> 实际GPS时间
SHOW_ACTUAL_TIME = False

# 图片分辨率
OUTPUT_DPI = 160

# 支持扫描的文件后缀
SUPPORTED_SUFFIXES = {".json", ".jsonw"}


# =============================================================================
# JSON与时间处理
# =============================================================================

def parse_datetime(value: Any) -> Optional[datetime]:
    """解析 received_at、actual_start_time 等时间字段。"""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    )

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def normalize_tasks(payload: Any) -> list[dict[str, Any]]:
    """兼容顶层任务列表、{"tasks": [...]}和单任务字典。"""
    if isinstance(payload, list):
        return [
            item
            for item in payload
            if isinstance(item, dict)
        ]

    if isinstance(payload, dict):
        tasks = payload.get("tasks")

        if isinstance(tasks, list):
            return [
                item
                for item in tasks
                if isinstance(item, dict)
            ]

        if (
            "task_id" in payload
            and "rt_message" in payload
        ):
            return [payload]

    return []


def load_json_tasks(
    file_path: Path,
) -> list[dict[str, Any]]:
    with file_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        payload = json.load(file)

    return normalize_tasks(payload)


def task_id_equal(
    value: Any,
    target_task_id: int | str,
) -> bool:
    """兼容JSON中task_id为整数或字符串。"""
    if value is None:
        return False

    value_text = str(value).strip()
    target_text = str(target_task_id).strip()

    if value_text == target_text:
        return True

    try:
        return int(value_text) == int(target_text)
    except (TypeError, ValueError):
        return False


# =============================================================================
# 文件扫描与任务定位
# =============================================================================

def iter_data_files(
    data_dir: Path,
    sn: str,
) -> Iterable[Path]:
    """
    优先扫描文件名包含sn的JSON/JSONW，
    再扫描目录中的其余文件。
    """
    files = sorted(
        path
        for path in data_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    )

    sn_lower = sn.lower()

    preferred_files = [
        path
        for path in files
        if sn_lower in path.name.lower()
    ]
    remaining_files = [
        path
        for path in files
        if sn_lower not in path.name.lower()
    ]

    yield from preferred_files
    yield from remaining_files


def find_task(
    data_dir: Path,
    task_id: int | str,
    sn: str,
) -> tuple[dict[str, Any], Path, int]:
    """按 task_id + sn 精确定位任务。"""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"GPS数据目录不存在：{data_dir}"
        )

    scanned_file_count = 0
    parse_errors: list[str] = []

    for file_path in iter_data_files(
        data_dir,
        sn,
    ):
        scanned_file_count += 1

        try:
            tasks = load_json_tasks(file_path)
        except Exception as error:
            parse_errors.append(
                f"{file_path}: {error}"
            )
            continue

        for task in tasks:
            task_sn = str(
                task.get("sn", "")
            ).strip()

            if (
                task_sn == sn
                and task_id_equal(
                    task.get("task_id"),
                    task_id,
                )
            ):
                return (
                    task,
                    file_path,
                    scanned_file_count,
                )

    message = (
        f"没有找到 task_id={task_id!r} "
        f"且 sn={sn!r} 的任务。\n"
        f"已扫描文件数：{scanned_file_count}\n"
        f"数据目录：{data_dir}"
    )

    if parse_errors:
        message += (
            f"\n解析失败文件数："
            f"{len(parse_errors)}"
            f"\n首个错误：{parse_errors[0]}"
        )

    raise LookupError(message)


# =============================================================================
# 速度序列提取
# =============================================================================

def extract_speed_series(
    task: dict[str, Any],
) -> tuple[list[datetime], np.ndarray]:
    """
    从rt_message中提取有效received_at和speed。
    重复时间戳保留最后一条。
    """
    messages = task.get("rt_message")

    if not isinstance(messages, list):
        raise ValueError(
            "任务缺少有效的rt_message列表。"
        )

    time_to_speed: dict[
        datetime,
        float,
    ] = {}

    for message in messages:
        if not isinstance(message, dict):
            continue

        timestamp = parse_datetime(
            message.get("received_at")
        )
        if timestamp is None:
            continue

        try:
            speed = float(
                message.get("speed")
            )
        except (TypeError, ValueError):
            continue

        if not math.isfinite(speed):
            continue

        if speed < 0:
            continue

        time_to_speed[timestamp] = speed

    if not time_to_speed:
        raise ValueError(
            "rt_message中没有有效的"
            "received_at和speed记录。"
        )

    ordered = sorted(
        time_to_speed.items(),
        key=lambda item: item[0],
    )

    timestamps = [
        item[0]
        for item in ordered
    ]
    speeds = np.asarray(
        [
            item[1]
            for item in ordered
        ],
        dtype=float,
    )

    return timestamps, speeds


def moving_average(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """计算边缘自适应的简单移动平均。"""
    if window <= 1 or len(values) <= 1:
        return values.copy()

    window = min(
        window,
        len(values),
    )

    result = np.empty(
        len(values),
        dtype=float,
    )

    queue: deque[float] = deque()
    running_sum = 0.0

    for index, value in enumerate(values):
        value_float = float(value)
        queue.append(value_float)
        running_sum += value_float

        if len(queue) > window:
            running_sum -= queue.popleft()

        result[index] = (
            running_sum / len(queue)
        )

    return result


# =============================================================================
# 绘图
# =============================================================================

def configure_plot_style() -> None:
    """
    显式设置文字颜色，避免深色主题导致
    标题、坐标和图例不可见。
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#D0D5DD",
        "axes.labelcolor": "#344054",
        "axes.titlecolor": "#101828",
        "text.color": "#101828",
        "xtick.color": "#475467",
        "ytick.color": "#475467",
        "legend.labelcolor": "#344054",
        "axes.grid": True,
        "grid.color": "#D0D5DD",
        "grid.alpha": 0.45,
        "grid.linestyle": "--",
        "axes.unicode_minus": False,
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
    })


def safe_filename(text: str) -> str:
    invalid_chars = '<>:"/\\|?*'

    filename = "".join(
        "_"
        if char in invalid_chars
        else char
        for char in text
    )

    return (
        filename.strip().strip(".")
        or "speed_plot"
    )


def plot_task_speed(
    task: dict[str, Any],
    source_file: Path,
) -> Path:
    timestamps, speeds_mps = (
        extract_speed_series(task)
    )

    if SPEED_UNIT == "kmh":
        speeds = speeds_mps * 3.6
        y_label = "速度（km/h）"
        speed_unit_text = "km/h"
    elif SPEED_UNIT == "mps":
        speeds = speeds_mps
        y_label = "速度（m/s）"
        speed_unit_text = "m/s"
    else:
        raise ValueError(
            "SPEED_UNIT只能设置为"
            '"mps"或"kmh"。'
        )

    if SMOOTH_WINDOW < 1:
        raise ValueError(
            "SMOOTH_WINDOW必须大于等于1。"
        )

    smooth_speed = moving_average(
        speeds,
        SMOOTH_WINDOW,
    )

    task_start = parse_datetime(
        task.get("actual_start_time")
    )
    if task_start is None:
        task_start = timestamps[0]

    elapsed_minutes = np.asarray([
        (
            timestamp - task_start
        ).total_seconds() / 60.0
        for timestamp in timestamps
    ])

    task_id = str(
        task.get("task_id", "unknown")
    )
    sn = str(
        task.get("sn", "unknown")
    )

    configure_plot_style()

    fig, ax = plt.subplots(
        figsize=(12.0, 6.4)
    )

    if SHOW_ACTUAL_TIME:
        x_values = timestamps
        x_label = "GPS时间"
    else:
        x_values = elapsed_minutes
        x_label = "任务开始后时间（分钟）"

    ax.plot(
        x_values,
        speeds,
        linewidth=1.0,
        alpha=0.38,
        color="#2E90FA",
        label="原始速度",
        zorder=2,
    )

    if SMOOTH_WINDOW > 1:
        ax.plot(
            x_values,
            smooth_speed,
            linewidth=2.0,
            alpha=0.95,
            color="#175CD3",
            label=(
                f"{SMOOTH_WINDOW}点移动平均"
            ),
            zorder=3,
        )

    max_position = int(
        np.argmax(speeds)
    )

    ax.scatter(
        [x_values[max_position]],
        [speeds[max_position]],
        s=52,
        color="#B42318",
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label="最大速度",
    )

    if SHOW_ACTUAL_TIME:
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M:%S")
        )
        fig.autofmt_xdate(rotation=25)

    record_duration_minutes = (
        timestamps[-1] - timestamps[0]
    ).total_seconds() / 60.0

    statistics_text = (
        f"GPS点数：{len(speeds)}\n"
        f"记录跨度："
        f"{record_duration_minutes:.2f} 分钟\n"
        f"平均速度："
        f"{np.mean(speeds):.3f} "
        f"{speed_unit_text}\n"
        f"最小速度："
        f"{np.min(speeds):.3f} "
        f"{speed_unit_text}\n"
        f"最大速度："
        f"{np.max(speeds):.3f} "
        f"{speed_unit_text}"
    )

    ax.text(
        0.985,
        0.965,
        statistics_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#D0D5DD",
            "alpha": 0.92,
        },
    )

    ax.set_title(
        "任务速度变化图\n"
        f"task_id={task_id}，sn={sn}",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(
        x_label,
        fontsize=11,
    )
    ax.set_ylabel(
        y_label,
        fontsize=11,
    )
    ax.legend(
        loc="upper left",
        frameon=False,
    )
    ax.grid(
        True,
        alpha=0.35,
    )
    ax.margins(x=0.01)
    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    fig.text(
        0.01,
        0.01,
        f"数据文件：{source_file}",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#667085",
    )

    fig.tight_layout(
        rect=(0, 0.035, 1, 1)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_filename = safe_filename(
        f"speed_task_{task_id}"
        f"_sn_{sn}_{SPEED_UNIT}.png"
    )
    output_path = (
        OUTPUT_DIR / output_filename
    )

    fig.savefig(
        output_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    return output_path


# =============================================================================
# 主程序
# =============================================================================

def main() -> int:
    try:
        task, source_file, scanned_count = (
            find_task(
                data_dir=DATA_DIR,
                task_id=TASK_ID,
                sn=SN,
            )
        )

        output_path = plot_task_speed(
            task=task,
            source_file=source_file,
        )

    except Exception as error:
        print(
            f"[错误] {error}",
            file=sys.stderr,
        )
        return 1

    print("任务定位成功。")
    print(f"task_id：{task.get('task_id')}")
    print(f"sn：{task.get('sn')}")
    print(f"扫描文件数：{scanned_count}")
    print(f"来源文件：{source_file}")
    print(f"速度图：{output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
