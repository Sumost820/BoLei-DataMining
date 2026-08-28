
# -*- coding: utf-8 -*-
"""
随机森林：JiangYi 任务时间 / 能耗超参数寻优

设计原则：
1. 数据严格按 actual_start_time 排序。
2. 最后 20% 作为最终测试集，超参数寻优阶段完全不使用。
3. 前 80% 内部使用 expanding-window TimeSeriesSplit。
4. 数值缺失填补、类别 One-Hot 都在每个 CV 训练折内 fit，避免预处理泄漏。
5. 时间和能耗分别建立 Optuna study，各自得到最佳超参数。
6. 最终保存最佳参数、trial 记录、测试指标和完整 sklearn Pipeline。

注意：
- 输入/输出文件地址已集中放在脚本开头的“0. 文件路径配置”中。
- 当前脚本直接读取“已经生成 OD 聚类特征”的 JSON。
- 如果 start_zone/end_zone/od_zone_pair 是基于完整 outer-train 预计算的，
  内部 CV 会存在轻微的无监督特征侧前视。最终 20% 测试评估仍是隔离的。
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple


# 限制底层线程，避免 Optuna trial 与模型内部并行叠加
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 0. 文件路径配置：通常只需要修改这里
# ============================================================
# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_AREA = "TianChi"

# 输入：已经加入 OD / 层次聚类新特征的 JSON
DATA_FILE = PROJECT_ROOT / "data" / TARGET_AREA / f"任务特征和标签-{TARGET_AREA}_层次聚类.json"

# 输出：该模型的超参数寻优结果
OUTPUT_DIR = PROJECT_ROOT / "target" / "HPO_RandomForest"

MODEL_NAME = "RandomForest"
TIME_TARGET = "task_duration_min"
ENERGY_TARGET = "total_energy_soc_delta_pct"
TEST_SIZE = 0.20

# 原始非“相似路线”特征
TIME_CORE_NUMERIC_FEATURE_COLS = [
    "straight_line_distance_m",
    "planned_total_distance_m",
    "planned_to_straight_distance_ratio",
    "endpoint_altitude_change_m",
    "planned_slope_mean",
    "planned_cumulative_ascent_m",
    "planned_cumulative_descent_m",
    "vehicle_recent10_duration_mean_sec",
    "vehicle_time_gap_since_prev_task_min",
    "vehicle_today_speed_mean_mps",
]

ENERGY_CORE_NUMERIC_FEATURE_COLS = [
    "straight_line_distance_m",
    "planned_total_distance_m",
    "planned_to_straight_distance_ratio",
    "endpoint_altitude_change_m",
    "planned_slope_mean",
    "planned_cumulative_ascent_m",
    "planned_cumulative_descent_m",
    "vehicle_recent5_energy_mean_pct",
    "vehicle_time_gap_since_prev_task_min",
    "vehicle_today_speed_mean_mps",
]

# 依赖历史相似任务搜索的特征
TIME_SIMILAR_HISTORY_FEATURE_COLS = [
    "vehicle_similar_top5_duration_mean_sec",
    "similar_top5_duration_mean_sec",
    "similar_top5_duration_std_sec",
    "similar_nearest_od_distance_m",
    "similar_nearest_time_gap_min",
]

ENERGY_SIMILAR_HISTORY_FEATURE_COLS = [
    "vehicle_similar_top5_energy_mean_pct",
    "similar_top5_energy_mean_pct",
    "similar_top5_energy_std_pct",
    "similar_nearest_od_distance_m",
    "similar_nearest_time_gap_min",
]

# 新增 OD / 空间特征
OD_DISPLACEMENT_FEATURE_COLS = [
    "od_delta_east_m",
    "od_delta_north_m",
]

OD_MIDPOINT_FEATURE_COLS = [
    # "od_midpoint_longitude",
    # "od_midpoint_latitude",
]

OD_ZONE_DISTANCE_FEATURE_COLS = [
    "start_zone_distance_m",
    "end_zone_distance_m",
]

OD_CATEGORICAL_FEATURE_COLS = [
    "start_zone_id",
    "end_zone_id",
    "od_zone_pair",
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} 时间/能耗 Optuna 超参数寻优")
    parser.add_argument("--trials", type=int, default=60, help="每个目标的 Optuna trial 数。")
    parser.add_argument("--cv-splits", type=int, default=4, help="前 80%% 内 TimeSeriesSplit 折数。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4, help="单个模型内部并行线程数。")
    parser.add_argument(
        "--od-feature-mode",
        choices=["none", "displacement", "geo", "zones", "all"],
        default="all",
    )
    parser.add_argument(
        "--use-similar-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 similar_* / vehicle_similar_* 历史相似特征。",
    )
    return parser.parse_args()


def get_od_feature_columns(mode: str) -> Tuple[List[str], List[str]]:
    if mode == "none":
        return [], []
    if mode == "displacement":
        return OD_DISPLACEMENT_FEATURE_COLS.copy(), []
    if mode == "geo":
        return OD_DISPLACEMENT_FEATURE_COLS + OD_MIDPOINT_FEATURE_COLS, []
    if mode == "zones":
        return OD_ZONE_DISTANCE_FEATURE_COLS.copy(), OD_CATEGORICAL_FEATURE_COLS.copy()
    if mode == "all":
        return (
            OD_DISPLACEMENT_FEATURE_COLS
            + OD_MIDPOINT_FEATURE_COLS
            + OD_ZONE_DISTANCE_FEATURE_COLS
        ), OD_CATEGORICAL_FEATURE_COLS.copy()
    raise ValueError(mode)


def build_feature_config(use_similar_history: bool, od_mode: str):
    od_num, od_cat = get_od_feature_columns(od_mode)
    time_num = (
        TIME_CORE_NUMERIC_FEATURE_COLS
        + (TIME_SIMILAR_HISTORY_FEATURE_COLS if use_similar_history else [])
        + od_num
    )
    energy_num = (
        ENERGY_CORE_NUMERIC_FEATURE_COLS
        + (ENERGY_SIMILAR_HISTORY_FEATURE_COLS if use_similar_history else [])
        + od_num
    )
    return {
        "time_num": time_num,
        "time_cat": od_cat.copy(),
        "energy_num": energy_num,
        "energy_cat": od_cat.copy(),
        "time_all": time_num + od_cat,
        "energy_all": energy_num + od_cat,
    }


def load_dataset(
    json_path: Path,
    feature_config: Dict[str, List[str]],
) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("JSON 顶层必须是任务数组。")

    df = pd.DataFrame(raw)
    all_features = list(dict.fromkeys(
        feature_config["time_all"] + feature_config["energy_all"]
    ))
    all_numeric = list(dict.fromkeys(
        feature_config["time_num"] + feature_config["energy_num"]
    ))
    all_categorical = list(dict.fromkeys(
        feature_config["time_cat"] + feature_config["energy_cat"]
    ))

    required = {
        "task_id", "actual_start_time", "actual_end_time",
        TIME_TARGET, ENERGY_TARGET, *all_features,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError("JSON 缺少字段：\n" + "\n".join(missing))

    df["actual_start_time"] = pd.to_datetime(
        df["actual_start_time"], format="mixed", errors="coerce"
    )
    df["actual_end_time"] = pd.to_datetime(
        df["actual_end_time"], format="mixed", errors="coerce"
    )

    for col in list(dict.fromkeys(all_numeric + [TIME_TARGET, ENERGY_TARGET])):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in all_categorical:
        df[col] = df[col].astype("string")
        df.loc[df[col].isin(["<NA>", "nan", "None"]), col] = pd.NA

    df = df.dropna(subset=["task_id", "actual_start_time"])
    df = df.drop_duplicates(subset="task_id", keep="first")
    df = df.sort_values(
        ["actual_start_time", "actual_end_time", "task_id"]
    ).reset_index(drop=True)

    required_current = [
        "planned_total_distance_m",
        "endpoint_altitude_change_m",
        "planned_slope_mean",
        "planned_cumulative_ascent_m",
    ]
    df = df.dropna(subset=required_current).copy()
    return df


def chronological_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    cut = int(len(df) * (1.0 - test_size))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    if train.empty or test.empty:
        raise ValueError("训练集或测试集为空。")
    if train["actual_start_time"].max() > test["actual_start_time"].min():
        raise AssertionError("时间切分异常：训练集晚于测试集。")
    return train, test


def make_onehot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )


def make_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    transformers = [
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ]),
            numeric_cols,
        )
    ]
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_onehot_encoder()),
                ]),
                categorical_cols,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rel_err = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "Within10%": float(np.mean(rel_err <= 0.10)),
    }



def suggest_params(trial: optuna.Trial, seed: int, n_jobs: int) -> Dict:
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=50),
        "max_depth": trial.suggest_categorical(
            "max_depth", [None, 6, 8, 10, 12, 16, 20, 24, 30]
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_float("max_features", 0.30, 1.00),
        "bootstrap": bootstrap,
        "random_state": seed,
        "n_jobs": n_jobs,
    }
    if bootstrap:
        params["max_samples"] = trial.suggest_float("max_samples", 0.60, 1.00)
    else:
        params["max_samples"] = None
    return params


def complete_params(params: Dict, seed: int, n_jobs: int) -> Dict:
    full = {
        "random_state": seed,
        "n_jobs": n_jobs,
    }
    full.update(params)
    if not full.get("bootstrap", True):
        full["max_samples"] = None
    return full


def make_model_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Dict,
    params_are_complete: bool = True,
) -> Pipeline:
    estimator = RandomForestRegressor(**params)
    return Pipeline([
        ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
        ("model", estimator),
    ])



def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    cv_splits: int,
    seed: int,
    n_jobs: int,
):
    splitter = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, seed=seed, n_jobs=n_jobs)
        fold_rmse = []

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
            X_fit = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_fit = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            pipeline = make_model_pipeline(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                params=params,
            )
            pipeline.fit(X_fit, y_fit)
            pred = np.maximum(pipeline.predict(X_val), 0.0)
            rmse = float(mean_squared_error(y_val, pred) ** 0.5)
            fold_rmse.append(rmse)

            trial.report(float(np.mean(fold_rmse)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmse))

    return objective


def optimize_target(
    target_label: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    output_dir: Path,
    n_trials: int,
    cv_splits: int,
    seed: int,
    n_jobs: int,
):
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(float).copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(float).copy()

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(5, min(10, n_trials // 5)),
        n_warmup_steps=2,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{MODEL_NAME}_{target_label}",
    )

    objective = make_objective(
        X=X_train,
        y=y_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cv_splits=cv_splits,
        seed=seed,
        n_jobs=n_jobs,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params.copy()
    # 把固定参数补回去，形成可直接复现实验的完整模型参数
    full_best_params = complete_params(best_params, seed=seed, n_jobs=n_jobs)

    final_pipeline = make_model_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        params=full_best_params,
        params_are_complete=True,
    )
    final_pipeline.fit(X_train, y_train)
    pred = np.maximum(final_pipeline.predict(X_test), 0.0)
    metrics = regression_metrics(y_test, pred)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(
        output_dir / f"{target_label}_optuna_trials.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (output_dir / f"{target_label}_best_params.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_cv_rmse": float(study.best_value),
                "best_trial_number": int(study.best_trial.number),
                "best_params": full_best_params,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pred_df = pd.DataFrame({
        "task_id": test_df["task_id"].values,
        "actual_start_time": test_df["actual_start_time"].astype(str).values,
        "y_true": y_test.values,
        "y_pred": pred,
        "abs_error": np.abs(pred - y_test.values),
    })
    pred_df.to_csv(
        output_dir / f"{target_label}_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        {
            "model_name": MODEL_NAME,
            "target": target_col,
            "pipeline": final_pipeline,
            "feature_columns": feature_cols,
            "numeric_feature_columns": numeric_cols,
            "categorical_feature_columns": categorical_cols,
            "best_params": full_best_params,
            "best_cv_rmse": float(study.best_value),
            "test_metrics": metrics,
        },
        output_dir / f"{target_label}_best_model.joblib",
    )

    print(f"\n[{target_label}] 最佳 CV RMSE: {study.best_value:.6f}")
    print(f"[{target_label}] 最佳参数：")
    print(json.dumps(full_best_params, ensure_ascii=False, indent=2))
    print(f"[{target_label}] 最终测试指标：")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    return {
        "target": target_label,
        "best_cv_rmse": float(study.best_value),
        **metrics,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data_file = DATA_FILE.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()

    if not data_file.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{data_file}\n"
            "请修改脚本开头的 PROJECT_ROOT 或 DATA_FILE。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_config = build_feature_config(
        use_similar_history=args.use_similar_history,
        od_mode=args.od_feature_mode,
    )
    df = load_dataset(data_file, feature_config)

    time_df = df.dropna(subset=[TIME_TARGET]).copy()
    time_df = time_df[time_df[TIME_TARGET] >= 0].reset_index(drop=True)
    energy_df = df.dropna(subset=[ENERGY_TARGET]).copy()
    energy_df = energy_df[energy_df[ENERGY_TARGET] >= 0].reset_index(drop=True)

    time_train, time_test = chronological_split(time_df)
    energy_train, energy_test = chronological_split(energy_df)

    print("=" * 88)
    print(f"模型：{MODEL_NAME}")
    print(f"数据：{data_file}")
    print(f"输出：{output_dir}")
    print(f"相似历史特征：{args.use_similar_history}")
    print(f"OD 特征模式：{args.od_feature_mode}")
    print(f"Trials / target：{args.trials}")
    print(f"CV splits：{args.cv_splits}")
    print(
        f"时间：train={len(time_train)} / test={len(time_test)} | "
        f"{time_train['actual_start_time'].min()} -> {time_test['actual_start_time'].max()}"
    )
    print(
        f"能耗：train={len(energy_train)} / test={len(energy_test)} | "
        f"{energy_train['actual_start_time'].min()} -> {energy_test['actual_start_time'].max()}"
    )
    print("=" * 88)

    time_result = optimize_target(
        target_label="time",
        train_df=time_train,
        test_df=time_test,
        feature_cols=feature_config["time_all"],
        numeric_cols=feature_config["time_num"],
        categorical_cols=feature_config["time_cat"],
        target_col=TIME_TARGET,
        output_dir=output_dir,
        n_trials=args.trials,
        cv_splits=args.cv_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    energy_result = optimize_target(
        target_label="energy",
        train_df=energy_train,
        test_df=energy_test,
        feature_cols=feature_config["energy_all"],
        numeric_cols=feature_config["energy_num"],
        categorical_cols=feature_config["energy_cat"],
        target_col=ENERGY_TARGET,
        output_dir=output_dir,
        n_trials=args.trials,
        cv_splits=args.cv_splits,
        seed=args.seed + 1000,
        n_jobs=args.n_jobs,
    )

    summary = pd.DataFrame([time_result, energy_result])
    summary.to_csv(
        output_dir / "final_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "model_name": MODEL_NAME,
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "test_size": TEST_SIZE,
        "cv_splits": args.cv_splits,
        "trials_per_target": args.trials,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "use_similar_history": args.use_similar_history,
        "od_feature_mode": args.od_feature_mode,
        "time_features": feature_config["time_all"],
        "energy_features": feature_config["energy_all"],
        "time_train_count": int(len(time_train)),
        "time_test_count": int(len(time_test)),
        "energy_train_count": int(len(energy_train)),
        "energy_test_count": int(len(energy_test)),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n最终结果：")
    print(summary.round(4).to_string(index=False))
    print(f"\n所有结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
