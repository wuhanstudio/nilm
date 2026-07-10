#!/usr/bin/env python3
"""Five-appliance Han-compatible active-cycle event benchmark helper.

This benchmark helper is self-contained for the Round 8D-A H0/M6 reproduction
path. Report mode reads bundled result tables. Smoke mode performs a tiny
standalone sanity check. H0/M6 mode implements P2 event pairing, F3 features,
train-only 8-bit booleanization, and a one-vs-rest TM wrapper.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import math
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
DOC_ROOT = PACKAGE_ROOT / "docs" / "round8d_five_appliance_event_benchmark"
TABLE_ROOT = DOC_ROOT / "tables"
APPLIANCES = ["fridge", "microwave", "dish washer", "electric furnace", "washer dryer"]
FEATURES_F3 = [
    "pos_transition_magnitude",
    "neg_transition_magnitude",
    "abs_transition",
    "log_abs_transition",
    "duration",
    "log_duration",
    "transition_duration_product",
    "transition_duration_ratio",
    "episode_mean_main",
    "episode_std_main",
    "episode_min_main",
    "episode_max_main",
    "episode_range_main",
    "internal_diff_mean_abs",
    "internal_diff_max_abs",
    "internal_edge_count",
    "subcycle_count_proxy",
    "active_fraction_proxy",
]
EXPECTED_H0 = {
    "accuracy": 0.9411,
    "macro_f1": 0.9259,
    "fridge_f1": 0.9663,
    "microwave_f1": 0.9909,
    "dish_washer_f1": 0.8408,
    "electric_furnace_f1": 0.8900,
    "washer_dryer_f1": 0.9415,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-appliance Han-compatible event benchmark helper")
    parser.add_argument("--mode", choices=["report", "smoke", "h0-m6", "model-size-estimate", "full"], default="report")
    parser.add_argument("--data-dir", default=os.environ.get("NILM_REDD_DATA_DIR", ""), help="Converted REDD CSV directory")
    parser.add_argument("--output-dir", default="", help="Optional directory for compact reproduction tables")
    parser.add_argument("--export-model-dir", default="", help="Optional explicit directory for protobuf/header model export")
    parser.add_argument("--max-rows", type=int, default=50000, help="Maximum rows to read per smoke CSV")
    parser.add_argument("--dry-run", action="store_true", help="Validate H0/M6 workload without training TMs")
    return parser.parse_args()


def archived_metrics() -> pd.DataFrame:
    path = TABLE_ROOT / "model_event_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing bundled metrics table: {path}")
    return pd.read_csv(path)


def best_row(metrics: pd.DataFrame, protocol: str) -> pd.Series:
    part = metrics[metrics["protocol"].astype(str) == protocol].copy()
    if part.empty:
        raise RuntimeError(f"No metrics for protocol {protocol}")
    return part.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]


def app_key(app: str) -> str:
    return app.replace(" ", "_")


def print_row(row: pd.Series) -> None:
    print(f"{row['protocol']} best model: {row['model']}")
    print(f"  accuracy: {float(row['accuracy']):.4f}")
    print(f"  macro F1: {float(row['macro_f1']):.4f}")
    print("  per-appliance F1:")
    for app in APPLIANCES:
        print(f"    {app}: {float(row[f'{app_key(app)}_f1']):.4f}")


def mode_report() -> None:
    metrics = archived_metrics()
    print("Round 8D-A five-appliance Han-compatible active-cycle event benchmark")
    print_row(best_row(metrics, "H0_random_event_split"))
    print()
    print_row(best_row(metrics, "H1_H3_heldout_reference"))
    print()
    print("Caveat: reference-only; H0/H1 use appliance-derived matched transitions.")
    print("This is not deployable aggregate-main NILM performance.")


def require_data_dir(data_dir: str) -> Path:
    if not data_dir:
        raise SystemExit("Set NILM_REDD_DATA_DIR or pass --data-dir for data-dependent modes.")
    path = Path(data_dir)
    if not path.exists():
        raise SystemExit("The provided REDD data directory does not exist.")
    return path


def output_dir_or_temp(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="round8d_h0_m6_"))


def parse_redd_name(path: Path) -> Tuple[Optional[int], Optional[int]]:
    match = re.match(r"redd_house(\d+)_(\d+)\.csv$", path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def discover_csvs(data_dir: Path) -> List[Tuple[Path, int, int]]:
    files: List[Tuple[Path, int, int]] = []
    for name in sorted(glob.glob(str(data_dir / "redd_house*_*.csv"))):
        path = Path(name)
        house, chunk = parse_redd_name(path)
        if house is not None and chunk is not None:
            files.append((path, house, chunk))
    if not files:
        raise SystemExit("No redd_house*_*.csv files found.")
    return files


def column_manifest(files: Sequence[Tuple[Path, int, int]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path, house, chunk in files:
        columns = {str(c) for c in pd.read_csv(path, nrows=0).columns}
        row: Dict[str, Any] = {
            "file": path.name,
            "house": int(house),
            "chunk": int(chunk),
        }
        for app in APPLIANCES:
            row[f"has_{app_key(app)}"] = app in columns
        rows.append(row)
    return pd.DataFrame(rows)


def validate_required_columns(files: Sequence[Tuple[Path, int, int]]) -> pd.DataFrame:
    manifest = column_manifest(files)
    missing_apps = []
    for app in APPLIANCES:
        col = f"has_{app_key(app)}"
        if col not in manifest.columns or int(manifest[col].sum()) == 0:
            missing_apps.append(app)
    if missing_apps:
        raise SystemExit("Required appliance columns were not found: " + ", ".join(missing_apps))
    return manifest


def numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)


def safe_slice(values: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, int(start))
    end = min(len(values) - 1, int(end))
    if end < start:
        return np.asarray([], dtype=np.float64)
    return values[start : end + 1]


def d1_edges(values: np.ndarray, pos_thr: float = 50.0, neg_thr: float = -50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = pd.Series(values).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    delta = np.diff(clean, prepend=clean[0])
    pos = np.flatnonzero(delta >= pos_thr).astype(np.int64)
    neg = np.flatnonzero(delta <= neg_thr).astype(np.int64)
    return pos, neg, delta


def p2_nearest_opposite_pairing(values: np.ndarray, max_duration: int = 20000) -> List[Dict[str, Any]]:
    pos, neg, delta = d1_edges(values)
    episodes: List[Dict[str, Any]] = []
    used_neg: set[int] = set()
    neg_list = [int(i) for i in neg]
    n = len(values)
    for start in [int(i) for i in pos]:
        candidates = [idx for idx in neg_list if idx > start and idx not in used_neg and idx - start <= max_duration]
        if not candidates:
            continue
        end = min(candidates, key=lambda idx: idx - start)
        used_neg.add(end)
        episodes.append(
            {
                "start": start,
                "end": min(end, n - 1),
                "duration": max(1, min(end, n - 1) - start + 1),
                "pos_delta": float(delta[start]),
                "neg_delta": float(delta[end]),
            }
        )
    return episodes


def episode_feature_row(values: np.ndarray, episode: Mapping[str, Any]) -> Dict[str, float]:
    start = int(episode["start"])
    end = int(episode["end"])
    ep = safe_slice(values, start, end)
    pre = safe_slice(values, start - 32, start - 1)
    post = safe_slice(values, end + 1, end + 32)
    pre_mean = float(np.nanmean(pre)) if len(pre) else 0.0
    post_mean = float(np.nanmean(post)) if len(post) else 0.0
    ep_mean = float(np.nanmean(ep)) if len(ep) else 0.0
    ep_max = float(np.nanmax(ep)) if len(ep) else 0.0
    ep_min = float(np.nanmin(ep)) if len(ep) else 0.0
    ep_std = float(np.nanstd(ep)) if len(ep) else 0.0
    baseline = pre_mean if len(pre) else ep_min
    filled = pd.Series(ep).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64) if len(ep) else np.asarray([], dtype=np.float64)
    delta = np.diff(filled, prepend=filled[0]) if len(filled) else np.asarray([], dtype=np.float64)
    duration = float(episode["duration"])
    pos_mag = float(episode["pos_delta"])
    neg_mag = float(abs(episode["neg_delta"]))
    abs_transition = 0.5 * (pos_mag + neg_mag)
    ep_range = ep_max - ep_min
    diffs = np.abs(np.diff(ep)) if len(ep) > 1 else np.asarray([], dtype=np.float64)
    edge_threshold = max(1.0, 0.25 * abs_transition)
    internal_edge_count = int(np.sum(diffs >= edge_threshold)) if len(diffs) else 0
    active_fraction = float(np.mean(ep >= (ep_min + 0.25 * ep_range))) if len(ep) and ep_range > 0 else 0.0
    return {
        "pos_transition_magnitude": pos_mag,
        "neg_transition_magnitude": neg_mag,
        "abs_transition": float(abs_transition),
        "log_abs_transition": float(math.log1p(abs_transition)),
        "duration": duration,
        "log_duration": float(math.log1p(duration)),
        "transition_duration_product": float(abs_transition * max(1.0, duration)),
        "transition_duration_ratio": float(abs_transition / max(1.0, duration)),
        "episode_mean_main": ep_mean,
        "episode_std_main": ep_std,
        "episode_min_main": ep_min,
        "episode_max_main": ep_max,
        "episode_range_main": ep_range,
        "internal_diff_mean_abs": float(np.mean(diffs)) if len(diffs) else 0.0,
        "internal_diff_max_abs": float(np.max(diffs)) if len(diffs) else 0.0,
        "internal_edge_count": internal_edge_count,
        "subcycle_count_proxy": int(max(0, internal_edge_count - 1)),
        "active_fraction_proxy": active_fraction,
        "episode_energy_estimate": float(np.nansum(np.maximum(ep - baseline, 0.0))) if len(ep) else 0.0,
        "post_minus_pre_mean": post_mean - pre_mean,
        "event_internal_edge_count": float(np.count_nonzero(np.abs(delta) >= 50.0)) if len(delta) else 0.0,
    }


def load_reference_events(data_dir: Path) -> pd.DataFrame:
    files = discover_csvs(data_dir)
    validate_required_columns(files)
    rows: List[Dict[str, Any]] = []
    houses = sorted({house for _path, house, _chunk in files})
    for house in houses:
        house_files = sorted([(path, chunk) for path, h, chunk in files if h == house], key=lambda item: item[1])
        parts: Dict[str, List[np.ndarray]] = {app: [] for app in APPLIANCES}
        for path, _chunk in house_files:
            header = [str(c) for c in pd.read_csv(path, nrows=0).columns]
            usecols = [app for app in APPLIANCES if app in header]
            if not usecols:
                continue
            df = pd.read_csv(path, usecols=usecols)
            for app in usecols:
                parts[app].append(numeric(df[app]))
        for app in APPLIANCES:
            if not parts[app]:
                continue
            values = np.concatenate(parts[app])
            episodes = p2_nearest_opposite_pairing(values)
            for idx, ep in enumerate(episodes):
                row: Dict[str, Any] = {
                    "episode_id": f"H{house}:{app_key(app)}:{idx}",
                    "house": int(house),
                    "true_label": app,
                    "start": int(ep["start"]),
                    "end": int(ep["end"]),
                    "duration": int(ep["duration"]),
                    "reference_only": True,
                }
                row.update(episode_feature_row(values, ep))
                rows.append(row)
    if not rows:
        raise RuntimeError("No appliance-derived reference episodes were generated.")
    return pd.DataFrame(rows).sort_values(["house", "true_label", "start", "end"]).reset_index(drop=True)


def print_h0_m6_workload(events: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("H0/M6 workload validation:")
    print(f"  target appliances: {', '.join(APPLIANCES)}")
    print(f"  feature columns: {', '.join(FEATURES_F3)}")
    print("  split: H0 stratified random event split, seed 6072, test fraction 0.30")
    print("  event pairing: appliance-derived P2 nearest-opposite, +50 W / -50 W, max duration 20000 samples")
    print("  model: one-vs-rest TM, n_clause=200, n_state=50, T=20, s=6.0, epochs=20")
    print(f"  generated events: {len(events)}")
    print(f"  train events: {len(train)}")
    print(f"  test events: {len(test)}")
    print("  events by appliance:")
    counts = events["true_label"].value_counts().reindex(APPLIANCES, fill_value=0)
    for app, count in counts.items():
        print(f"    {app}: {int(count)}")


def stratified_random_split(df: pd.DataFrame, test_fraction: float = 0.30, seed: int = 6072) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for _label, group in df.groupby("true_label", sort=False):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue
        test_n = max(1, int(round(len(idx) * test_fraction)))
        test_n = min(test_n, len(idx) - 1)
        test_idx.extend(idx[:test_n].tolist())
        train_idx.extend(idx[test_n:].tolist())
    train = df.loc[train_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = df.loc[test_idx].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train, test


def paper_8bit_fit(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std[std == 0] = 1.0
    train_z = np.clip((np.nan_to_num(train_x, nan=0.0) - mean) / (3.0 * std), -1.0, 1.0)
    scaled = np.rint((train_z + 1.0) * 127.5)
    minv = np.nanmin(scaled, axis=0)
    maxv = np.nanmax(scaled, axis=0)
    maxv[maxv == minv] = minv[maxv == minv] + 1.0
    return mean, std, minv, maxv


def paper_8bit_apply(x: np.ndarray, fit: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std, minv, maxv = fit
    z = np.clip((np.nan_to_num(x, nan=0.0) - mean) / (3.0 * std), -1.0, 1.0)
    scaled = np.rint((z + 1.0) * 127.5)
    scaled = np.clip((scaled - minv) / (maxv - minv), 0.0, 1.0)
    byte_values = np.rint(scaled * 255.0).astype(np.uint8)
    return np.unpackbits(byte_values, axis=1).astype(np.uint8)


def resolve_tsetlin_class() -> Tuple[type, str]:
    errors = []
    search_roots = [Path.cwd(), PACKAGE_ROOT]
    search_roots.extend(PACKAGE_ROOT.parents)
    for root in search_roots:
        if (root / "tsetlin").exists():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
    for module_name in ["tsetlin", "tsetlin.tsetlin"]:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, "Tsetlin")
            return cls, f"{module_name}.Tsetlin"
        except Exception as exc:
            errors.append(f"{module_name}: {exc!r}")
    raise RuntimeError("Tsetlin implementation not found or incompatible. Tried: " + "; ".join(errors))


def instantiate_tm(tm_cls: type, n_features: int, params: Mapping[str, Any]) -> Any:
    try:
        return tm_cls(N_feature=n_features, N_class=2, N_clause=params["n_clause"], N_state=params["n_state"])
    except Exception as exc:
        raise RuntimeError(f"Tsetlin implementation not found or incompatible: constructor failed with {exc!r}") from exc


def train_binary_tm(tm_cls: type, x: np.ndarray, y: np.ndarray, params: Mapping[str, Any], seed: int) -> Any:
    if not hasattr(tm_cls, "__call__"):
        raise RuntimeError("Tsetlin implementation not found or incompatible")
    rng = np.random.default_rng(seed)
    model = instantiate_tm(tm_cls, x.shape[1], params)
    if not hasattr(model, "step") or not hasattr(model, "predict"):
        raise RuntimeError("Tsetlin implementation not found or incompatible: expected step() and predict().")
    for _epoch in range(int(params["epochs"])):
        for idx in rng.permutation(len(x)):
            random.seed(int(rng.integers(0, 2**31 - 1)))
            model.step(x[idx], int(y[idx]), T=float(params["T"]), s=float(params["s"]))
    return model


def tm_scores(model: Any, x: np.ndarray) -> np.ndarray:
    try:
        _pred, votes = model.predict(x, return_votes=True)
        arr = np.asarray(votes, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1] - arr[:, 0]
    except Exception:
        pass
    try:
        return np.asarray(model.predict(x), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"Tsetlin implementation not found or incompatible: predict failed with {exc!r}") from exc


def export_inference_model(model: Any, app: str, export_dir: str) -> List[str]:
    if not export_dir:
        return []
    if not hasattr(model, "save_model"):
        raise RuntimeError("Tsetlin implementation not found or incompatible: expected save_model() for export.")
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = app_key(app)
    pb_path = out_dir / f"m6_ovr_{key}_inference.pb"
    model.save_model(str(pb_path), type="inference")
    written = [str(pb_path)]

    try:
        from tsetlin.compiler.write import tsetlin_compile
    except Exception as exc:
        print(f"Model protobuf written for {app}; header compile skipped: {exc!r}")
        return written

    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        header_name = f"m6_ovr_{key}_inference.h"
        tsetlin_compile(pb_path.name, header_name)
        written.extend([str(out_dir / "tsetlin_model.h"), str(out_dir / header_name)])
    finally:
        os.chdir(old_cwd)
    return written


def h0_m6_predict(train: pd.DataFrame, test: pd.DataFrame, export_model_dir: str = "") -> Tuple[np.ndarray, str, List[str]]:
    tm_cls, tm_name = resolve_tsetlin_class()
    params = {"n_clause": 200, "n_state": 50, "T": 20, "s": 6.0, "epochs": 20}
    train_x_raw = train[FEATURES_F3].to_numpy(dtype=np.float64)
    test_x_raw = test[FEATURES_F3].to_numpy(dtype=np.float64)
    fit = paper_8bit_fit(train_x_raw)
    train_x = paper_8bit_apply(train_x_raw, fit)
    test_x = paper_8bit_apply(test_x_raw, fit)
    train_y = train["true_label"].astype(str).to_numpy()
    scores: List[np.ndarray] = []
    exported: List[str] = []
    for offset, app in enumerate(APPLIANCES):
        y = (train_y == app).astype(np.uint8)
        if int(y.sum()) == 0:
            scores.append(np.full(len(test), -np.inf, dtype=np.float64))
            continue
        model = train_binary_tm(tm_cls, train_x, y, params, seed=6072 + offset * 101)
        scores.append(tm_scores(model, test_x))
        exported.extend(export_inference_model(model, app, export_model_dir))
    matrix = np.vstack(scores).T
    pred = np.asarray([APPLIANCES[int(i)] for i in np.argmax(matrix, axis=1)], dtype=object)
    return pred, tm_name, exported


def binary_counts(truth: np.ndarray, pred: np.ndarray, label: str) -> Tuple[int, int, int, int]:
    tp = int(np.count_nonzero((truth == label) & (pred == label)))
    fp = int(np.count_nonzero((truth != label) & (pred == label)))
    fn = int(np.count_nonzero((truth == label) & (pred != label)))
    tn = int(np.count_nonzero((truth != label) & (pred != label)))
    return tp, fp, fn, tn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def metric_row(protocol: str, model: str, truth: np.ndarray, pred: np.ndarray, note: str) -> Dict[str, Any]:
    per: Dict[str, Tuple[float, float, float]] = {}
    for app in APPLIANCES:
        tp, fp, fn, _tn = binary_counts(truth, pred, app)
        per[app] = prf(tp, fp, fn)
    row: Dict[str, Any] = {
        "protocol": protocol,
        "model": model,
        "accuracy": float(np.mean(truth == pred)) if len(truth) else 0.0,
        "macro_precision": float(np.mean([per[app][0] for app in APPLIANCES])),
        "macro_recall": float(np.mean([per[app][1] for app in APPLIANCES])),
        "macro_f1": float(np.mean([per[app][2] for app in APPLIANCES])),
        "reference_only": True,
        "notes": note,
    }
    for app in APPLIANCES:
        key = app_key(app)
        row[f"{key}_precision"] = per[app][0]
        row[f"{key}_recall"] = per[app][1]
        row[f"{key}_f1"] = per[app][2]
    return row


def confusion_rows(truth: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for true_label in APPLIANCES:
        for pred_label in APPLIANCES:
            rows.append(
                {
                    "protocol": "H0_random_event_split",
                    "model": "M6_Tsetlin_shape_plus_paper_style",
                    "feature_set": "F3_event_shape_plus_reference",
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(np.count_nonzero((truth == true_label) & (pred == pred_label))),
                }
            )
    return pd.DataFrame(rows)


def print_expected_and_observed(row: Mapping[str, Any]) -> None:
    print("Expected archived H0/M6 metrics:")
    for key, value in EXPECTED_H0.items():
        print(f"  {key}: {value:.4f}")
    print("Observed H0/M6 metrics:")
    print(f"  accuracy: {float(row['accuracy']):.4f}")
    print(f"  macro_precision: {float(row['macro_precision']):.4f}")
    print(f"  macro_recall: {float(row['macro_recall']):.4f}")
    print(f"  macro_f1: {float(row['macro_f1']):.4f}")
    for app in APPLIANCES:
        print(f"  {app_key(app)}_f1: {float(row[f'{app_key(app)}_f1']):.4f}")


def mode_model_size_estimate() -> None:
    continuous_features = len(FEATURES_F3)
    bits_per_feature = 8
    boolean_features = continuous_features * bits_per_feature
    literal_count = 2 * boolean_features
    clauses = 200
    ovr_models = len(APPLIANCES)
    n_state = 50
    literal_positions_per_model = literal_count * clauses
    literal_positions_total = literal_positions_per_model * ovr_models
    state_bits = int(math.ceil(math.log2(2 * n_state + 1)))
    packed_state_bytes = int(math.ceil(literal_positions_total * state_bits / 8.0))
    uint8_state_bytes = literal_positions_total
    uint16_state_bytes = literal_positions_total * 2

    print("Round 8D-A M6 model-size estimate")
    print(f"  continuous features: {continuous_features}")
    print(f"  bits per feature: {bits_per_feature}")
    print(f"  boolean features: {boolean_features}")
    print(f"  literal count: {literal_count}")
    print(f"  clauses per OVR model: {clauses}")
    print(f"  OVR models: {ovr_models}")
    print(f"  n_state: {n_state}")
    print(f"  literal positions per model: {literal_positions_per_model}")
    print(f"  total literal positions: {literal_positions_total}")
    print(f"  packed state estimate: {packed_state_bytes} bytes ({packed_state_bytes / 1024.0:.1f} KiB)")
    print(f"  uint8 state estimate: {uint8_state_bytes} bytes ({uint8_state_bytes / 1024.0:.1f} KiB)")
    print(f"  uint16 state estimate: {uint16_state_bytes} bytes ({uint16_state_bytes / 1024.0:.1f} KiB)")
    print("  Caveat: Han paper reports about 18 KB for the simpler transition-duration setting.")
    print("  Round 8D-A M6 is expected to be larger; exact MCU footprint requires future export/compile.")


def mode_smoke(data_dir: str, max_rows: int) -> None:
    root = require_data_dir(data_dir)
    rows: List[Dict[str, Any]] = []
    for path, _house, _chunk in discover_csvs(root)[:3]:
        header = pd.read_csv(path, nrows=0).columns
        usecols = [app for app in APPLIANCES if app in header]
        if not usecols:
            continue
        df = pd.read_csv(path, usecols=usecols, nrows=max_rows)
        for app in usecols:
            values = numeric(df[app])
            for ep in p2_nearest_opposite_pairing(values, max_duration=512)[:20]:
                row = {"true_label": app}
                row.update(episode_feature_row(values, ep))
                rows.append(row)
    events = pd.DataFrame(rows)
    if events.empty or events["true_label"].nunique() < 2:
        raise SystemExit("Smoke mode could not build enough event rows for a sanity classifier.")
    print("Smoke mode completed.")
    print(f"  event rows: {len(events)}")
    print(f"  labels: {', '.join(sorted(events['true_label'].unique()))}")
    print("  This is pipeline sanity only, not headline reproduction.")


def mode_h0_m6(data_dir: str, output_dir: str, dry_run: bool = False, export_model_dir: str = "") -> None:
    root = require_data_dir(data_dir)
    start = time.perf_counter()
    print("Building appliance-derived P2 reference events...")
    events = load_reference_events(root)
    train, test = stratified_random_split(events, test_fraction=0.30, seed=6072)
    print_h0_m6_workload(events, train, test)
    if dry_run:
        missing_features = [name for name in FEATURES_F3 if name not in events.columns]
        if missing_features:
            raise RuntimeError("Missing F3 feature columns: " + ", ".join(missing_features))
        print(f"Dry-run completed in {time.perf_counter() - start:.2f} seconds. No TM training was run.")
        return
    out = output_dir_or_temp(output_dir)
    print(f"Generated events: {len(events)}; train: {len(train)}; test: {len(test)}")
    print("Training M6_Tsetlin_shape_plus_paper_style. This may take significant time.")
    pred, tm_name, exported_models = h0_m6_predict(train, test, export_model_dir=export_model_dir)
    truth = test["true_label"].astype(str).to_numpy()
    row = metric_row(
        "H0_random_event_split",
        "M6_Tsetlin_shape_plus_paper_style",
        truth,
        pred,
        f"self-contained H0/M6 reproduction using {tm_name}; reference-only appliance-derived events",
    )
    metrics = pd.DataFrame([row])
    metrics.to_csv(out / "h0_m6_reproduction_metrics.csv", index=False)
    confusion_rows(truth, pred).to_csv(out / "h0_m6_reproduction_confusion.csv", index=False)
    pd.DataFrame(
        [
            {
                "events": int(len(events)),
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "runtime_seconds": float(time.perf_counter() - start),
                "output_dir": "omitted",
                "notes": "No raw REDD snippets, full per-sample arrays, model binaries, npy/npz, logs, or outputs folders are written.",
            }
        ]
    ).to_csv(out / "h0_m6_reproduction_manifest.csv", index=False)
    print_expected_and_observed(row)
    if exported_models:
        print("Explicit model export requested; generated artifacts:")
        for path in exported_models:
            print(f"  {Path(path).name}")
    print("Compact output tables written to the selected output directory.")
    print("Caveat: reference-only; not deployable aggregate-main NILM.")


def mode_full() -> None:
    print("Full reproduction is intentionally omitted from this candidate script.")
    print("Use --mode h0-m6 for the headline reproduction path; full multi-model reproduction can be added after upstream review.")


def main() -> None:
    args = parse_args()
    if args.mode == "report":
        mode_report()
    elif args.mode == "smoke":
        mode_smoke(args.data_dir, args.max_rows)
    elif args.mode == "h0-m6":
        mode_h0_m6(args.data_dir, args.output_dir, dry_run=args.dry_run, export_model_dir=args.export_model_dir)
    elif args.mode == "model-size-estimate":
        mode_model_size_estimate()
    elif args.mode == "full":
        mode_full()


if __name__ == "__main__":
    main()
