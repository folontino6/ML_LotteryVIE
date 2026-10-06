#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_bingo18.py - Ensemble thống kê + Machine Learning chuyên cho Bingo18
========================================================================

CẢNH BÁO QUAN TRỌNG
    Bingo18 là quá trình ngẫu nhiên (3 viên xúc xắc 1..6). KHÔNG có mô hình nào đảm
    bảo dự đoán đúng. Mô hình luôn được so sánh với các baseline đơn giản; nếu không
    vượt được baseline một cách có ý nghĩa thống kê thì chương trình sẽ nói rõ điều đó.

ĐỊNH DẠNG CSV (cố định)
    draw_id,n1,n2,n3,sum
    1,2,5,6,13
    2,1,3,4,8
    Chỉ dùng 5 cột này. n1..n3 trong 1..6, sum = n1 + n2 + n3 (3..18).

MỤC TIÊU (TARGET)
    Phân phối xác suất của TỔNG (3..18) ở kỳ kế tiếp = bài toán phân loại 16 lớp.
    Lý do chọn tổng thay vì xác suất từng số 1..6:
      * Tổng là đại lượng Bingo18 thực sự trả thưởng và không phụ thuộc thứ tự n1,n2,n3.
      * Có phân phối lý thuyết chính xác (tích chập 3 xúc xắc) làm thành phần nền;
        phân tích dữ liệu cho thấy tổng khớp phân phối này (xem README/báo cáo).
      * Đo bằng log-loss/Brier trên cả 16 lớp nên phát hiện được cả những lệch nhỏ.

QUY TRÌNH
    load_data -> validate_data -> create_features -> build_dataset (chia theo thời gian
    80/10/10) -> dự báo thống kê + ML -> tối ưu ensemble trên validation ->
    đánh giá trên test giữ riêng -> dự đoán kỳ kế tiếp

CHỐNG RÒ RỈ DỮ LIỆU (data leakage)
    * Đặc trưng của kỳ t chỉ dùng các kỳ < t (rolling rồi shift(1)).
    * Chia theo thứ tự thời gian, không xáo trộn: train 80% -> validation 10% -> test 10%.
    * check_no_leakage() tự kiểm chứng: sửa dữ liệu tương lai không làm đổi đặc trưng quá khứ.
    * Baseline tần suất cũng chỉ dùng các kỳ xảy ra trước tập test.

CÁCH CHẠY
    pip install numpy pandas scikit-learn scipy joblib
    python ml_bingo18.py                              # train + evaluate + predict
    python ml_bingo18.py --csv data/bingo18.csv --mode train
    python ml_bingo18.py --csv data/bingo18.csv --mode predict
    python ml_bingo18.py --csv data/bingo18.csv --mode validate

TỐI ƯU CHO CPU + RAM (Ryzen 5, 16 GB): không dùng GPU; dữ liệu ~56k dòng x 52 đặc trưng
float32 (vài MB); Gradient Boosting dạng histogram của scikit-learn chạy đa luồng trên CPU.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Hiển thị tiếng Việt đúng trên console Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from scipy.optimize import minimize
except ImportError as _exc:  # pragma: no cover
    print(
        f"[LỖI] Thiếu thư viện Python: '{_exc.name}'. "
        "Hãy cài đặt bằng lệnh: pip install numpy pandas scikit-learn scipy joblib"
    )
    sys.exit(1)

# ----------------------------------------------------------------------------
# Hằng số Bingo18
# ----------------------------------------------------------------------------
REQUIRED_COLUMNS = ["draw_id", "n1", "n2", "n3", "sum"]
FACES = (1, 2, 3, 4, 5, 6)
SUM_MIN, SUM_MAX = 3, 18
SUMS = list(range(SUM_MIN, SUM_MAX + 1))
N_CLASSES = len(SUMS)  # 16
WINDOWS = (10, 20, 50)
MAX_WINDOW = max(WINDOWS)
TREND_WINDOW = 200
MIN_DRAWS = 500
ROLLING_BASELINE_WINDOW = 100
RECENCY_DECAY_DRAWS = 500
FREQUENCY_PRIOR_STRENGTH = 24.0
MARKOV_PRIOR_STRENGTH = 32.0
MIN_ENSEMBLE_WEIGHT = 0.005
HGB_MAX_DEPTH = 3
BIG_THRESHOLD = 11  # tổng >= 11: "lớn", <= 10: "nhỏ"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILENAME = "mo_hinh_bingo18.joblib"
MODEL_NAMES = ["hist_gradient_boosting", "random_forest"]
MODEL_VI = {
    "hist_gradient_boosting": "Gradient Boosting (histogram)",
    "random_forest": "Random Forest",
}
STATISTICAL_COMPONENTS = [
    "theoretical_dice",
    "historical_frequency",
    "recent_frequency",
    "exponential_trend",
    "sum_transition",
]
COMPONENT_LABELS = {
    "theoretical_dice": "phân phối lý thuyết 3 xúc xắc",
    "historical_frequency": "tần suất toàn lịch sử",
    "recent_frequency": "tần suất 100 kỳ gần nhất",
    "exponential_trend": "tần suất suy giảm theo thời gian",
    "sum_transition": "phụ thuộc chuyển tiếp tổng",
    "hist_gradient_boosting": MODEL_VI["hist_gradient_boosting"],
    "random_forest": MODEL_VI["random_forest"],
}

WARNING_VI = (
    "Đây là công cụ thử nghiệm thống kê/học máy. Bingo18 là quá trình ngẫu nhiên; "
    "không có phương pháp nào đảm bảo dự đoán đúng. Chỉ dùng cho mục đích tham khảo và học tập."
)


def _theory_pmf() -> np.ndarray:
    """Phân phối lý thuyết của tổng 3 xúc xắc công bằng (16 giá trị, tổng = 1)."""
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for a, b, c in itertools.product(FACES, repeat=3):
        counts[a + b + c - SUM_MIN] += 1
    return counts / counts.sum()


THEORY_PMF = _theory_pmf()
THEORY_ENTROPY = float(-(THEORY_PMF * np.log(THEORY_PMF)).sum())  # log-loss kỳ vọng nếu xúc xắc công bằng


def _build_feature_names() -> List[str]:
    names: List[str] = []
    for w in WINDOWS:
        names += [f"freq{k}_w{w}" for k in FACES]
        names += [
            f"sum_mean_w{w}", f"sum_max_w{w}", f"sum_min_w{w}",
            f"rate_double_w{w}", f"rate_triple_w{w}", f"odd_mean_w{w}", f"low_mean_w{w}",
        ]
    names += ["sum_last", "sum_prev1", "sum_prev2", "last_repeated", "last_odd", "last_low"]
    names += [
        "n1_last", "n2_last", "n3_last",
        "same_position_count_last", "sum_delta_last",
        "sum_trend_w10", "sum_trend_w50",
    ]
    return names


FEATURE_NAMES = _build_feature_names()
N_FEATURES = len(FEATURE_NAMES)


class DataError(Exception):
    """Lỗi dữ liệu / cấu hình có thông báo tiếng Việt cho người dùng."""


def log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------
@dataclass
class Config:
    csv_file: str
    mode: str = "full"
    output_dir: str = "output"
    model_path: str = ""
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    max_iter: int = 300
    learning_rate: float = 0.03
    n_estimators: int = 150
    max_depth: int = 8
    min_samples_leaf: int = 100
    n_jobs: int = -1
    no_save_model: bool = False
    debug: bool = False


# ----------------------------------------------------------------------------
# Tiện ích ghi file
# ----------------------------------------------------------------------------
def _to_builtin(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    raise TypeError(f"Không thể ghi JSON kiểu {type(obj)}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_to_builtin)
    log("LƯU", f"Đã ghi {path}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig để Excel đọc đúng tiếng Việt
    log("LƯU", f"Đã ghi {path}")


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ----------------------------------------------------------------------------
# 1) Đọc + kiểm tra dữ liệu
# ----------------------------------------------------------------------------
def resolve_csv_file(raw: Optional[str]) -> str:
    """Tìm file CSV: đường dẫn người dùng đưa, hoặc data/bingo18.csv, hoặc bingo18.csv."""
    if raw:
        for cand in (Path(raw), PROJECT_ROOT / raw):
            if cand.is_file():
                return str(cand.resolve())
        raise DataError(f"Không tìm thấy file CSV: {raw}")
    for cand in (PROJECT_ROOT / "data" / "bingo18.csv", PROJECT_ROOT / "bingo18.csv"):
        if cand.is_file():
            return str(cand)
    raise DataError("Không tìm thấy data/bingo18.csv. Hãy cung cấp --csv đường_dẫn_file.csv")


def load_data(path: str) -> pd.DataFrame:
    """Đọc CSV thô (chuỗi), thử nhiều bảng mã. Chưa làm sạch."""
    p = Path(path)
    if not p.is_file():
        raise DataError(f"Không tìm thấy file: {p}")
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            df = pd.read_csv(p, encoding=encoding, dtype=str)
            df.columns = [str(c).strip().lower() for c in df.columns]
            log("DATA", f"Đã đọc '{p.name}': {len(df):,} dòng")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise DataError(f"File '{p.name}' rỗng.") from exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise DataError(f"Không đọc được file '{p.name}': {last_error}")


def validate_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Validate Bingo18 rows and keep only complete, internally consistent draws."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataError(f"Thiếu cột bắt buộc: {', '.join(missing)}. Định dạng: {','.join(REQUIRED_COLUMNS)}")
    df = raw[REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    total_raw = len(df)
    report: dict = {"canh_bao": []}

    numeric = df.to_numpy(dtype=np.float64)
    finite = np.isfinite(numeric).all(axis=1)
    df = df.loc[finite].copy()
    dropped_na = total_raw - len(df)

    values = df[REQUIRED_COLUMNS].to_numpy(dtype=np.float64)
    draw_ids = values[:, 0]
    faces = values[:, 1:4]
    calculated_sums = faces.sum(axis=1)
    totals = values[:, 4]
    invalid_sum_count = int((totals != calculated_sums).sum())
    good = (
        (draw_ids >= 1)
        & (draw_ids <= np.iinfo(np.int64).max)
        & (draw_ids % 1 == 0)
        & np.all((faces % 1 == 0) & (faces >= 1) & (faces <= 6), axis=1)
        & (totals % 1 == 0)
        & (totals >= SUM_MIN)
        & (totals <= SUM_MAX)
        & (totals == calculated_sums)
    )
    dropped_bad = int((~good).sum())
    df = df.loc[good].copy()

    df = df.sort_values("draw_id", kind="mergesort")
    duplicate_id_mask = df.duplicated(subset=["draw_id"], keep="first")
    dropped_dup = int(duplicate_id_mask.sum())
    df = df.loc[~duplicate_id_mask]

    df = df.astype({"draw_id": np.int64, "n1": np.int16, "n2": np.int16, "n3": np.int16, "sum": np.int16})
    if invalid_sum_count:
        msg = f"Loại {invalid_sum_count:,} dòng có sum khác n1+n2+n3."
        report["canh_bao"].append(msg)
        log("CẢNH BÁO", msg)
    gaps = int((df["draw_id"].diff().dropna() != 1).sum())
    if gaps:
        msg = f"draw_id không liên tục ở {gaps:,} chỗ (thiếu kỳ): đặc trưng rolling sẽ nối các kỳ còn lại."
        report["canh_bao"].append(msg)
        log("CẢNH BÁO", msg)

    df = df.reset_index(drop=True)
    n = len(df)
    if n < MIN_DRAWS:
        raise DataError(f"Chỉ có {n:,} kỳ hợp lệ, cần tối thiểu {MIN_DRAWS:,} kỳ để huấn luyện và đánh giá.")
    report.update(
        {
            "tong_dong_doc": int(total_raw),
            "loai_do_thieu_so": int(dropped_na),
            "loai_do_so_khong_hop_le": int(dropped_bad),
            "loai_do_trung_draw_id": int(dropped_dup),
            "dong_sum_sai_bi_loai": invalid_sum_count,
            "cho_thieu_ky": gaps,
            "so_ky_hop_le": int(n),
            "draw_id_dau": int(df["draw_id"].iloc[0]),
            "draw_id_cuoi": int(df["draw_id"].iloc[-1]),
        }
    )
    log(
        "VALIDATION",
        f"Tổng {total_raw:,} dòng | loại: thiếu số {dropped_na:,}, số sai {dropped_bad:,}, trùng draw_id {dropped_dup:,} "
        f"| hợp lệ: {n:,} kỳ (draw_id {report['draw_id_dau']:,} -> {report['draw_id_cuoi']:,})",
    )
    report["thong_ke"] = describe_data(df)
    return df, report


def describe_data(df: pd.DataFrame) -> dict:
    """Kiểm tra dữ liệu có khớp xúc xắc công bằng không (chi-bình phương)."""
    face_counts = np.bincount(df[["n1", "n2", "n3"]].to_numpy().ravel(), minlength=7)[1:]
    sum_counts = np.bincount(df["sum"].to_numpy() - SUM_MIN, minlength=N_CLASSES)
    n = len(df)
    exp_face = np.full(6, 3 * n / 6.0)
    exp_sum = THEORY_PMF * n
    chi_face = float(((face_counts - exp_face) ** 2 / exp_face).sum())
    chi_sum = float(((sum_counts - exp_sum) ** 2 / exp_sum).sum())
    p_face = p_sum = None
    try:
        from scipy import stats as sps  # type: ignore

        p_face = float(sps.chi2.sf(chi_face, 5))
        p_sum = float(sps.chi2.sf(chi_sum, N_CLASSES - 1))
    except Exception:  # noqa: BLE001
        pass
    x = df["sum"].to_numpy(dtype=np.float64) - 10.5
    acf1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    faces = df[["n1", "n2", "n3"]].to_numpy(dtype=np.float64)
    position_corr = np.corrcoef(faces, rowvar=False)
    sum_values = df["sum"].to_numpy(dtype=np.int64) - SUM_MIN
    transitions = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    np.add.at(transitions, (sum_values[:-1], sum_values[1:]), 1.0)
    transition_prob = transitions / max(1.0, transitions.sum())
    previous_prob = transition_prob.sum(axis=1, keepdims=True)
    next_prob = transition_prob.sum(axis=0, keepdims=True)
    independent_prob = previous_prob @ next_prob
    positive = transition_prob > 0
    mutual_information = float(
        np.sum(
            transition_prob[positive]
            * np.log(transition_prob[positive] / independent_prob[positive])
        )
    )
    out = {
        "tan_suat_mat_1_6": face_counts.tolist(),
        "p_value_chi2_mat": p_face,
        "p_value_chi2_tong_vs_ly_thuyet": p_sum,
        "tu_tuong_quan_tong_lag1": acf1,
        "tuong_quan_giua_vi_tri_xuc_xac": position_corr.tolist(),
        "mutual_information_chuyen_tiep_tong_lag1_nats": mutual_information,
        "ghi_chu": "p-value lớn nghĩa là chưa thấy bằng chứng dữ liệu lệch khỏi xúc xắc công bằng.",
    }
    if p_face is not None and p_sum is not None:
        log("THỐNG KÊ", f"Chi-bình phương mặt 1..6: p = {p_face:.4f} | tổng so với lý thuyết: p = {p_sum:.4f} | tự tương quan lag-1 của tổng = {acf1:+.4f}")
    return out


# ----------------------------------------------------------------------------
# 2) Đặc trưng (chỉ dùng dữ liệu của các kỳ trước)
# ----------------------------------------------------------------------------
def _draw_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Chỉ báo của từng kỳ (chưa rolling)."""
    faces = df[["n1", "n2", "n3"]].to_numpy()
    srt = np.sort(faces, axis=1)
    n_unique = 1 + (srt[:, 1] != srt[:, 0]).astype(int) + (srt[:, 2] != srt[:, 1]).astype(int)
    ind = pd.DataFrame(index=df.index)
    for k in FACES:
        ind[f"face{k}"] = (faces == k).sum(axis=1) / 3.0  # tỉ lệ mặt k trong kỳ
    ind["sum"] = df["sum"].to_numpy(dtype=np.float64)
    ind["is_double"] = (n_unique == 2).astype(np.float64)  # đúng 2 số giống nhau
    ind["is_triple"] = (n_unique == 1).astype(np.float64)  # cả 3 số giống nhau
    ind["repeated"] = (3 - n_unique).astype(np.float64)  # số lần lặp trong kỳ: 0, 1, 2
    ind["odd_cnt"] = (faces % 2 == 1).sum(axis=1).astype(np.float64)
    ind["low_cnt"] = (faces <= 3).sum(axis=1).astype(np.float64)  # số "thấp" 1-3 (số "cao" = 3 - thấp)
    for index, column in enumerate(("n1", "n2", "n3")):
        ind[f"{column}_value"] = df[column].to_numpy(dtype=np.float64)
    ind["same_position_count"] = (
        (faces[:, 0] == faces[:, 1]).astype(np.float64)
        + (faces[:, 0] == faces[:, 2]).astype(np.float64)
        + (faces[:, 1] == faces[:, 2]).astype(np.float64)
    )
    return ind


def _rolling_state(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê cuốn chiếu KẾT THÚC TẠI kỳ t (gồm cả kỳ t). Dòng t của create_features = dòng t-1 của đây."""
    ind = _draw_indicators(df)
    cols: Dict[str, pd.Series] = {}
    for w in WINDOWS:
        mean = ind.rolling(w, min_periods=w).mean()
        for k in FACES:
            cols[f"freq{k}_w{w}"] = mean[f"face{k}"]
        cols[f"sum_mean_w{w}"] = mean["sum"]
        cols[f"sum_max_w{w}"] = ind["sum"].rolling(w, min_periods=w).max()
        cols[f"sum_min_w{w}"] = ind["sum"].rolling(w, min_periods=w).min()
        cols[f"rate_double_w{w}"] = mean["is_double"]
        cols[f"rate_triple_w{w}"] = mean["is_triple"]
        cols[f"odd_mean_w{w}"] = mean["odd_cnt"]
        cols[f"low_mean_w{w}"] = mean["low_cnt"]
    cols["sum_last"] = ind["sum"]
    cols["sum_prev1"] = ind["sum"].shift(1)
    cols["sum_prev2"] = ind["sum"].shift(2)
    cols["last_repeated"] = ind["repeated"]
    cols["last_odd"] = ind["odd_cnt"]
    cols["last_low"] = ind["low_cnt"]
    for column in ("n1_value", "n2_value", "n3_value"):
        cols[f"{column[:-6]}_last"] = ind[column]
    cols["same_position_count_last"] = ind["same_position_count"]
    cols["sum_delta_last"] = ind["sum"] - ind["sum"].shift(1)
    mean10 = ind["sum"].rolling(10, min_periods=10).mean()
    mean50 = ind["sum"].rolling(50, min_periods=50).mean()
    cols["sum_trend_w10"] = mean10 - mean50
    cols["sum_trend_w50"] = mean50 - ind["sum"].rolling(
        TREND_WINDOW, min_periods=TREND_WINDOW
    ).mean()
    return pd.DataFrame(cols, index=df.index)[FEATURE_NAMES]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng để dự đoán kỳ t, tính CHỈ từ các kỳ < t (shift(1) chống rò rỉ). Các dòng đầu là NaN."""
    return _rolling_state(df).shift(1)


def next_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng cho kỳ TIẾP THEO (chưa có kết quả): thống kê kết thúc ở kỳ cuối cùng."""
    required_history = max(MAX_WINDOW, TREND_WINDOW)
    if len(df) < required_history:
        raise DataError(f"Cần ít nhất {required_history} kỳ lịch sử để tạo đặc trưng dự đoán.")
    return _rolling_state(df).iloc[[-1]]


def check_no_leakage(df: pd.DataFrame, seed: int = 0) -> None:
    """Kiểm chứng: thay toàn bộ dữ liệu từ kỳ `cut` trở đi thì đặc trưng của các kỳ <= cut KHÔNG đổi."""
    cut = len(df) // 2
    rng = np.random.default_rng(seed)
    alt = df.copy()
    new_faces = rng.integers(1, 7, size=(len(alt) - cut, 3))
    for column_index, column in enumerate(("n1", "n2", "n3")):
        alt.loc[alt.index[cut:], column] = new_faces[:, column_index].astype(np.int16)
    alt["sum"] = (alt["n1"] + alt["n2"] + alt["n3"]).astype(np.int16)
    a = create_features(df).iloc[: cut + 1].to_numpy()
    b = create_features(alt).iloc[: cut + 1].to_numpy()
    if not np.allclose(a, b, equal_nan=True):
        raise DataError("PHÁT HIỆN RÒ RỈ DỮ LIỆU: đặc trưng quá khứ thay đổi khi sửa dữ liệu tương lai.")
    log("LEAKAGE", "Đã kiểm tra: đặc trưng của kỳ t không phụ thuộc dữ liệu từ kỳ t trở đi (OK)")


@dataclass
class Dataset:
    X: np.ndarray  # (m, N_FEATURES) float32, theo thứ tự thời gian
    y: np.ndarray  # (m,) lớp 0..15 (= tổng - 3)
    draw_ids: np.ndarray
    pos: np.ndarray  # vị trí dòng trong df gốc (dùng cho baseline cuốn chiếu)
    y_full: np.ndarray  # lớp của TẤT CẢ các kỳ (gồm cả các kỳ khởi động)
    train: slice
    val: slice
    test: slice

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def build_dataset(df: pd.DataFrame, cfg: Config) -> Dataset:
    """Ghép đặc trưng + nhãn, bỏ các kỳ khởi động (chưa đủ cửa sổ), chia theo thời gian (không xáo trộn)."""
    feats = create_features(df)
    valid = feats.notna().all(axis=1).to_numpy()
    pos = np.flatnonzero(valid)
    X = feats.to_numpy(dtype=np.float32)[pos]
    y_full = (df["sum"].to_numpy().astype(np.int64) - SUM_MIN)
    y = y_full[pos]
    m = len(pos)
    if not (0.0 < cfg.train_frac < 1.0 and 0.0 < cfg.val_frac < 1.0 and cfg.train_frac + cfg.val_frac < 1.0):
        raise DataError("--train-frac và --val-frac phải nằm trong (0,1) và tổng nhỏ hơn 1.")
    i_tr = int(m * cfg.train_frac)
    i_va = int(m * (cfg.train_frac + cfg.val_frac))
    if i_tr < 200 or (i_va - i_tr) < 50 or (m - i_va) < 50:
        raise DataError("Không đủ dữ liệu để chia train/validation/test hợp lý.")
    ds = Dataset(
        X=X, y=y, draw_ids=df["draw_id"].to_numpy()[pos], pos=pos, y_full=y_full,
        train=slice(0, i_tr), val=slice(i_tr, i_va), test=slice(i_va, m),
    )
    log(
        "DATASET",
        f"{m:,} mẫu x {N_FEATURES} đặc trưng (bỏ {len(df) - m} kỳ khởi động) | "
        f"train {i_tr:,} | validation {i_va - i_tr:,} | test {m - i_va:,} (chia theo thời gian, không xáo trộn)",
    )
    return ds


# ----------------------------------------------------------------------------
# 3) Mô hình
# ----------------------------------------------------------------------------
def _align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Đưa xác suất về đủ 16 cột (lớp thiếu trong dữ liệu huấn luyện = 0)."""
    classes = np.asarray(classes, dtype=np.int64)
    if proba.shape[1] == N_CLASSES and np.array_equal(classes, np.arange(N_CLASSES)):
        return proba
    out = np.zeros((proba.shape[0], N_CLASSES), dtype=np.float64)
    out[:, classes] = proba
    return out


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    P = _align_proba(model.predict_proba(X), model.classes_)
    return P / P.sum(axis=1, keepdims=True)


def statistical_forecasts(
    y_full: np.ndarray,
    positions: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Make causal sum-distribution forecasts using only draws before each target."""
    positions = np.asarray(positions, dtype=np.int64)
    forecasts = {
        name: np.zeros((len(positions), N_CLASSES), dtype=np.float64)
        for name in STATISTICAL_COMPONENTS
    }
    theory = THEORY_PMF.astype(np.float64)
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    transitions = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    observed_until = 0

    for row, position in enumerate(positions):
        while observed_until < position:
            current = int(y_full[observed_until])
            counts[current] += 1.0
            if observed_until:
                previous = int(y_full[observed_until - 1])
                transitions[previous, current] += 1.0
            observed_until += 1

        forecasts["theoretical_dice"][row] = theory
        forecasts["historical_frequency"][row] = (
            counts + FREQUENCY_PRIOR_STRENGTH * theory
        ) / (counts.sum() + FREQUENCY_PRIOR_STRENGTH)

        recent = y_full[max(0, position - ROLLING_BASELINE_WINDOW):position]
        recent_counts = np.bincount(recent, minlength=N_CLASSES).astype(np.float64)
        forecasts["recent_frequency"][row] = (
            recent_counts + FREQUENCY_PRIOR_STRENGTH * theory
        ) / (len(recent) + FREQUENCY_PRIOR_STRENGTH)

        recent_for_decay = y_full[max(0, position - RECENCY_DECAY_DRAWS):position]
        if len(recent_for_decay):
            ages = np.arange(len(recent_for_decay) - 1, -1, -1, dtype=np.float64)
            decay_weights = np.exp(-ages / max(1.0, RECENCY_DECAY_DRAWS / 3.0))
            weighted_counts = np.bincount(
                recent_for_decay,
                weights=decay_weights,
                minlength=N_CLASSES,
            )
            forecasts["exponential_trend"][row] = (
                weighted_counts + FREQUENCY_PRIOR_STRENGTH * theory
            ) / (decay_weights.sum() + FREQUENCY_PRIOR_STRENGTH)
        else:
            forecasts["exponential_trend"][row] = theory

        if position:
            previous_sum = int(y_full[position - 1])
            transition_counts = transitions[previous_sum]
            forecasts["sum_transition"][row] = (
                transition_counts + MARKOV_PRIOR_STRENGTH * theory
            ) / (transition_counts.sum() + MARKOV_PRIOR_STRENGTH)
        else:
            forecasts["sum_transition"][row] = theory

    return forecasts


def combine_probabilities(
    components: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """Return a normalized, validation-weighted mixture of forecast distributions."""
    if not components:
        raise DataError("Không có thành phần dự đoán để kết hợp.")
    result = np.zeros_like(next(iter(components.values())), dtype=np.float64)
    total_weight = 0.0
    for name, probabilities in components.items():
        weight = float(weights.get(name, 0.0))
        if weight < 0:
            raise DataError(f"Trọng số dự đoán âm ở thành phần {name}.")
        result += weight * probabilities
        total_weight += weight
    if total_weight <= 0:
        raise DataError("Tổng trọng số dự đoán phải lớn hơn 0.")
    result /= total_weight
    row_sums = result.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(result)) or np.any(row_sums <= 0):
        raise DataError("Phân phối dự đoán không hợp lệ.")
    return result / row_sums


def fit_ensemble_weights(
    components: Dict[str, np.ndarray],
    y: np.ndarray,
) -> Dict[str, float]:
    """Fit nonnegative mixture weights by log-loss on chronological validation only."""
    names = list(components)
    scores = np.stack([
        np.clip(components[name][np.arange(len(y)), y], 1e-12, 1.0)
        for name in names
    ], axis=1)

    def objective(weights: np.ndarray) -> float:
        return float(-np.log(np.clip(scores @ weights, 1e-12, 1.0)).mean())

    def gradient(weights: np.ndarray) -> np.ndarray:
        return -(scores / np.clip(scores @ weights, 1e-12, 1.0)[:, None]).mean(axis=0)

    result = minimize(
        objective,
        np.full(len(names), 1.0 / len(names)),
        jac=gradient,
        method="SLSQP",
        bounds=[(MIN_ENSEMBLE_WEIGHT, 1.0)] * len(names),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0,
                     "jac": lambda weights: np.ones_like(weights)},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise DataError(f"Không tối ưu được trọng số ensemble: {result.message}")
    fitted = np.clip(result.x, MIN_ENSEMBLE_WEIGHT, 1.0)
    fitted /= fitted.sum()
    return dict(zip(names, fitted.tolist()))


def train_model(name: str, X: np.ndarray, y: np.ndarray, cfg: Config, n_iter: Optional[int] = None):
    """Huấn luyện một mô hình trên (X, y). Người gọi đảm bảo X, y chỉ gồm các kỳ trong quá khứ so với tập đánh giá."""
    if name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=cfg.learning_rate,
            max_iter=int(n_iter or cfg.max_iter),
            max_depth=HGB_MAX_DEPTH,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=5.0,
            early_stopping=False,  # early stopping nội bộ của sklearn xáo trộn ngẫu nhiên -> tự chọn theo thời gian
            random_state=cfg.seed,
        )
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features="sqrt",
            n_jobs=cfg.n_jobs,
            random_state=cfg.seed,
        )
    else:
        raise DataError(f"Mô hình không được hỗ trợ: {name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def select_model(ds: Dataset, cfg: Config) -> dict:
    """Fit causal base forecasts and choose mixture weights on validation only."""
    X_tr, y_tr = ds.X[ds.train], ds.y[ds.train]
    X_va, y_va = ds.X[ds.val], ds.y[ds.val]
    val_positions = ds.pos[ds.val]
    component_forecasts = statistical_forecasts(ds.y_full, val_positions)
    best_iters: Dict[str, Optional[int]] = {}

    for name in MODEL_NAMES:
        t0 = time.time()
        log("ML", f"Đang huấn luyện {MODEL_VI[name]} trên {len(y_tr):,} kỳ (train)")
        model = train_model(name, X_tr, y_tr, cfg)
        if name == "hist_gradient_boosting":
            curve = []
            for P in model.staged_predict_proba(X_va):
                P = _align_proba(P, model.classes_)
                P /= P.sum(axis=1, keepdims=True)
                curve.append(float(nll_per_draw(P, y_va).mean()))
            best_iter = int(np.argmin(curve)) + 1
            model = train_model(name, X_tr, y_tr, cfg, best_iter)
            best_iters[name] = best_iter
        else:
            best_iters[name] = None
        component_forecasts[name] = predict_proba(model, X_va)
        log(
            "ML",
            f"  {MODEL_VI[name]}: validation log-loss = "
            f"{nll_per_draw(component_forecasts[name], y_va).mean():.5f} "
            f"| {time.time() - t0:.1f}s",
        )

    weights = fit_ensemble_weights(component_forecasts, y_va)
    ensemble = combine_probabilities(component_forecasts, weights)
    losses = {
        name: float(nll_per_draw(probabilities, y_va).mean())
        for name, probabilities in component_forecasts.items()
    }
    losses["validation_weighted_ensemble"] = float(nll_per_draw(ensemble, y_va).mean())
    log("ENSEMBLE", "Trọng số tối ưu trên validation:")
    for name, weight in weights.items():
        log("ENSEMBLE", f"  {COMPONENT_LABELS.get(name, name)}: {weight:.3f}")
    return {
        "best_name": "validation_weighted_ensemble",
        "best_iters": best_iters,
        "ensemble_weights": weights,
        "minimum_component_weight": MIN_ENSEMBLE_WEIGHT,
        "validation_logloss": losses,
    }


# ----------------------------------------------------------------------------
# 4) Đánh giá: baseline + kiểm định
# ----------------------------------------------------------------------------
def nll_per_draw(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def brier_score(P: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((P - onehot) ** 2).sum(axis=1).mean())


def topk_accuracy(P: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator) -> float:
    jitter = rng.random(P.shape) * 1e-9  # phá hòa ngẫu nhiên (baseline đều có nhiều giá trị bằng nhau)
    top = np.argsort(-(P + jitter), axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def make_baselines(ds: Dataset) -> Dict[str, np.ndarray]:
    """Create causal statistical forecasts for every requested test draw."""
    y_test = ds.y[ds.test]
    n_test = len(y_test)
    test_pos = ds.pos[ds.test]
    forecasts = statistical_forecasts(ds.y_full, test_pos)
    forecasts["uniform_random"] = np.full(
        (n_test, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64
    )
    return forecasts


def summarize_predictions(P: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    nll = nll_per_draw(P, y)
    return {
        "log_loss": float(nll.mean()),
        "brier": brier_score(P, y),
        "top1": topk_accuracy(P, y, 1, rng),
        "top3": topk_accuracy(P, y, 3, rng),
        "_nll": nll,
    }


def evaluate_model(
    models: Dict[str, Any],
    weights: Dict[str, float],
    ds: Dataset,
    cfg: Config,
) -> dict:
    """Evaluate the fixed validation-weighted ensemble on the chronological test tail."""
    rng = np.random.default_rng(cfg.seed)
    y_test = ds.y[ds.test]
    components = make_baselines(ds)
    for name, model in models.items():
        components[name] = predict_proba(model, ds.X[ds.test])
    P_model = combine_probabilities(components, weights)
    table: Dict[str, dict] = {
        "validation_weighted_ensemble": summarize_predictions(P_model, y_test, rng)
    }
    for name, P in components.items():
        table[name] = summarize_predictions(P, y_test, rng)
    n = len(y_test)
    comparisons: Dict[str, dict] = {}
    for name, row in table.items():
        if name == "validation_weighted_ensemble":
            continue
        d = row["_nll"] - table["validation_weighted_ensemble"]["_nll"]
        se = float(d.std(ddof=1) / math.sqrt(n))
        z = float(d.mean() / se) if se > 0 else 0.0
        comparisons[name] = {
            "chenh_lech_log_loss": float(d.mean()),  # > 0: mô hình tốt hơn
            "z": z,
            "p_value_mot_phia": normal_sf(z),
        }
    alpha = 0.05 / max(1, len(comparisons))  # hiệu chỉnh Bonferroni
    theory_cmp = comparisons["theoretical_dice"]
    beats = [k for k, v in comparisons.items() if v["p_value_mot_phia"] < alpha]
    if theory_cmp["p_value_mot_phia"] < alpha:
        verdict = (
            "Mô hình vượt phân phối lý thuyết có ý nghĩa thống kê trên tập test. Cần kiểm chứng lại trên dữ liệu mới "
            "trước khi tin kết quả (nhiều mô hình/cấu hình đã được thử)."
        )
    else:
        verdict = (
            "Mô hình KHÔNG vượt được phân phối lý thuyết của 3 xúc xắc công bằng một cách có ý nghĩa thống kê. "
            "Phù hợp với bản chất ngẫu nhiên: dự đoán tốt nhất vẫn là phân phối lý thuyết."
        )
    result = {
        "so_ky_test": int(n),
        "draw_id_test": [int(ds.draw_ids[ds.test][0]), int(ds.draw_ids[ds.test][-1])],
        "log_loss_ky_vong_neu_xuc_xac_cong_bang": THEORY_ENTROPY,
        "bang_so_sanh": {k: {m: v for m, v in row.items() if not m.startswith("_")} for k, row in table.items()},
        "kiem_dinh_so_voi_baseline": comparisons,
        "baseline_bi_vuot": beats,
        "alpha_bonferroni": alpha,
        "ket_luan": verdict,
    }
    print_evaluation(result)
    return result


def print_evaluation(result: dict) -> None:
    log("KẾT QUẢ", f"Đánh giá trên tập TEST: {result['so_ky_test']:,} kỳ (draw_id {result['draw_id_test'][0]:,} -> {result['draw_id_test'][1]:,})")
    log("KẾT QUẢ", f"Log-loss kỳ vọng nếu xúc xắc công bằng (entropy) = {result['log_loss_ky_vong_neu_xuc_xac_cong_bang']:.5f}; "
                    f"ngẫu nhiên đều = {math.log(N_CLASSES):.5f}")
    print(f"          {'Dự đoán':<42}{'LogLoss':>9}{'Brier':>9}{'Top-1':>8}{'Top-3':>8}{'Δ vs ensemble':>14}{'z':>8}{'p':>8}")
    for name, row in result["bang_so_sanh"].items():
        cmp_ = result["kiem_dinh_so_voi_baseline"].get(name)
        extra = f"{cmp_['chenh_lech_log_loss']:>+14.5f}{cmp_['z']:>+8.2f}{cmp_['p_value_mot_phia']:>8.3f}" if cmp_ else f"{'':>30}"
        label = "ENSEMBLE" if name == "validation_weighted_ensemble" else COMPONENT_LABELS.get(name, name)
        print(f"          {label:<42}{row['log_loss']:>9.5f}{row['brier']:>9.5f}{row['top1']:>8.4f}{row['top3']:>8.4f}{extra}")
    log("KẾT QUẢ", "Δ vs mô hình = log-loss(baseline) - log-loss(mô hình); dương = mô hình tốt hơn. p một phía, ngưỡng "
                    f"Bonferroni = {result['alpha_bonferroni']:.4f}")
    log("KẾT QUẢ", result["ket_luan"])


# ----------------------------------------------------------------------------
# 5) Huấn luyện toàn bộ quy trình + lưu / nạp mô hình
# ----------------------------------------------------------------------------
def run_training(df: pd.DataFrame, cfg: Config, out: Path) -> dict:
    ds = build_dataset(df, cfg)
    selection = select_model(ds, cfg)
    best_iters = selection["best_iters"]
    ensemble_weights = selection["ensemble_weights"]
    trval_end = ds.val.stop
    log("ML", f"Huấn luyện các mô hình trên train+validation ({trval_end:,} kỳ) để chấm điểm tập test")
    eval_models = {
        name: train_model(name, ds.X[:trval_end], ds.y[:trval_end], cfg, best_iters[name])
        for name in MODEL_NAMES
    }
    evaluation = evaluate_model(eval_models, ensemble_weights, ds, cfg)
    log("ML", f"Huấn luyện ensemble cuối cùng trên toàn bộ {ds.n:,} kỳ")
    final_models = {
        name: train_model(name, ds.X, ds.y, cfg, best_iters[name])
        for name in MODEL_NAMES
    }
    bundle = {
        "bundle_version": 2,
        "models": final_models,
        "ensemble_weights": ensemble_weights,
        "validation_logloss": selection["validation_logloss"],
        "best_iters": best_iters,
        "model_name": "validation_weighted_ensemble",
        "best_iters": best_iters,
        "feature_names": FEATURE_NAMES,
        "trained_on_draws": int(len(df)),
        "last_draw_id": int(df["draw_id"].iloc[-1]),
        "danh_gia_test": evaluation,
        "ket_luan": evaluation["ket_luan"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary = {
        "thoi_diem": bundle["created_at"],
        "canh_bao": WARNING_VI,
        "muc_tieu": "Phân phối xác suất của tổng 3..18 ở kỳ kế tiếp (16 lớp)",
        "phuong_phap_kiem_dinh": f"Chia theo thời gian: train {cfg.train_frac:.0%} / validation {cfg.val_frac:.0%} / test {1 - cfg.train_frac - cfg.val_frac:.0%}",
        "thanh_phan_ensemble": list(selection["ensemble_weights"].keys()),
        "trong_so_ensemble": selection["ensemble_weights"],
        "so_ky_du_lieu": int(len(df)),
        "so_mau": ds.n,
        "so_dac_trung": N_FEATURES,
        "cau_hinh": asdict(cfg),
        "chon_mo_hinh": selection,
        "danh_gia_test": evaluation,
    }
    save_json(out / "bingo18_ket_qua.json", summary)
    rows = [{"Dự đoán": k, **{m: v for m, v in row.items()}} for k, row in evaluation["bang_so_sanh"].items()]
    save_csv(pd.DataFrame(rows), out / "bingo18_so_sanh_baseline.csv")
    if not cfg.no_save_model:
        save_model(bundle, Path(cfg.model_path) if cfg.model_path else out / MODEL_FILENAME)
    return bundle


def save_model(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    log("LƯU", f"Đã lưu mô hình (joblib): {path}")


def load_model(path: Path) -> dict:
    if not path.is_file():
        raise DataError(f"Chưa có mô hình tại {path}. Hãy chạy trước: python ml_bingo18.py --mode train")
    bundle = joblib.load(path)
    if bundle.get("bundle_version") != 2 or "models" not in bundle:
        raise DataError("Mô hình lưu cũ không tương thích với ensemble Bingo18 hiện tại. Hãy chạy --mode train.")
    if bundle.get("feature_names") != FEATURE_NAMES:
        raise DataError("Mô hình đã lưu dùng bộ đặc trưng khác phiên bản hiện tại. Hãy huấn luyện lại (--mode train).")
    log("ML", f"Đã nạp ensemble Bingo18 (huấn luyện trên {bundle['trained_on_draws']:,} kỳ)")
    return bundle


# ----------------------------------------------------------------------------
# 6) Dự đoán kỳ tiếp theo
# ----------------------------------------------------------------------------
def predict_next(bundle: dict, df: pd.DataFrame, out: Optional[Path] = None) -> dict:
    """Forecast the next Bingo18 sum with the validation-weighted hybrid ensemble."""
    feats = next_features(df)
    positions = np.asarray([len(df)], dtype=np.int64)
    y_full = df["sum"].to_numpy(dtype=np.int64) - SUM_MIN
    components = statistical_forecasts(y_full, positions)
    for name, model in bundle["models"].items():
        components[name] = predict_proba(model, feats.to_numpy(dtype=np.float32))
    P = combine_probabilities(components, bundle["ensemble_weights"])[0]
    if len(df) > bundle["trained_on_draws"]:
        log("CẢNH BÁO", f"Dữ liệu có {len(df):,} kỳ, mô hình huấn luyện trên {bundle['trained_on_draws']:,} kỳ: "
                         "đặc trưng dùng kỳ mới nhất nhưng trọng số mô hình chưa cập nhật (nên chạy lại --mode train).")
    p_small = float(P[: BIG_THRESHOLD - SUM_MIN].sum())
    p_big = float(P[BIG_THRESHOLD - SUM_MIN:].sum())
    order = np.argsort(-P)
    top_sum_index = int(order[0])
    normalized_entropy = float(
        -(P * np.log(np.clip(P, 1e-12, 1.0))).sum() / math.log(N_CLASSES)
    )
    table = [
        {
            "tong": s,
            "xac_suat_ensemble": float(P[i]),
            "xac_suat_ly_thuyet": float(THEORY_PMF[i]),
            "chenh_lech": float(P[i] - THEORY_PMF[i]),
        }
        for i, s in enumerate(SUMS)
    ]
    component_confidence = {
        name: {
            "top_sum": int(SUMS[int(np.argmax(probabilities[0]))]),
            "top_probability": float(np.max(probabilities[0])),
        }
        for name, probabilities in components.items()
    }
    payload = {
        "thoi_diem_tao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "canh_bao": WARNING_VI,
        "based_on": {"draw_count": int(len(df)), "last_draw_id": int(df["draw_id"].iloc[-1]),
                     "last_draw": [int(df[c].iloc[-1]) for c in ("n1", "n2", "n3")]},
        "mo_hinh": bundle["model_name"],
        "trong_so_thanh_phan": bundle["ensemble_weights"],
        "du_doan_tong_cao_nhat": int(SUMS[top_sum_index]),
        "confidence_probability": float(P[top_sum_index]),
        "normalized_entropy": normalized_entropy,
        "confidence_note": "Xác suất từ ensemble, không phải đảm bảo độ chính xác hay xác suất trúng.",
        "component_predictions": component_confidence,
        "phan_phoi_tong": table,
        "top3_tong": [int(SUMS[i]) for i in order[:3]],
        "p_nho_3_10": p_small,
        "p_lon_11_18": p_big,
        "ket_luan_danh_gia": bundle.get("ket_luan", ""),
        "ghi_chu": "Kết quả chỉ mang tính tham khảo thống kê, không đảm bảo trúng thưởng.",
    }
    log("DỰ ĐOÁN", f"Kỳ kế tiếp sau draw_id {payload['based_on']['last_draw_id']:,} (kỳ cuối: {payload['based_on']['last_draw']})")
    print(f"          {'Tổng':>5}{'Ensemble':>10}{'Lý thuyết':>11}{'Chênh':>9}")
    for row in table:
        print(f"          {row['tong']:>5}{row['xac_suat_ensemble']:>10.4f}{row['xac_suat_ly_thuyet']:>11.4f}{row['chenh_lech']:>+9.4f}")
    log(
        "CONFIDENCE",
        f"Top sum {payload['du_doan_tong_cao_nhat']}: "
        f"P={payload['confidence_probability']:.4f} "
        f"| normalized entropy={normalized_entropy:.4f}",
    )
    for name, component in component_confidence.items():
        log(
            "COMPONENT",
            f"{COMPONENT_LABELS.get(name, name)}: top {component['top_sum']} "
            f"(P={component['top_probability']:.4f})",
        )
    log("DỰ ĐOÁN", f"Top-3 tổng theo mô hình: {payload['top3_tong']} | P(nhỏ 3-10) = {p_small:.4f} | P(lớn 11-18) = {p_big:.4f}")
    log("DỰ ĐOÁN", payload["ket_luan_danh_gia"])
    if out is not None:
        save_json(out / "bingo18_du_doan.json", payload)
        save_csv(pd.DataFrame(table), out / "bingo18_du_doan_phan_phoi_tong.csv")
    return payload


def save_vietnamese_report(
    out: Path,
    data_report: dict,
    bundle: Optional[dict] = None,
    prediction: Optional[dict] = None,
) -> Path:
    """Write a readable Vietnamese summary of validation, models, and predictions."""
    lines = [
        "BÁO CÁO PHÂN TÍCH VÀ DỰ ĐOÁN BINGO18",
        "=" * 48,
        f"Thời điểm tạo báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "1. CHẤT LƯỢNG DỮ LIỆU",
        f"- Số dòng đã đọc: {data_report.get('tong_dong_doc', 0):,}",
        f"- Số kỳ hợp lệ: {data_report.get('so_ky_hop_le', 0):,}",
        f"- Dòng thiếu hoặc không đọc được số: {data_report.get('loai_do_thieu_so', 0):,}",
        f"- Dòng có số Bingo18 không hợp lệ: {data_report.get('loai_do_so_khong_hop_le', 0):,}",
        f"- Dòng trùng draw_id bị loại: {data_report.get('loai_do_trung_draw_id', 0):,}",
        f"- Dòng có tổng sai bị loại: {data_report.get('dong_sum_sai_bi_loai', 0):,}",
        f"- Khoảng draw_id sau sắp xếp: {data_report.get('draw_id_dau', 'N/A')} -> "
        f"{data_report.get('draw_id_cuoi', 'N/A')}",
    ]

    warnings_found = data_report.get("canh_bao", [])
    if warnings_found:
        lines.append("- Cảnh báo dữ liệu:")
        lines.extend(f"  * {warning}" for warning in warnings_found)
    else:
        lines.append("- Cảnh báo dữ liệu: Không phát hiện.")

    stats = data_report.get("thong_ke", {})
    if stats:
        lines.extend([
            "",
            "2. PHÂN TÍCH THỐNG KÊ",
            f"- Tần suất các mặt 1..6 (gộp ba vị trí): {stats.get('tan_suat_mat_1_6', [])}",
            f"- Kiểm định độ đồng đều các mặt (p-value): "
            f"{_format_report_number(stats.get('p_value_chi2_mat'))}",
            f"- So sánh phân phối tổng với lý thuyết ba xúc xắc (p-value): "
            f"{_format_report_number(stats.get('p_value_chi2_tong_vs_ly_thuyet'))}",
            f"- Tự tương quan tổng giữa hai kỳ liên tiếp: "
            f"{_format_report_number(stats.get('tu_tuong_quan_tong_lag1'))}",
            f"- Thông tin tương hỗ của chuyển tiếp tổng (nats): "
            f"{_format_report_number(stats.get('mutual_information_chuyen_tiep_tong_lag1_nats'))}",
            "- Ma trận tương quan giữa vị trí n1, n2, n3:",
        ])
        for matrix_row in stats.get("tuong_quan_giua_vi_tri_xuc_xac", []):
            lines.append("  " + "  ".join(_format_report_number(value) for value in matrix_row))
        lines.append(f"- Diễn giải: {stats.get('ghi_chu', '')}")

    evaluation = None
    if bundle is not None:
        evaluation = bundle.get("danh_gia_test")
        weights = bundle.get("ensemble_weights", {})
        lines.extend([
            "",
            "3. PHƯƠNG PHÁP VÀ TRỌNG SỐ ENSEMBLE",
            "- Kết quả mục tiêu là tổng ba số Bingo18, từ 3 đến 18.",
            "- Đặc trưng chỉ dùng các kỳ trước; dữ liệu được chia theo thứ tự thời gian.",
            "- Trọng số ensemble được tối ưu trên tập validation, không dùng tập test để chọn trọng số.",
        ])
        for name, weight in weights.items():
            lines.append(
                f"- {COMPONENT_LABELS.get(name, name)}: {float(weight):.2%}"
            )
        validation_losses = bundle.get("validation_logloss", {})
        if validation_losses:
            lines.extend([
                "",
                "Log-loss trên tập validation (chỉ dùng để chọn trọng số ensemble):",
            ])
            for name, value in validation_losses.items():
                lines.append(
                    f"- {COMPONENT_LABELS.get(name, name)}: "
                    f"{float(value):.5f}"
                )
            lines.append(
                "Giải thích: log-loss càng thấp thì phân phối xác suất càng "
                "đánh giá tốt kết quả thực tế trên validation."
            )

    if evaluation:
        lines.extend([
            "",
            "4. ĐÁNH GIÁ TRÊN TẬP KIỂM TRA THEO THỜI GIAN",
            f"- Số kỳ kiểm tra: {evaluation.get('so_ky_test', 0):,}",
            f"- Khoảng draw_id kiểm tra: {evaluation.get('draw_id_test', [])}",
            f"- Log-loss tham chiếu theo lý thuyết ba xúc xắc: "
            f"{float(evaluation.get('log_loss_ky_vong_neu_xuc_xac_cong_bang', 0.0)):.5f}",
            f"- Mức ý nghĩa sau hiệu chỉnh Bonferroni: "
            f"{float(evaluation.get('alpha_bonferroni', 0.0)):.5f}",
            "",
            "Chỉ số các phương pháp (log-loss và Brier càng thấp càng tốt; "
            "Top-1/Top-3 càng cao càng tốt):",
        ])
        scores = evaluation.get("bang_so_sanh", {})
        comparisons = evaluation.get("kiem_dinh_so_voi_baseline", {})
        for name, values in scores.items():
            label = (
                "Ensemble kết hợp"
                if name == "validation_weighted_ensemble"
                else COMPONENT_LABELS.get(name, name)
            )
            lines.append(
                f"- {label}: log-loss={float(values.get('log_loss', 0.0)):.5f}; "
                f"Brier={float(values.get('brier', 0.0)):.5f}; "
                f"Top-1={float(values.get('top1', 0.0)):.2%}; "
                f"Top-3={float(values.get('top3', 0.0)):.2%}"
            )
            comparison = comparisons.get(name)
            if comparison:
                lines.append(
                    f"  So với ensemble (chênh lệch log-loss, z, p-value một phía): "
                    f"{float(comparison.get('chenh_lech_log_loss', 0.0)):+.5f}, "
                    f"{float(comparison.get('z', 0.0)):+.2f}, "
                    f"{float(comparison.get('p_value_mot_phia', 1.0)):.5f}"
                )
        lines.extend([
            f"- Kết luận đánh giá: {evaluation.get('ket_luan', '')}",
            f"- Các phương pháp được ensemble vượt theo kiểm định: "
            f"{evaluation.get('baseline_bi_vuot', [])}",
        ])

    if prediction:
        lines.extend([
            "",
            "5. DỰ ĐOÁN KỲ TIẾP THEO",
            f"- Số kỳ lịch sử sử dụng: {prediction.get('based_on', {}).get('draw_count', 0):,}",
            f"- Kỳ cuối đã biết (ba số): {prediction.get('based_on', {}).get('last_draw', [])}",
            f"- Tổng có xác suất cao nhất: {prediction.get('du_doan_tong_cao_nhat', 'N/A')}",
            f"- Xác suất ensemble của tổng đứng đầu: "
            f"{float(prediction.get('confidence_probability', 0.0)):.2%}",
            f"- Mức bất định entropy chuẩn hóa (0 rõ hơn, 1 phân tán hơn): "
            f"{float(prediction.get('normalized_entropy', 0.0)):.4f}",
            f"- Ba tổng có xác suất cao nhất: {prediction.get('top3_tong', [])}",
            f"- P(tổng nhỏ 3-10): {float(prediction.get('p_nho_3_10', 0.0)):.2%}",
            f"- P(tổng lớn 11-18): {float(prediction.get('p_lon_11_18', 0.0)):.2%}",
            "",
            "Phân phối xác suất dự đoán:",
            "Tổng | Ensemble | Lý thuyết ba xúc xắc | Chênh lệch",
        ])
        for row in prediction.get("phan_phoi_tong", []):
            lines.append(
                f"{int(row['tong']):>4} | "
                f"{float(row['xac_suat_ensemble']):>8.2%} | "
                f"{float(row['xac_suat_ly_thuyet']):>19.2%} | "
                f"{float(row['chenh_lech']):>+9.4%}"
            )
        lines.extend(["", "Dự đoán theo từng thành phần:"])
        for name, component in prediction.get("component_predictions", {}).items():
            lines.append(
                f"- {COMPONENT_LABELS.get(name, name)}: tổng đứng đầu "
                f"{component.get('top_sum', 'N/A')}, xác suất "
                f"{float(component.get('top_probability', 0.0)):.2%}"
            )

    lines.extend([
        "",
        "LƯU Ý",
        WARNING_VI,
        "Xác suất và confidence là độ bất định của phân phối ước lượng, "
        "không phải cam kết chính xác hay bảo đảm trúng thưởng.",
        "",
    ])
    path = out / "bao_cao_tieng_viet.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    log("LƯU", f"Đã ghi báo cáo tiếng Việt: {path}")
    return path


def _format_report_number(value: Any) -> str:
    if value is None:
        return "không có"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.5f}" if math.isfinite(number) else "không xác định"


# ----------------------------------------------------------------------------
# 7) Dòng lệnh
# ----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description="Mô hình thống kê + Machine Learning cho Bingo18 (KHÔNG đảm bảo dự đoán đúng).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", default=None, metavar="FILE", help="File CSV (draw_id,n1,n2,n3,sum). Mặc định: data/bingo18.csv")
    p.add_argument("--mode", choices=["validate", "train", "predict", "full"], default="full",
                   help="validate: chỉ kiểm tra dữ liệu | train: chọn mô hình + đánh giá + lưu | "
                        "predict: dự đoán bằng mô hình đã lưu | full: train + predict")
    p.add_argument("--output-dir", default="output", help="Thư mục lưu kết quả")
    p.add_argument("--model-path", default="", help="Đường dẫn file mô hình (mặc định: <output-dir>/mo_hinh_bingo18.joblib)")
    p.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên (tái lập kết quả)")
    p.add_argument("--train-frac", type=float, default=0.8, help="Tỉ lệ train (đầu chuỗi thời gian)")
    p.add_argument("--val-frac", type=float, default=0.1, help="Tỉ lệ validation (tiếp theo); phần còn lại là test")
    p.add_argument("--max-iter", type=int, default=300, help="Số vòng boosting tối đa (chọn số tốt nhất trên validation)")
    p.add_argument("--learning-rate", type=float, default=0.03, help="Tốc độ học của Gradient Boosting")
    p.add_argument("--n-estimators", type=int, default=150, help="Số cây Random Forest")
    p.add_argument("--max-depth", type=int, default=8, help="Độ sâu tối đa mỗi cây Random Forest")
    p.add_argument("--min-samples-leaf", type=int, default=100, help="Số mẫu tối thiểu ở lá (chống overfit)")
    p.add_argument("--n-jobs", type=int, default=-1, help="Số luồng CPU cho Random Forest (-1 = tất cả)")
    p.add_argument("--no-save-model", action="store_true", help="Không lưu mô hình joblib")
    p.add_argument("--debug", action="store_true", help="In chi tiết lỗi (traceback)")
    ns = p.parse_args(argv)
    csv_file = resolve_csv_file(ns.csv)
    cfg = Config(csv_file=csv_file, **{k: v for k, v in vars(ns).items() if k != "csv"})
    if cfg.max_iter < 1 or cfg.n_estimators < 1 or cfg.min_samples_leaf < 1:
        raise DataError("Tham số không hợp lệ: --max-iter, --n-estimators, --min-samples-leaf phải ≥ 1.")
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    debug = "--debug" in (argv if argv is not None else sys.argv)
    try:
        cfg = parse_args(argv)
        out = Path(cfg.output_dir)
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log("HỆ THỐNG", "Mô hình Bingo18 (3 xúc xắc, tổng 3..18) - " + WARNING_VI)
        df, report = validate_data(load_data(cfg.csv_file))
        save_json(out / "bao_cao_du_lieu.json", {"canh_bao": WARNING_VI, "kiem_tra_du_lieu": report})
        if cfg.mode == "validate":
            save_vietnamese_report(out, report)
            log("RESULT", "Đã hoàn thành kiểm tra dữ liệu")
            return 0
        bundle = None
        prediction = None
        if cfg.mode in ("train", "full"):
            check_no_leakage(df, cfg.seed)
            bundle = run_training(df, cfg, out)
        if cfg.mode in ("predict", "full"):
            if bundle is None:
                bundle = load_model(Path(cfg.model_path) if cfg.model_path else out / MODEL_FILENAME)
            prediction = predict_next(bundle, df, out)
        save_vietnamese_report(out, report, bundle, prediction)
        log("RESULT", f"Đã hoàn thành trong {time.time() - t0:.1f}s. Kết quả nằm trong thư mục: {out.resolve()}")
        return 0
    except DataError as exc:
        log("LỖI", str(exc))
        if debug:
            traceback.print_exc()
        return 2
    except KeyboardInterrupt:
        log("HỦY", "Đã dừng theo yêu cầu của người dùng")
        return 130
    except MemoryError:
        log("LỖI", "Hết bộ nhớ RAM. Hãy giảm --n-estimators hoặc --max-iter.")
        return 3
    except Exception as exc:  # noqa: BLE001
        log("LỖI", f"Lỗi không mong đợi: {exc}. Chạy lại với --debug để xem chi tiết.")
        if debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())