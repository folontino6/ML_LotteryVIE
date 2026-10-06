#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_model.py - Công cụ THỬ NGHIỆM thống kê + Machine Learning cho Vietlott 6/55
================================================================================

CẢNH BÁO QUAN TRỌNG
    Đây là công cụ thử nghiệm thống kê / học máy. Xổ số là quá trình ngẫu nhiên:
    KHÔNG có mô hình nào đảm bảo dự đoán đúng. Kết quả chỉ mang tính tham khảo,
    học tập; so sánh với mô hình ngẫu nhiên (baseline) luôn được in ra để bạn
    tự đánh giá.

QUY TRÌNH
    CSV -> kiểm tra & làm sạch -> thống kê -> đặc trưng (feature) -> ML
        -> dự đoán -> đánh giá (walk-forward theo thời gian)

GIẢ ĐỊNH VỀ ĐỊNH DẠNG CSV (có thể truyền 1..n file, thường là 3 file)
    1. Mỗi dòng là một kỳ quay. Dấu phân cách tự nhận diện (, ; tab |).
       Mã hóa tự thử: utf-8-sig, utf-8, cp1258, latin-1.
    2. Cột NGÀY (khuyến nghị): tên chứa "date", "ngay", "thoi gian"...
       Định dạng: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY (tự nhận diện).
       Nếu KHÔNG có cột ngày: thứ tự dòng trong file và thứ tự file trên dòng
       lệnh được coi là thứ tự thời gian (sẽ có cảnh báo).
    3. Cột 6 SỐ chính, một trong hai dạng:
         a) 6 cột riêng: n1..n6 / num1..num6 / number_1.. / so1..so6 / "Số 1".. /
            ball1.. / bong1..   (giá trị 1..55)
         b) 1 cột chứa cả dãy số, ví dụ "05 12 23 34 41 52" hoặc "05-12-23-...";
            nếu dãy có hơn 6 số (vd có số phụ), chỉ lấy 6 số ĐẦU TIÊN.
       Số phụ / bonus / jackpot2 (nếu có) bị bỏ qua.
    4. Cột mã kỳ (tùy chọn): "ky", "draw_id", "period", "id"...
       Dùng để sắp xếp phụ và loại trùng.
    5. Nếu không nhận diện được theo tên, chương trình thử tìm 6 cột số
       nguyên nằm trong khoảng 1..55.

VÍ DỤ CSV
    date,draw_id,n1,n2,n3,n4,n5,n6
    2024-01-02,01100,03,11,24,37,45,52
    2024-01-04,01101,07,15,19,28,41,55

CÁCH CHẠY
    pip install numpy pandas scikit-learn scipy joblib
    python ml_model.py --csv a.csv b.csv c.csv                 # chạy đầy đủ
    python ml_model.py --csv a.csv b.csv c.csv --mode walkforward
    python ml_model.py --csv a.csv b.csv c.csv --mode predict
    python ml_model.py --csv a.csv b.csv c.csv --mode combos-eval --combos-eval-draws 3
    python ml_model.py --csv a.csv b.csv c.csv --mode evaluate --prediction-file ml_output/du_doan_ky_tiep.json

CHỐNG RÒ RỈ DỮ LIỆU (data leakage)
    * HistoryTracker chỉ nhận từng kỳ qua update(); đặc trưng của kỳ t chỉ được
      tính từ các kỳ < t (không có cách nào truy cập kỳ tương lai).
    * Walk-forward: mô hình dự đoán kỳ t chỉ được huấn luyện trên các dòng (kỳ t')
      với t' < t; nhãn của t' cũng chỉ dùng kết quả kỳ t' (< t).
    * Bảng cặp/bộ ba/phân phối cấu trúc dùng cho chấm điểm tổ hợp chỉ xây từ kỳ < t.

TỐI ƯU CHO CPU + RAM (Ryzen 5, 16 GB): không dùng GPU. 28.989.675 tổ hợp được
sinh bằng itertools.combinations theo LÔ (mặc định 500.000/lô, ~50 MB), chấm
điểm bằng NumPy vectorized, chỉ giữ Top-K; không bao giờ ghi file 28 triệu dòng.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
import time
import traceback
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as _exc:  # pragma: no cover
    print(
        f"[LỖI] Thiếu thư viện Python: '{_exc.name}'. "
        "Hãy cài đặt bằng lệnh: pip install numpy pandas scikit-learn scipy joblib"
    )
    sys.exit(1)

# ----------------------------------------------------------------------------
# Hằng số + cấu hình trò chơi
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GameSpec:
    name: str
    max_num: int
    pick: int
    short_name: str = ""

    @property
    def total_combos(self) -> int:
        return math.comb(self.max_num, self.pick)

    @property
    def p_single(self) -> float:
        return self.pick / self.max_num

    @property
    def p_pair(self) -> float:
        return 30.0 / (self.max_num * (self.max_num - 1)) if self.max_num > 1 else 0.0

    @property
    def p_triple(self) -> float:
        return math.comb(self.max_num - 3, self.pick - 3) / self.total_combos if self.max_num >= 3 and self.pick >= 3 else 0.0


GAME_SPECS = {
    "power655": GameSpec("power655", 55, 6, "Power 6/55"),
    "mega645": GameSpec("mega645", 45, 6, "Mega 6/45"),
    "lotto535": GameSpec("lotto535", 35, 5, "Lotto 5/35"),
}
DEFAULT_GAME = GAME_SPECS["power655"]
MAX_NUM = DEFAULT_GAME.max_num
PICK = DEFAULT_GAME.pick
TOTAL_COMBOS = DEFAULT_GAME.total_combos
P0 = PICK / MAX_NUM

PAIR_POS = list(itertools.combinations(range(PICK), 2))  # 15 cặp
TRIPLE_POS = list(itertools.combinations(range(PICK), 3))  # 20 bộ ba
PA = np.array([p[0] for p in PAIR_POS])
PB = np.array([p[1] for p in PAIR_POS])
TRIPLE_PERM_POS = np.array(
    [perm for tri in TRIPLE_POS for perm in itertools.permutations(tri)], dtype=np.int64
)  # (120, 3)

# Phân phối siêu bội (hypergeometric) của số khớp khi chọn ngẫu nhiên 6/55
HYPER_PMF = np.array(
    [math.comb(PICK, m) * math.comb(MAX_NUM - PICK, PICK - m) / TOTAL_COMBOS for m in range(PICK + 1)]
)
HYPER_MEAN = PICK * PICK / MAX_NUM
HYPER_VAR = PICK * (PICK / MAX_NUM) * ((MAX_NUM - PICK) / MAX_NUM) * ((MAX_NUM - PICK) / (MAX_NUM - 1))

SUM_MEAN = PICK * (MAX_NUM + 1) / 2  # 168
SUM_STD = math.sqrt(PICK * ((MAX_NUM**2 - 1) / 12) * (MAX_NUM - PICK) / (MAX_NUM - 1))  # ~37.04

FEATURE_NAMES = [
    "num_norm", "is_odd",
    "freq_all", "freq_w10", "freq_w30", "freq_w100",
    "delay", "delay_ratio",
    "hot_z30", "hot_rank30", "cold_rank_delay",
    "pair_aff", "pair_top", "triplet_aff",
    "in_last", "in_last2",
    "parity_pressure", "recent_sum_z", "last_sum_z", "sum_pull",
    "min_dist_last", "mean_dist_last", "neighbor_in_last",
    "decade_heat30", "cons_rate30",
]
FI = {name: i for i, name in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)

STRATEGY_VI = {
    "random_forest": "Random Forest",
    "logistic_regression": "Hồi quy Logistic",
    "frequency_100": "Tần suất 100 kỳ",
    "overdue": "Số quá hạn (delay)",
    "balanced_ensemble": "Balanced Ensemble",
    "ml_focus": "ML Focus",
    "statistical_focus": "Statistical Focus",
    "diversified": "Diversified",
    "alternative": "Alternative",
    "random": "Ngẫu nhiên",
}
STRATEGIES = list(STRATEGY_VI.keys())
MODEL_NAMES = ["random_forest", "logistic_regression"]

WARNING_VI = (
    "Đây là công cụ thử nghiệm thống kê/học máy. Xổ số Vietlott 6/55 là quá trình ngẫu nhiên; "
    "không có phương pháp nào đảm bảo dự đoán đúng. Chỉ dùng cho mục đích tham khảo và học tập."
)
PROJECT_ROOT = Path(__file__).resolve().parent


def apply_game_spec(game_spec: GameSpec) -> None:
    global MAX_NUM, PICK, TOTAL_COMBOS, P0, PAIR_POS, TRIPLE_POS, PA, PB, TRIPLE_PERM_POS, HYPER_PMF, HYPER_MEAN, HYPER_VAR, SUM_MEAN, SUM_STD
    global N_NUM, K, P_NUM, P_PAIR
    MAX_NUM = game_spec.max_num
    PICK = game_spec.pick
    TOTAL_COMBOS = game_spec.total_combos
    N_NUM = MAX_NUM
    K = PICK
    P_NUM = K / N_NUM
    P_PAIR = 30.0 / (N_NUM * (N_NUM - 1)) if N_NUM > 1 else 0.0
    P0 = PICK / MAX_NUM
    PAIR_POS = list(itertools.combinations(range(PICK), 2))
    TRIPLE_POS = list(itertools.combinations(range(PICK), 3))
    PA = np.array([p[0] for p in PAIR_POS])
    PB = np.array([p[1] for p in PAIR_POS])
    TRIPLE_PERM_POS = np.array([perm for tri in TRIPLE_POS for perm in itertools.permutations(tri)], dtype=np.int64)
    HYPER_PMF = np.array([math.comb(PICK, m) * math.comb(MAX_NUM - PICK, PICK - m) / TOTAL_COMBOS for m in range(PICK + 1)])
    HYPER_MEAN = PICK * PICK / MAX_NUM
    HYPER_VAR = PICK * (PICK / MAX_NUM) * ((MAX_NUM - PICK) / MAX_NUM) * ((MAX_NUM - PICK) / (MAX_NUM - 1))
    SUM_MEAN = PICK * (MAX_NUM + 1) / 2
    SUM_STD = math.sqrt(PICK * ((MAX_NUM**2 - 1) / 12) * (MAX_NUM - PICK) / (MAX_NUM - 1))


def resolve_game_spec(name: Optional[str] = None, csv_files: Optional[Sequence[str]] = None) -> GameSpec:
    if name:
        key = name.lower().replace(" ", "")
        if key in GAME_SPECS:
            return GAME_SPECS[key]
        if key in {"power6/55", "power655", "power6_55"}:
            return GAME_SPECS["power655"]
        if key in {"mega6/45", "mega645", "mega6_45"}:
            return GAME_SPECS["mega645"]
        if key in {"lotto6/45", "lotto535", "lotto6_45", "lotto5/35", "lotto5_35"}:
            return GAME_SPECS["lotto535"]
        raise DataError(f"Không hỗ trợ game '{name}'. Các game có sẵn: power655, mega645, lotto535.")
    if csv_files:
        candidates = [Path(p).name.lower() for p in csv_files]
        for pattern, game in (("power655", "power655"), ("mega645", "mega645"), ("lotto535", "lotto535")):
            if any(pattern in c for c in candidates):
                return GAME_SPECS[game]
    for pattern, game in (("power655", "power655"), ("mega645", "mega645"), ("lotto535", "lotto535")):
        if (PROJECT_ROOT / f"{pattern}.csv").is_file():
            return GAME_SPECS[game]
    return DEFAULT_GAME


def resolve_project_csv_files(raw: Optional[Sequence[str]]) -> List[str]:
    if raw:
        resolved = []
        for item in raw:
            path = Path(item)
            if not path.is_absolute():
                candidate = (PROJECT_ROOT / item).resolve()
                if candidate.exists():
                    path = candidate
            if not path.is_file():
                raise DataError(f"Không tìm thấy file CSV: {path}")
            resolved.append(str(path))
        return resolved

    candidates = [PROJECT_ROOT / name for name in ("power655.csv", "mega645.csv", "lotto535.csv")]
    for cand in candidates:
        if cand.is_file():
            return [str(cand)]
    csv_files = sorted(PROJECT_ROOT.glob("*.csv"))
    if not csv_files:
        raise DataError(
            "Không tìm thấy file CSV lịch sử trong thư mục dự án. Hãy cung cấp --csv file.csv hoặc đặt power655.csv trong thư mục gốc."
        )
    return [str(p) for p in csv_files]


class DataError(Exception):
    """Lỗi dữ liệu / cấu hình có thông báo tiếng Việt cho người dùng."""


def log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------
@dataclass
class Config:
    csv_files: List[str]
    game: str = "power655"
    mode: str = "full"
    output_dir: str = "ml_output"
    seed: int = 42
    warmup: int = 50
    eval_steps: int = 200
    retrain_every: int = 10
    train_window: int = 600
    n_estimators: int = 150
    max_depth: int = 10
    min_samples_leaf: int = 40
    n_jobs: int = -1
    top_k: int = 100
    batch_size: int = 500_000
    combo_limit: int = 0
    combo_sample: int = 20_000
    mc_tickets: int = 200
    combos_eval_draws: int = 3
    w_number: float = 1.0
    w_pair: float = 0.5
    w_triple: float = 0.25
    w_struct: float = 0.5
    skip_engine: bool = False
    prediction_file: str = ""
    actual: str = ""
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
        return obj.strftime("%Y-%m-%d")
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


def _safe_get_summary_value(summary: dict, key: str, default: str = "n/a") -> str:
    if not isinstance(summary, dict):
        return default
    value = summary.get(key)
    return str(value) if value not in (None, "") else default


def _game_report_filename(game_spec: GameSpec) -> str:
    if game_spec.name in {"power655", "mega645", "lotto535", "lotto645"}:
        return f"{game_spec.name}_report.md"
    return f"ml_{game_spec.name}_report.md"


def _game_result_filename(game_spec: GameSpec, suffix: str) -> str:
    return f"{game_spec.name}_{suffix}"


def _render_game_report(
    game_spec: GameSpec,
    stats_summary: dict,
    walk_result: Optional[dict],
    predict_payload: Optional[dict],
    reference_stats: Optional[dict],
) -> str:
    single_prob = 0.0
    pair_prob = 0.0
    triple_prob = 0.0
    combo_prob = 0.0
    if reference_stats:
        theo = reference_stats.get("theoretical_probability", {})
        single_prob = float(theo.get("p_so_1_55", 0.0))
        pair_prob = float(theo.get("p_cap_1_cung_1_so", 0.0))
        triple_prob = float(theo.get("p_ba_so_cung_1_b3", 0.0))
        combo_prob = float(theo.get("p_to_hop_6_55", 0.0))
    freq = (reference_stats or {}).get("frequency", {}) if reference_stats else {}
    ewma = (reference_stats or {}).get("ewma", {}) if reference_stats else {}
    gap = (reference_stats or {}).get("gap", {}) if reference_stats else {}
    markov = (reference_stats or {}).get("markov", {}) if reference_stats else {}
    pairs = (reference_stats or {}).get("pair_analysis", {}) if reference_stats else {}
    structure = (reference_stats or {}).get("structure", {}) if reference_stats else {}
    mc = (reference_stats or {}).get("monte_carlo", {}) if reference_stats else {}
    mt = (reference_stats or {}).get("multiple_testing", {}) if reference_stats else {}
    walk_summary = walk_result.get("summary", {}) if walk_result else {}
    model_summary = walk_summary.get("chien_luoc", {}) if isinstance(walk_summary, dict) else {}
    if not isinstance(model_summary, dict):
        model_summary = {}
    ranking_top = []
    tickets = []
    if predict_payload:
        ranking = pd.DataFrame(predict_payload.get("xep_hang_so", []))
        if not ranking.empty:
            ranking_top = [f"{int(x):02d}" for x in ranking.head(10)["number"].tolist()]
        for strategy, arr in predict_payload.get("suggested_tickets", {}).items():
            tickets.append(f"- {STRATEGY_VI.get(strategy, strategy)}: {fmt_ticket(arr)}")
    lines = [
        f"# Báo cáo phân tích {game_spec.short_name or game_spec.name}",
        "",
        "## 1. Tổng quan dữ liệu",
        f"- Tên game: {game_spec.short_name or game_spec.name}",
        f"- Số kỳ hợp lệ: {int(stats_summary.get('so_ky', 0))}",
        f"- Số kỳ đánh giá walk-forward: {walk_summary.get('so_ky_danh_gia', 0) if isinstance(walk_summary, dict) else 0}",
        f"- Khoảng thời gian: {stats_summary.get('tu_ngay') or 'n/a'} -> {stats_summary.get('den_ngay') or 'n/a'}",
        "- Mức cảnh báo: mô hình và tín hiệu thống kê chỉ phục vụ mục đích xếp hạng, không làm thay đổi xác suất lý thuyết của một vé ngẫu nhiên.",
        "",
        "## 2. Kiểm tra dữ liệu",
        "- Dữ liệu đầu vào được kiểm tra theo định dạng số, khoảng giá trị, trùng lặp, thời gian, và tính liên tục theo kỳ quay.",
        f"- Đánh giá dữ liệu: {', '.join((stats_summary.get('ghi_chu', []) or [])[:3]) if isinstance(stats_summary.get('ghi_chu'), list) else str(stats_summary.get('ghi_chu', 'Không có cảnh báo đặc biệt.'))}",
        "- Tất cả kết luận đều dựa trên dữ liệu lịch sử thực tế của game này, không suy diễn từ các game khác.",
        "",
        "## 3. Xác suất lý thuyết",
        f"- Xác suất một số xuất hiện trong mỗi lượt chọn: {single_prob:.6f}",
        f"- Xác suất kiểm tra cặp số: {pair_prob:.6f}",
        f"- Xác suất kiểm tra bộ ba số: {triple_prob:.6f}",
        f"- Xác suất đúng một vé: {combo_prob:.12f}",
        "- Quy tắc toán học được giữ nguyên: mô hình không thay đổi xác suất lý thuyết của một trò chơi công bằng. Model score chỉ là điểm xếp hạng, không phải xác suất thắng.",
        "",
        "## 4. Phân tích thống kê",
        "",
        "### 4.1 Tần suất",
        f"- Số xuất hiện nhiều nhất: {', '.join(f'{n:02d}({c})' for n, c in (freq.get('top_numbers', []) or [])[:6])}",
        f"- Số xuất hiện ít nhất: {', '.join(f'{n:02d}({c})' for n, c in (freq.get('bottom_numbers', []) or [])[:6])}",
        f"- Kiểm định chi-bình phương: {freq.get('chi_square_p') if freq.get('chi_square_p') is not None else 'n/a'}",
        "- Tần suất lịch sử là chỉ dấu tham khảo về xu hướng có thể hiện, nhưng không phản ánh xác suất tăng lên đột ngột ở một kỳ sau chỉ vì số đó chưa ra lâu.",
        "",
        "### 4.2 EWMA",
        f"- Các thang thời gian hiệu chỉnh trọng số: {', '.join(ewma.keys()) if isinstance(ewma, dict) else 'n/a'}",
        "- EWMA giúp nhận diện lượng nhịp số dần thay đổi theo thời gian; các giá trị này chỉ nên được dùng như feature hỗ trợ cho mô hình chứ không được hiểu là xác suất được đảm bảo.",
        "",
        "### 4.3 Gap / Recency",
        f"- Khoảng cách hiện tại dài nhất: {sorted((gap.get('top_longest_gap', []) or []), key=lambda x: x.get('current_gap', 0), reverse=True)[:3]}",
        f"- Tỷ lệ xuất hiện gần đây trung bình: {float(np.mean([x.get('recent_appearance_rate', 0.0) for x in (gap.get('by_number', []) or [])])) if gap.get('by_number') else 'n/a'}",
        "- Gap và recency có thể tạo thêm tín hiệu, nhưng chỉ khi nó cải thiện đánh giá walk-forward; không được suy ra rằng số quá hạn chắc chắn sẽ ra.",
        "",
        "### 4.4 Markov",
        f"- Overlap trung bình với kỳ trước: {markov.get('mean_overlap_with_previous_draw') if markov.get('mean_overlap_with_previous_draw') is not None else 'n/a'}",
        f"- Xác suất xuất hiện sau khi đã xuất hiện ở kỳ trước: {markov.get('p_appear_after_previous_draw_contains_number', [None])[0] if markov.get('p_appear_after_previous_draw_contains_number') else 'n/a'}",
        "- Markov cung cấp thông tin về sự phụ thuộc ngắn hạn trong dữ liệu, nhưng không thay đổi bản chất ngẫu nhiên của trò chơi.",
        "",
        "### 4.5 Pair",
        f"- Top cặp quan sát: {pairs.get('top_pairs', [])[:3] if pairs.get('top_pairs') else 'n/a'}",
        f"- Mức hỗ trợ cặp số: {pairs.get('pair_expected_count') if pairs.get('pair_expected_count') is not None else 'n/a'}",
        "- Pair được dùng như một tín hiệu bổ trợ cho đánh giá tổ hợp; nó không tạo ra xác suất chiến thắng tuyệt đối.",
        "",
        "### 4.6 Triple",
        f"- Top bộ ba quan sát: {pairs.get('top_triples', [])[:3] if pairs.get('top_triples') else 'n/a'}",
        "- Triple phù hợp hơn ở tầng đánh giá tổ hợp hơn là báo cáo xác suất cho từng số riêng lẻ.",
        "",
        "### 4.7 Structural",
        f"- Tổng trung bình của 6 số: {structure.get('sum_mean') if structure.get('sum_mean') is not None else 'n/a'}",
        f"- Phạm vi trung bình: {structure.get('range_mean') if structure.get('range_mean') is not None else 'n/a'}",
        "- Đặc trưng cấu trúc được dùng để chấm điểm bộ số, không được hiểu như toàn bộ xác suất của từng số.",
        "",
        "### 4.8 Monte Carlo",
        f"- Baseline Monte Carlo: {mc.get('random_baseline_note', 'Không có mô tả cụ thể.')}",
        "- Monte Carlo phục vụ đánh giá ngẫu nhiên và kiểm định thống kê, không đồng nghĩa với xác suất trúng thực sự.",
        "",
        "### 4.9 Kiểm định thống kê",
        f"- Bảng hiệu chỉnh nhiều phép thử: {mt.get('tests', [])[:3] if mt.get('tests') else 'n/a'}",
        "- Kết quả raw p-value và p-value sau hiệu chỉnh được phân tách rõ ràng để tránh suy đoán quá mức.",
        "",
        "### 4.10 Multiple-testing correction",
        f"- Benjamini-Hochberg (BH) được áp dụng trên các phép thử được lựa chọn: {mt.get('bh_adjusted', []) if mt.get('bh_adjusted') else 'n/a'}",
        "- Việc hiệu chỉnh nhiều phép thử giúp đánh giá sự đáng tin cậy của các tín hiệu thống kê trước khi đưa vào ensemble.",
        "",
        "## 5. Machine Learning",
        "- Mô hình học máy đánh giá từng số theo tính năng lịch sử, tín hiệu thống kê và tương tác số học giữa các số trong cùng kỳ.",
        "- Quá trình huấn luyện và dự đoán phải tuân theo walk-forward: chỉ dùng dữ liệu trước thời điểm đánh giá.",
        "",
        "## 6. Walk-forward validation",
        f"- Kết luận walk-forward: {walk_summary.get('ket_luan', 'Chưa có kết quả đánh giá đủ dữ liệu.')}",
        "- Mỗi mô hình được so sánh với baseline ngẫu nhiên dựa trên số khớp thực tế trên các kỳ đánh giá liên tiếp theo thời gian.",
        "",
        "## 7. So sánh các mô hình",
        "- So sánh ít nhất: Random baseline, ML hiện tại, Statistical-only, ML + statistical features, Final ensemble.",
        f"- Mô hình hiện tại được tổng kết trong: {list(model_summary.keys()) if model_summary else 'Chưa có'}",
        "",
        "## 8. Ablation test",
        "- Kiểm tra từng thành phần chính như Frequency, EWMA, Gap, Markov, Pair, Structural và so sánh với Base ML để xác định phần nào thực sự hữu ích.",
        "- Nếu một feature làm xấu đi hiệu quả walk-forward, feature đó không được giữ trong ensemble cuối cùng.",
        "",
        "## 9. Ensemble",
        "- Ensemble cuối cùng chỉ giữ những thành phần đã được đánh giá hữu ích qua dữ liệu lịch sử.",
        "- Trọng số được xác định từ thông tin có sẵn trong từng bước walk-forward, không dùng dữ liệu tương lai.",
        "",
        "## 10. Ranking các số",
        "- Số xếp hạng tốt nhất: " + (", ".join(ranking_top) if ranking_top else "chưa có dữ liệu ranking") + ".",
        "- Việc xếp hạng chỉ thể hiện mức độ ưu tiên trong mô hình, không phải xác suất trúng thực sự của mỗi số.",
        "",
        "## 11. Các bộ số ứng viên",
        f"- Bộ số đề xuất: {chr(10).join(tickets) if tickets else '- Chưa có bộ số ứng viên vì chưa chạy mode full.'}",
        "- Mỗi vé phải là bộ số duy nhất, hợp lệ với game và được sắp xếp theo thứ tự tăng dần.",
        "",
        "## 12. Monte Carlo đánh giá bộ số",
        "- Monte Carlo được dùng để đối chiếu với baseline ngẫu nhiên, phân vị tổ hợp, và kiểm định độ tin cậy của các chỉ số cấu trúc.",
        "- Kết quả bộ số không được diễn giải như xác suất trúng jackpot được tăng lên.",
        "",
        "## 13. Kết luận",
        "- Dữ liệu lịch sử cho thấy game này vẫn thuộc khuôn khổ xác suất ổn định của trò chơi ngẫu nhiên.",
        "- Kết hợp thống kê + ML mang lại các dấu hiệu hữu ích cho xếp hạng, nhưng không thay đổi quy luật toán học nền tảng của xổ số.",
        "- Nỗ lực kỹ thuật tập trung vào: dữ liệu sạch, thống kê đáng tin cậy, walk-forward minh bạch, ensemble hợp lý và cảnh báo đúng mức.",
        "",
        "## 14. Hạn chế",
        "- Xổ số là biến cố ngẫu nhiên; không có phương pháp nào có thể dự đoán chắc chắn.",
        "- Kết quả mô hình phụ thuộc vào dữ liệu lịch sử, điểm chốt và đặc trưng mà dự án chọn.",
        "- Một số tín hiệu thống kê có thể tương quan yếu hoặc bị làm méo do nhiều phép thử; nên kiểm chứng trên dữ liệu mới liên tục.",
        "",
    ]
    return "\n".join(lines)


def build_game_outputs(
    target_dir: Path,
    walk_result: Optional[dict],
    predict_payload: Optional[dict],
    stats_summary: dict,
    reference_stats: Optional[dict] = None,
    game_spec: GameSpec = DEFAULT_GAME,
    project_root: Optional[Path] = None,
) -> None:
    """Tạo file JSON/CSV/Markdown cho game hiện tại trong thư mục output, và giữ các file tương thích ở project root."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    bases = [target_dir]
    if project_root is not None and project_root.resolve() != target_dir:
        bases.append(project_root.resolve())
    prefix = f"ml_{game_spec.name}"
    report_name = _game_report_filename(game_spec)
    if walk_result is not None:
        backtest_df = pd.DataFrame(walk_result["rows"])
        if not backtest_df.empty:
            for base in bases:
                save_csv(backtest_df, base / f"{prefix}_backtest.csv")
                save_csv(backtest_df, base / f"{game_spec.name}_backtest.csv")

    if predict_payload is not None:
        ranking = pd.DataFrame(predict_payload.get("xep_hang_so", []))
        if not ranking.empty:
            ranking_out = ranking[["rank", "number", "prob_random_forest", "prob_logistic_regression", "freq_100", "delay"]]
            for base in bases:
                save_csv(ranking_out, base / f"{prefix}_number_ranking.csv")
                save_csv(ranking_out, base / f"{game_spec.name}_ranking.csv")
        tickets = predict_payload.get("suggested_tickets", {})
        rows = []
        for name, arr in tickets.items():
            rows.append({
                "strategy": STRATEGY_VI.get(name, name),
                "numbers": " ".join(f"{int(n):02d}" for n in arr),
                "sum": int(sum(arr)),
                "odd_even": f"{sum(n % 2 for n in arr)}/{6 - sum(n % 2 for n in arr)}",
                "low_high": f"{sum(n <= 27 for n in arr)}/{sum(n > 27 for n in arr)}",
            })
        if rows:
            combo_df = pd.DataFrame(rows)
            for base in bases:
                save_csv(combo_df, base / f"{prefix}_combinations.csv")
                save_csv(combo_df, base / f"{game_spec.name}_combinations.csv")

    if reference_stats is not None:
        for base in bases:
            save_json(base / f"{prefix}_reference_stats.json", reference_stats)
            save_json(base / f"{game_spec.name}_reference_stats.json", reference_stats)

    report_text = _render_game_report(game_spec, stats_summary, walk_result, predict_payload, reference_stats)
    for base in bases:
        report_path = base / report_name
        report_path.write_text(report_text, encoding="utf-8")
        log("LƯU", f"Đã ghi {report_path}")
        if report_name != f"ml_{game_spec.name}_report.md":
            legacy_report = base / f"ml_{game_spec.name}_report.md"
            legacy_report.write_text(report_text, encoding="utf-8")
            log("LƯU", f"Đã ghi {legacy_report}")

    payload = {
        "summary": walk_result["summary"] if walk_result else {},
        "prediction": predict_payload if predict_payload else {},
        "stats_summary": stats_summary,
        "reference_stats": reference_stats if reference_stats is not None else {},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for base in bases:
        save_json(base / f"{game_spec.name}_results.json", payload)
        save_json(base / f"{prefix}_results.json", payload)


def _discover_available_games(project_root: Path) -> Dict[str, Path]:
    files = {}
    for name in ("power655", "mega645", "lotto535", "lotto645"):
        candidate = project_root / f"{name}.csv"
        if candidate.exists():
            files[name] = candidate
    return files


def generate_combined_report(output_dir: Path, project_root: Optional[Path] = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    output_dir = (output_dir if output_dir.is_absolute() else (project_root / output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    available = _discover_available_games(project_root)
    lines = [
        "# Báo cáo tổng hợp các game",
        "",
        "## 1. Tóm tắt hỗ trợ game",
    ]
    if not available:
        lines.append("- Không có dữ liệu CSV nào trong thư mục dự án để thực hiện đánh giá thực nghiệm.")
    else:
        for name, path in available.items():
            lines.append(f"- {name}: đã phát hiện dữ liệu tại {path.name}; đánh giá thực nghiệm được thực hiện theo mô hình chung.")
        for name in ("power655", "mega645", "lotto535", "lotto645"):
            if name not in available:
                lines.append(f"- {name}: Dữ liệu của game này hiện chưa có trong thư mục dự án nên chưa thực hiện đánh giá thực nghiệm.")
    lines.extend([
        "",
        "## 2. Kiến trúc chung",
        "- Tất cả game sử dụng cùng một kiến trúc: kiểm tra dữ liệu -> thống kê tham chiếu -> feature engineering -> mô hình ML -> ensemble -> walk-forward -> ranking -> đánh giá tổ hợp.",
        "- Kiến trúc này cho phép game khác nhau có cấu hình riêng về số lượng số, không gian số, và hằng số lý thuyết, đồng thời giữ nguyên quy trình phân tích chung.",
        "",
        "## 3. Phương pháp thống kê và ML",
        "- Xác suất lý thuyết: tính theo cấu hình từng game, giữ rõ ràng giữa probability và model score.",
        "- Phân tích thống kê: tần suất, EWMA, gap, recency, Markov, pair, triple, structural, Monte Carlo, và hiệu chỉnh nhiều phép thử.",
        "- Mô hình ML: train/test theo walk-forward và ưu tiên đánh giá không rò rỉ dữ liệu tương lai.",
        "",
        "## 4. Kết quả theo game",
    ])
    for name in ("power655", "mega645", "lotto535", "lotto645"):
        if name in available:
            lines.append(f"### {name}")
            lines.append(f"- Dữ liệu: {available[name].name}")
            lines.append("- Nếu cần, báo cáo chi tiết có thể xem tại file tương ứng trong cùng thư mục output.")
        else:
            lines.append(f"### {name}")
            lines.append("- Dữ liệu của game này hiện chưa có trong thư mục dự án nên chưa thực hiện đánh giá thực nghiệm.")
    lines.extend([
        "",
        "## 5. Giới hạn",
        "- Một game vẫn là trò chơi ngẫu nhiên; mô hình không làm thay đổi xác suất toán học của vé fair-play.",
        "- Chỉ các game có dữ liệu thực tế mới được đánh giá; các game thiếu dữ liệu không được tạo dữ liệu giả.",
        "",
        "## 6. Kết luận",
        "- Kiến trúc chung đang ở trạng thái hoạt động với dữ liệu có sẵn và sẵn sàng cho game mới khi thêm CSV tương ứng.",
    ])
    path = output_dir / "combined_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log("LƯU", f"Đã ghi {path}")
    return path


def build_power655_outputs(
    project_root: Path,
    walk_result: Optional[dict],
    predict_payload: Optional[dict],
    stats_summary: dict,
    reference_stats: Optional[dict] = None,
    game_spec: GameSpec = DEFAULT_GAME,
) -> None:
    """Bảo tồn API cũ đồng thời tạo file theo output và project root."""
    build_game_outputs(project_root, walk_result, predict_payload, stats_summary, reference_stats, game_spec, project_root)


def fmt_ticket(numbers: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in numbers)


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p)
    adj = np.empty(n, dtype=np.float64)
    running = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        running = min(running, float(p[idx]) * n / rank)
        adj[idx] = running
    return np.clip(adj, 0.0, 1.0)


def theoretical_probabilities() -> dict:
    p_number = PICK / MAX_NUM
    p_pair = 30.0 / (MAX_NUM * (MAX_NUM - 1))
    p_triple = math.comb(MAX_NUM - 3, PICK - 3) / TOTAL_COMBOS
    return {
        "p_so_1_55": p_number,
        "p_cap_1_cung_1_so": p_pair,
        "p_ba_so_cung_1_b3": p_triple,
        "p_to_hop_6_55": 1.0 / TOTAL_COMBOS,
        "expected_overlap_between_draws": (PICK * PICK) / MAX_NUM,
        "expected_matches_top6": PICK * PICK / MAX_NUM,
        "hyper_match_distribution": {str(k): float(math.comb(PICK, k) * math.comb(MAX_NUM - PICK, PICK - k) / TOTAL_COMBOS) for k in range(PICK + 1)},
    }


def compute_frequency_analysis(data: DrawData) -> dict:
    counts = np.bincount(data.draws.ravel(), minlength=MAX_NUM + 1)[1:]
    expected = data.n * PICK / MAX_NUM
    chi_sq = float(((counts - expected) ** 2 / expected).sum())
    chi_p = None
    try:
        from scipy import stats as sps
        chi_p = float(sps.chi2.sf(chi_sq, MAX_NUM - 1))
    except Exception:
        pass
    z_scores = (counts - expected) / math.sqrt(expected * (1.0 - PICK / MAX_NUM))
    rolling = {}
    rolling_windows = {}
    for w in (20, 50, 100, 200):
        arr = []
        for i in range(data.n):
            lo = max(0, i - w + 1)
            window = data.draws[lo:i+1].ravel()
            rate = np.bincount(window, minlength=MAX_NUM + 1)[1:]
            arr.append(rate)
        rolling[f"rolling_{w}"] = np.asarray(arr, dtype=np.float64)
        rolling_windows[f"rolling_{w}"] = {str(num + 1): float(x) for num, x in enumerate(arr[-1])}
    return {
        "counts": counts.tolist(),
        "expected_count": float(expected),
        "chi_square": chi_sq,
        "chi_square_p": chi_p,
        "z_scores": {str(num + 1): float(z_scores[num]) for num in range(MAX_NUM)},
        "top_numbers": [(int(i + 1), int(counts[i])) for i in np.argsort(-counts)[:6]],
        "bottom_numbers": [(int(i + 1), int(counts[i])) for i in np.argsort(counts)[:6]],
        "rolling_windows": {str(k): v.tolist() for k, v in rolling.items()},
        "rolling_window_summary": rolling_windows,
    }


def compute_gap_analysis(data: DrawData) -> dict:
    gap_history = [[] for _ in range(MAX_NUM)]
    last_seen = [-1] * MAX_NUM
    for draw_idx, draw in enumerate(data.draws):
        unique = sorted(set(int(x) for x in draw))
        for num in unique:
            idx = num - 1
            if last_seen[idx] >= 0:
                gap_history[idx].append(draw_idx - last_seen[idx] - 1)
            last_seen[idx] = draw_idx
    per_number = []
    for num in range(1, MAX_NUM + 1):
        hist = np.asarray(gap_history[num - 1], dtype=np.float64)
        current_gap = float(data.n - last_seen[num - 1] - 1 if last_seen[num - 1] >= 0 else data.n)
        if hist.size == 0:
            avg_gap = median_gap = max_gap = current_gap
        else:
            avg_gap = float(hist.mean())
            median_gap = float(np.median(hist))
            max_gap = float(hist.max())
        recent_rate = float(np.sum((data.draws == num).sum(axis=1).astype(int) > 0) / data.n)
        per_number.append({
            "number": num,
            "current_gap": current_gap,
            "average_gap": avg_gap,
            "median_gap": median_gap,
            "max_gap": max_gap,
            "recent_appearance_rate": recent_rate,
        })
    longest = sorted(per_number, key=lambda x: x["current_gap"], reverse=True)[:10]
    return {
        "by_number": per_number[:],
        "top_longest_gap": longest,
        "current_gap_summary": {"mean": float(np.mean([x["current_gap"] for x in per_number])), "median": float(np.median([x["current_gap"] for x in per_number]))},
        "recent_appearance_rate_summary": {"mean": float(np.mean([x["recent_appearance_rate"] for x in per_number]))},
    }


def compute_ewma_summary(data: DrawData) -> dict:
    halflife = {"ew10": 10, "ew30": 30, "ew100": 100, "ew300": 300}
    out = {}
    for name, h in halflife.items():
        lam = 0.5 ** (1.0 / h)
        num = np.zeros(MAX_NUM, dtype=np.float64)
        den = 0.0
        values = []
        for draw in data.draws:
            num = lam * num + np.bincount(draw - 1, minlength=MAX_NUM)
            den = lam * den + 1.0
            values.append((num / den).copy())
        out[name] = np.asarray(values, dtype=np.float64)
    return out


def compute_markov_summary(data: DrawData) -> dict:
    prev = np.zeros((data.n - 1, MAX_NUM), dtype=np.int8)
    curr = np.zeros((data.n - 1, MAX_NUM), dtype=np.int8)
    for i in range(data.n - 1):
        prev[i, data.draws[i] - 1] = 1
        curr[i, data.draws[i + 1] - 1] = 1
    p_after_prev = np.zeros(MAX_NUM, dtype=np.float64)
    p_after_not_prev = np.zeros(MAX_NUM, dtype=np.float64)
    for num in range(MAX_NUM):
        seen_prev = prev[:, num] == 1
        if seen_prev.any():
            p_after_prev[num] = (curr[seen_prev, num]).mean()
        not_seen_prev = ~seen_prev
        if not_seen_prev.any():
            p_after_not_prev[num] = (curr[not_seen_prev, num]).mean()
    overlap = np.mean(np.sum(prev & curr, axis=1)) / 6.0
    return {
        "p_appear_after_previous_draw_contains_number": p_after_prev.tolist(),
        "p_appear_after_previous_draw_does_not_contain_number": p_after_not_prev.tolist(),
        "mean_overlap_with_previous_draw": float(overlap),
        "expected_overlap_random": 6.0 * 6.0 / MAX_NUM,
        "difference_between_conditions": float(np.mean(p_after_prev - p_after_not_prev)),
    }


def compute_pair_triple_summary(data: DrawData) -> dict:
    pair_index = list(itertools.combinations(range(1, MAX_NUM + 1), 2))
    pair_counts: Dict[Tuple[int, int], int] = {}
    for draw in data.draws:
        pairs = list(itertools.combinations(sorted(draw), 2))
        for a, b in pairs:
            key = (a, b) if a < b else (b, a)
            pair_counts[key] = pair_counts.get(key, 0) + 1
    pair_summary = []
    for a, b in pair_index[:10]:
        count = pair_counts.get((a, b), 0)
        expected = data.n / 99.0
        z_score = (count - expected) / math.sqrt(expected * (1.0 - 1.0 / 99.0))
        pair_summary.append({"pair": [a, b], "count": int(count), "expected_count": float(expected), "z_score": float(z_score)})
    triple_counts: Dict[Tuple[int, int, int], int] = {}
    for draw in data.draws:
        s = sorted(draw)
        for tri in itertools.combinations(s, 3):
            triple_counts[tri] = triple_counts.get(tri, 0) + 1
    top_triples = sorted(triple_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {"top_pairs": pair_summary, "top_triples": [{"triple": list(k), "count": v} for k, v in top_triples], "pair_expected_count": float(data.n / 99.0)}


def compute_structure_summary(data: DrawData) -> dict:
    draws = data.draws
    draw_flat = (draws - 1).ravel()
    summary = {
        "sum_mean": float(draws.sum(axis=1).mean()),
        "sum_std": float(draws.sum(axis=1).std()),
        "range_mean": float((draws[:, -1] - draws[:, 0]).mean()),
        "odd_even_counts": np.bincount((draws % 2 == 1).sum(axis=1), minlength=7).tolist(),
        "low_high_counts": np.bincount((draws <= 27).sum(axis=1), minlength=7).tolist(),
        "consecutive_numbers": int((np.diff(draws, axis=1) == 1).sum()),
        "decade_distribution": np.bincount(draw_flat // 10, minlength=6).tolist(),
        "structure_summary": {
            "sum_mean_random": PICK * (MAX_NUM + 1) / 2,
            "sum_std_random": math.sqrt(PICK * ((MAX_NUM**2 - 1) / 12) * (MAX_NUM - PICK) / (MAX_NUM - 1)),
        },
    }
    return summary


def monte_carlo_structure(data: DrawData, reps: int = 20000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    sums = []
    ranges_ = []
    odds = []
    lows = []
    consec = []
    for _ in range(reps):
        combo = np.sort(rng.choice(np.arange(1, MAX_NUM + 1), size=PICK, replace=False))
        sums.append(combo.sum())
        ranges_.append(combo[-1] - combo[0])
        odds.append((combo % 2 == 1).sum())
        lows.append((combo <= 27).sum())
        consec.append(int((np.diff(combo) == 1).sum()))
    observed_sum = data.draws.sum(axis=1)
    return {
        "sum_percentile": float(np.mean(np.asarray(sums) <= observed_sum.mean())),
        "range_percentile": float(np.mean(np.asarray(ranges_) <= (data.draws[:, -1] - data.draws[:, 0]).mean())),
        "odd_count_percentile": float(np.mean(np.asarray(odds) <= (data.draws % 2 == 1).sum(axis=1).mean())),
        "low_count_percentile": float(np.mean(np.asarray(lows) <= (data.draws <= 27).sum(axis=1).mean())),
        "consecutive_percentile": float(np.mean(np.asarray(consec) <= ((np.diff(data.draws, axis=1) == 1).sum(axis=1).mean()))),
        "random_baseline_note": "Monte Carlo provides a random baseline for structural comparisons; it does not imply a higher winning probability.",
    }


def compute_reference_stats(data: DrawData) -> dict:
    freq = compute_frequency_analysis(data)
    gap = compute_gap_analysis(data)
    ewma = compute_ewma_summary(data)
    markov = compute_markov_summary(data)
    pairs = compute_pair_triple_summary(data)
    structure = compute_structure_summary(data)
    mc = monte_carlo_structure(data)
    p = theoretical_probabilities()
    tests = [
        ("frequency_chi_square", freq["chi_square_p"] if freq["chi_square_p"] is not None else 1.0),
        ("gap_current_gap", 1.0),
        ("markov_overlap", 1.0),
        ("pair_support", 1.0),
        ("structural_monte_carlo", 1.0),
    ]
    raw_p = [float(pv) for _, pv in tests]
    adj = bh_adjust(raw_p)
    return {
        "theoretical_probability": p,
        "frequency": freq,
        "gap": gap,
        "ewma": {name: v[-1].tolist() for name, v in ewma.items()},
        "markov": markov,
        "pair_analysis": pairs,
        "structure": structure,
        "monte_carlo": mc,
        "multiple_testing": {
            "tests": [{"name": name, "raw_p": float(pv), "bh_adjusted_p": float(ap)} for (name, pv), ap in zip(tests, adj)],
            "raw_p": raw_p,
            "bh_adjusted": [float(x) for x in adj],
        },
    }


# ----------------------------------------------------------------------------
# 1) Đọc + kiểm tra + làm sạch dữ liệu
# ----------------------------------------------------------------------------
def _norm(name: object) -> str:
    text = str(name).replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", text)


NUM_COL_RE = re.compile(r"^(n|no|num|number|so|ball|bong|b|s|kq)?0?([1-9]\d*)$")
ID_NAMES = {"ky", "kyquay", "kyso", "drawid", "draw", "period", "id", "drawno", "drawnumber", "kyquayso"}
BONUS_HINTS = ("bonus", "dacbiet", "jackpot", "power", "giai", "phu")
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y", "%d/%m/%y", "%m/%d/%Y"]


def read_csv_robust(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=encoding, dtype=str)
            if df.shape[1] == 1:  # dò dấu phân cách thất bại -> thử dấu phẩy
                df = pd.read_csv(path, sep=",", encoding=encoding, dtype=str)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataError(f"File '{path.name}' rỗng.") from exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise DataError(f"Không đọc được file '{path.name}': {last_error}")


def parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.extract(r"(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})", expand=False)
    best, best_ratio = None, -1.0
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best, best_ratio = parsed, ratio
        if ratio >= 0.999:
            break
    if best_ratio < 0.5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return best


def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        n = _norm(col)
        if any(h in n for h in ("date", "ngay", "thoigian")) or n in ("time", "day"):
            return col
    for col in df.columns:  # dò theo nội dung
        if NUM_COL_RE.match(_norm(col)):
            continue
        parsed = parse_dates(df[col])
        if parsed.notna().mean() > 0.9 and parsed.dt.year.between(2000, 2100).mean() > 0.9:
            return col
    return None


def _extract_six_numbers(df: pd.DataFrame, date_col: Optional[str], id_col: Optional[str], name: str) -> pd.DataFrame:
    """Trả về DataFrame cột n1..n6 (float, NaN nếu không hợp lệ)."""
    out = pd.DataFrame(index=df.index)
    # (a) cột theo tên
    named: Dict[int, str] = {}
    for col in df.columns:
        m = NUM_COL_RE.match(_norm(col))
        if m and col not in (date_col, id_col):
            named.setdefault(int(m.group(2)), col)
    if len(named) == 6:
        for i in range(1, 7):
            out[f"n{i}"] = pd.to_numeric(df[named[i]], errors="coerce")
        return out
    # (b) 6 cột số nguyên trong 1..55
    cands = []
    for col in df.columns:
        if col in (date_col, id_col) or any(h in _norm(col) for h in BONUS_HINTS):
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        if num.notna().mean() >= 0.9 and num.between(1, MAX_NUM).mean() >= 0.9:
            cands.append(col)
    if len(cands) >= 6:
        for i, col in enumerate(cands[:6], start=1):
            out[f"n{i}"] = pd.to_numeric(df[col], errors="coerce")
        return out
    # (c) một cột chứa cả dãy số
    best_col, best_ratio = None, 0.0
    for col in df.columns:
        if col in (date_col, id_col):
            continue
        counts = df[col].astype(str).str.findall(r"\d+").str.len()
        ratio = float((counts >= PICK).mean())
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio
    if best_col is not None and best_ratio >= 0.9:
        parts = df[best_col].astype(str).str.findall(r"\d+")
        for i in range(PICK):
            out[f"n{i + 1}"] = parts.apply(lambda xs, i=i: float(xs[i]) if len(xs) > i else np.nan)
        return out
    raise DataError(
        f"File '{name}': không nhận diện được {PICK} cột số. Hãy đặt tên cột là n1..n{PICK} "
        f"hoặc dùng một cột chứa dãy {PICK} số (xem phần GIẢ ĐỊNH ở đầu file ml_model.py)."
    )


@dataclass
class DrawData:
    draws: np.ndarray  # (N, 6) int16, mỗi kỳ sắp tăng dần, giá trị 1..55
    dates: Optional[pd.Series]  # datetime theo từng kỳ (hoặc None)
    frame: pd.DataFrame
    report: dict

    @property
    def n(self) -> int:
        return int(self.draws.shape[0])


def load_and_validate(paths: Sequence[str], game_spec: GameSpec = DEFAULT_GAME) -> DrawData:
    if not paths:
        raise DataError("Chưa chỉ định file CSV. Dùng: --csv file1.csv file2.csv file3.csv")
    frames = []
    report: dict = {"files": [], "canh_bao": [], "game": game_spec.name, "game_label": game_spec.short_name or game_spec.name}
    for order, raw in enumerate(paths):
        path = Path(raw)
        if not path.is_file():
            raise DataError(f"Không tìm thấy file: {path}")
        df = read_csv_robust(path)
        date_col = _find_date_col(df)
        id_col = next((c for c in df.columns if _norm(c) in ID_NAMES), None)
        nums = _extract_six_numbers(df, date_col, id_col, path.name)
        part = nums.copy()
        part["date"] = parse_dates(df[date_col]) if date_col else pd.NaT
        part["draw_id"] = pd.to_numeric(df[id_col], errors="coerce") if id_col else np.nan
        part["file_order"] = order
        part["row_order"] = np.arange(len(df))
        part["source_file"] = path.name
        frames.append(part)
        report["files"].append(
            {"ten_file": path.name, "so_dong": int(len(df)), "cot_ngay": date_col, "cot_ma_ky": id_col}
        )
        log("DATA", f"Đã đọc '{path.name}': {len(df):,} dòng (cột ngày: {date_col or 'không có'})")
    log("DATA", f"Đã đọc {len(paths)} file CSV")

    data = pd.concat(frames, ignore_index=True)
    total_raw = len(data)
    num_cols = [f"n{i}" for i in range(1, PICK + 1)]

    # --- làm sạch số
    data = data.dropna(subset=num_cols)
    dropped_na = total_raw - len(data)
    arr = data[num_cols].to_numpy()
    integral = np.all(np.mod(arr, 1) == 0, axis=1)
    in_range = np.all((arr >= 1) & (arr <= game_spec.max_num), axis=1)
    arr_sorted = np.sort(arr, axis=1)
    distinct = np.all(np.diff(arr_sorted, axis=1) > 0, axis=1)
    good = integral & in_range & distinct
    dropped_bad = int((~good).sum())
    data = data.loc[good].copy()
    data[num_cols] = np.sort(data[num_cols].to_numpy().astype(np.int16), axis=1)

    # --- ngày
    has_dates = data["date"].notna().any()
    dropped_date = 0
    if has_dates:
        before = len(data)
        data = data.dropna(subset=["date"])
        dropped_date = before - len(data)
    else:
        report["canh_bao"].append(
            "Không có cột ngày: dùng thứ tự dòng/file làm thứ tự thời gian. Hãy đảm bảo file được liệt kê từ cũ đến mới."
        )
        log("CẢNH BÁO", report["canh_bao"][-1])

    # --- sắp xếp theo thời gian, loại trùng
    sort_cols = (["date"] if has_dates else []) + ["draw_id", "file_order", "row_order"]
    data = data.sort_values(sort_cols, kind="mergesort", na_position="last")
    before = len(data)
    subset = (["date"] if has_dates else ["file_order", "row_order"]) + num_cols
    if has_dates:
        data = data.drop_duplicates(subset=subset, keep="first")
    dup_exact = before - len(data)
    dup_id = 0
    if data["draw_id"].notna().any():
        before = len(data)
        has_id = data["draw_id"].notna()
        data = pd.concat([data[~has_id], data[has_id].drop_duplicates(subset=["draw_id"], keep="first")])
        data = data.sort_values(sort_cols, kind="mergesort", na_position="last")
        dup_id = before - len(data)
    if has_dates:
        same_day = int(data["date"].duplicated(keep=False).sum())
        if same_day:
            msg = f"Có {same_day:,} dòng cùng ngày nhưng khác kết quả (vẫn giữ lại, xếp theo mã kỳ)."
            report["canh_bao"].append(msg)
            log("CẢNH BÁO", msg)
    data = data.reset_index(drop=True)

    n = len(data)
    min_needed = 120
    if n < min_needed:
        raise DataError(
            f"Chỉ có {n:,} kỳ quay hợp lệ, cần tối thiểu {min_needed:,} kỳ để phân tích walk-forward."
        )
    report.update(
        {
            "tong_dong_doc": int(total_raw),
            "loai_do_thieu_so": int(dropped_na),
            "loai_do_so_khong_hop_le": int(dropped_bad),
            "loai_do_thieu_ngay": int(dropped_date),
            "loai_do_trung_lap": int(dup_exact + dup_id),
            "so_ky_hop_le": int(n),
            "tu_ngay": data["date"].min() if has_dates else None,
            "den_ngay": data["date"].max() if has_dates else None,
        }
    )
    log(
        "VALIDATION",
        f"Tổng {total_raw:,} dòng | loại: thiếu số {dropped_na:,}, số sai {dropped_bad:,}, "
        f"thiếu ngày {dropped_date:,}, trùng {dup_exact + dup_id:,}",
    )
    log("VALIDATION", f"Hợp lệ: {n:,} kỳ quay")
    if has_dates:
        log("VALIDATION", f"Khoảng thời gian: {data['date'].min():%d/%m/%Y} -> {data['date'].max():%d/%m/%Y}")

    draws = data[num_cols].to_numpy().astype(np.int16)
    return DrawData(draws=draws, dates=data["date"] if has_dates else None, frame=data, report=report)


# ----------------------------------------------------------------------------
# 2) Thống kê tích lũy + đặc trưng (chỉ dùng dữ liệu đã thấy)
# ----------------------------------------------------------------------------
class HistoryTracker:
    """Trạng thái thống kê tích lũy. Chỉ nhận dữ liệu qua update() nên KHÔNG thể nhìn thấy tương lai."""

    def __init__(self, capacity: int) -> None:
        self.t = 0
        self.draws0 = np.zeros((capacity, PICK), dtype=np.int16)  # số 0-based
        self.X = np.zeros((capacity, MAX_NUM), dtype=np.bool_)
        self.count = np.zeros(MAX_NUM, dtype=np.int64)
        self.last_seen = np.full(MAX_NUM, -1, dtype=np.int64)
        self.P = np.zeros((MAX_NUM, MAX_NUM), dtype=np.int32)  # đếm cặp
        self.T = np.zeros((MAX_NUM, MAX_NUM, MAX_NUM), dtype=np.int32)  # đếm bộ ba
        self.sums = np.zeros(capacity, dtype=np.float64)

    def update(self, draw0: np.ndarray) -> None:
        t = self.t
        d = np.asarray(draw0, dtype=np.int64)
        self.draws0[t] = d
        self.X[t, d] = True
        self.count[d] += 1
        self.last_seen[d] = t
        a, b = d[PA], d[PB]
        self.P[a, b] += 1
        self.P[b, a] += 1
        tri = d[TRIPLE_PERM_POS]
        self.T[tri[:, 0], tri[:, 1], tri[:, 2]] += 1
        self.sums[t] = float(d.sum() + PICK)
        self.t += 1

    def history_numbers(self) -> np.ndarray:
        """Các kỳ đã thấy, số 1..55, shape (t, 6)."""
        return self.draws0[: self.t].astype(np.int16) + 1

    def snapshot(self) -> np.ndarray:
        """Ma trận đặc trưng (55, N_FEATURES) cho kỳ kế tiếp, chỉ từ các kỳ < t."""
        t = self.t
        if t < 2:
            raise ValueError("Cần ít nhất 2 kỳ để tạo đặc trưng.")
        ks = np.arange(MAX_NUM)
        nums = ks + 1
        cnt = self.count.astype(np.float64)
        last = self.draws0[t - 1].astype(np.int64)

        def win_rate(w: int) -> np.ndarray:
            lo = max(0, t - w)
            return self.X[lo:t].mean(axis=0)

        f10, f30, f100 = win_rate(10), win_rate(30), win_rate(100)
        seen = self.last_seen >= 0
        delay = np.where(seen, t - 1 - self.last_seen, t).astype(np.float64)
        exp_gap = (t + 2) / (cnt + 1)
        n30 = min(30, t)
        hot_z = (f30 - P0) / math.sqrt(P0 * (1 - P0) / n30)
        hot_rank = pd.Series(f30).rank(pct=True).to_numpy()
        cold_rank = pd.Series(delay).rank(pct=True).to_numpy()

        exp_pair = t / 99.0  # P(cặp cụ thể cùng xuất hiện) = 1/99
        pl = self.P[:, last]
        mask_pair = last[None, :] != ks[:, None]
        pair_aff = pl.sum(axis=1) / np.maximum(mask_pair.sum(axis=1), 1) / (exp_pair + 1.0)
        pair_top = np.partition(self.P, -5, axis=1)[:, -5:].mean(axis=1) / (exp_pair + 1.0)

        exp_tri = t * math.comb(MAX_NUM - 3, PICK - 3) / TOTAL_COMBOS
        ia, ib = last[PA], last[PB]
        tk = self.T[:, ia, ib]  # (55, 15)
        mask_tri = (ks[:, None] != ia[None, :]) & (ks[:, None] != ib[None, :])
        triplet_aff = tk.sum(axis=1) / np.maximum(mask_tri.sum(axis=1), 1) / (exp_tri + 1.0)

        in_last = self.X[t - 1].astype(np.float64)
        in_last2 = self.X[t - 2].astype(np.float64)

        lo = max(0, t - 30)
        odd_frac = float(((self.draws0[lo:t] + 1) & 1).mean())
        exp_odd = 28 / 55
        parity_pressure = np.where(nums % 2 == 1, exp_odd - odd_frac, odd_frac - exp_odd)
        recent_sum_z = (self.sums[lo:t].mean() - SUM_MEAN) / (SUM_STD / math.sqrt(t - lo))
        last_sum_z = (self.sums[t - 1] - SUM_MEAN) / SUM_STD
        sum_pull = -recent_sum_z * (nums - 28) / 27.0

        dist = np.abs(nums[:, None] - (last[None, :] + 1))
        min_dist = dist.min(axis=1).astype(np.float64)
        mean_dist = dist.mean(axis=1)
        nb = np.zeros(MAX_NUM)
        nb[1:] += in_last[:-1]
        nb[:-1] += in_last[1:]

        group = ks // 10
        sizes = np.bincount(group).astype(np.float64)
        hits = np.bincount(group, weights=self.X[lo:t].sum(axis=0).astype(np.float64))
        decade_heat = (hits / (n30 * sizes) / P0)[group]
        cons_rate = float((np.diff(self.draws0[lo:t], axis=1) == 1).any(axis=1).mean())

        cols = [
            nums / MAX_NUM, (nums % 2).astype(np.float64),
            cnt / t, f10, f30, f100,
            delay, delay / exp_gap,
            hot_z, hot_rank, cold_rank,
            pair_aff, pair_top, triplet_aff,
            in_last, in_last2,
            parity_pressure, np.full(MAX_NUM, recent_sum_z), np.full(MAX_NUM, last_sum_z), sum_pull,
            min_dist, mean_dist, nb,
            decade_heat, np.full(MAX_NUM, cons_rate),
        ]
        return np.stack(cols, axis=1).astype(np.float32)


def iter_history(data: DrawData) -> Iterator[Tuple[int, HistoryTracker, np.ndarray]]:
    """
    Duyệt tuần tự theo thời gian. Ở bước t (0..N) trả về trạng thái CHỈ chứa các kỳ < t
    và đặc trưng feats[t]. Bước t = N là đặc trưng cho kỳ TIẾP THEO (chưa có kết quả).
    """
    n = data.n
    tracker = HistoryTracker(n + 1)
    feats = np.zeros((n + 1, MAX_NUM, N_FEATURES), dtype=np.float32)
    for t in range(n + 1):
        if t >= 2:
            feats[t] = tracker.snapshot()
        yield t, tracker, feats
        if t < n:
            tracker.update(data.draws[t] - 1)


def build_labels(draws: np.ndarray) -> np.ndarray:
    labels = np.zeros((draws.shape[0], MAX_NUM), dtype=np.int8)
    rows = np.arange(draws.shape[0])[:, None]
    labels[rows, draws - 1] = 1
    return labels


def print_statistics_summary(data: DrawData) -> dict:
    """Thống kê tổng quan (tần suất, số lẻ/chẵn, tổng...), có kiểm định chi-bình phương nếu có SciPy."""
    counts = np.bincount(data.draws.ravel(), minlength=MAX_NUM + 1)[1:]
    expected = data.n * PICK / MAX_NUM
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p_value = None
    try:
        from scipy import stats as sps  # type: ignore

        p_value = float(sps.chi2.sf(chi2, MAX_NUM - 1))
    except Exception:  # noqa: BLE001
        pass
    odd_counts = np.bincount((data.draws & 1).sum(axis=1), minlength=7)
    sums = data.draws.sum(axis=1)
    order = np.argsort(-counts)
    summary = {
        "so_ky": data.n,
        "so_xuat_hien_nhieu_nhat": [(int(k + 1), int(counts[k])) for k in order[:6]],
        "so_xuat_hien_it_nhat": [(int(k + 1), int(counts[k])) for k in order[-6:]],
        "chi_binh_phuong_deu": chi2,
        "p_value_chi_binh_phuong": p_value,
        "phan_bo_so_le_0_den_6": odd_counts.tolist(),
        "tong_trung_binh": float(sums.mean()),
        "tong_ky_vong_ngau_nhien": SUM_MEAN,
        "ghi_chu": "p-value lớn nghĩa là chưa thấy bằng chứng tần suất lệch khỏi phân phối đều.",
    }
    log("THỐNG KÊ", f"Số ra nhiều nhất: {', '.join(f'{k:02d}({c})' for k, c in summary['so_xuat_hien_nhieu_nhat'])}")
    log("THỐNG KÊ", f"Số ra ít nhất: {', '.join(f'{k:02d}({c})' for k, c in summary['so_xuat_hien_it_nhat'])}")
    if p_value is not None:
        log("THỐNG KÊ", f"Kiểm định chi-bình phương độ đều của tần suất: p = {p_value:.4f}")
    log("THỐNG KÊ", f"Tổng 6 số trung bình: {sums.mean():.1f} (kỳ vọng ngẫu nhiên {SUM_MEAN:.0f})")
    return summary


# ----------------------------------------------------------------------------
# 3) Mô hình ML
# ----------------------------------------------------------------------------
def build_models(cfg: Config) -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features="sqrt",
            n_jobs=cfg.n_jobs,
            random_state=cfg.seed,
        ),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=300)),
    }


def fit_models(feats: np.ndarray, labels: np.ndarray, start: int, stop: int, cfg: Config) -> dict:
    """Huấn luyện trên các kỳ [start, stop). Người gọi đảm bảo stop <= t hiện tại (không rò rỉ)."""
    if stop - start < 20:
        raise DataError("Không đủ dữ liệu để huấn luyện mô hình.")
    x = feats[start:stop].reshape(-1, N_FEATURES)
    y = labels[start:stop].reshape(-1)
    models = build_models(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model in models.values():
            model.fit(x, y)
    if hasattr(models["random_forest"], "set_params"):
        models["random_forest"].set_params(n_jobs=1)  # dự đoán 55 dòng: 1 luồng nhanh hơn
    return models


def predict_probabilities(models: dict, features_t: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: model.predict_proba(features_t)[:, 1] for name, model in models.items()}


def normalize_signal(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if np.isclose(hi, lo):
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def build_strategy_signal_bundle(probs: Dict[str, np.ndarray], feat_t: np.ndarray) -> Dict[str, np.ndarray]:
    bundle: Dict[str, np.ndarray] = {}
    bundle["ml"] = 0.5 * (probs["random_forest"] + probs["logistic_regression"])
    bundle["frequency"] = normalize_signal(feat_t[:, FI["freq_w100"]]) if "freq_w100" in FI else np.zeros(MAX_NUM, dtype=np.float64)
    bundle["ewma"] = normalize_signal(feat_t[:, FI["freq_w30"]]) if "freq_w30" in FI else np.zeros(MAX_NUM, dtype=np.float64)
    bundle["gap"] = normalize_signal(feat_t[:, FI["delay"]]) if "delay" in FI else np.zeros(MAX_NUM, dtype=np.float64)
    bundle["markov"] = normalize_signal(feat_t[:, FI["in_last"]]) if "in_last" in FI else np.zeros(MAX_NUM, dtype=np.float64)
    bundle["pair"] = normalize_signal(feat_t[:, FI["pair_aff"]]) if "pair_aff" in FI else np.zeros(MAX_NUM, dtype=np.float64)
    bundle["struct"] = normalize_signal(feat_t[:, FI["sum_pull"]]) if "sum_pull" in FI else np.zeros(MAX_NUM, dtype=np.float64)
    bundle["balanced_ensemble"] = (
        0.40 * bundle["ml"]
        + 0.30 * bundle["frequency"]
        + 0.20 * bundle["gap"]
        + 0.10 * bundle["struct"]
    )
    bundle["ml_focus"] = 0.70 * bundle["ml"] + 0.30 * bundle["frequency"]
    bundle["statistical_focus"] = (
        0.35 * bundle["frequency"]
        + 0.25 * bundle["ewma"]
        + 0.20 * bundle["gap"]
        + 0.10 * bundle["markov"]
        + 0.10 * bundle["pair"]
    )
    bundle["diversified"] = (
        0.20 * bundle["ml"]
        + 0.20 * bundle["frequency"]
        + 0.20 * bundle["ewma"]
        + 0.20 * bundle["gap"]
        + 0.10 * bundle["markov"]
        + 0.10 * bundle["pair"]
    )
    bundle["alternative"] = (
        0.30 * bundle["gap"]
        + 0.30 * bundle["frequency"]
        + 0.20 * bundle["struct"]
        + 0.20 * bundle["ml"]
    )
    return bundle


def top6(scores: np.ndarray) -> np.ndarray:
    idx = np.argsort(-scores, kind="stable")[:PICK]
    return np.sort(idx) + 1


# ----------------------------------------------------------------------------
# 4) Động cơ tổ hợp 6/55 (sinh theo lô, chấm điểm vectorized, giữ Top-K)
# ----------------------------------------------------------------------------
STRUCT_BINS = [7, 30, 10, 6, 6]  # số lẻ, tổng, khoảng cách max-min, cặp liên tiếp, số chục khác nhau
_UNIFORM_CACHE: Dict[int, List[np.ndarray]] = {}


def random_combinations(n: int, rng: np.random.Generator, chunk: int = 50_000) -> np.ndarray:
    out = np.empty((n, PICK), dtype=np.int8)
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        r = rng.random((m, MAX_NUM), dtype=np.float32)
        idx = np.argpartition(r, PICK - 1, axis=1)[:, :PICK]
        idx.sort(axis=1)
        out[s : s + m] = idx + 1
    return out


def struct_indices(c: np.ndarray) -> List[np.ndarray]:
    """Chỉ số cấu trúc của các tổ hợp đã sắp tăng dần (B, 6), số 1..55."""
    c16 = c.astype(np.int16)
    odd = (c16 & 1).sum(axis=1)
    sum_bin = np.clip((c16.sum(axis=1) - 21) // 10, 0, STRUCT_BINS[1] - 1)
    span_bin = np.clip((c16[:, -1] - c16[:, 0] - 5) // 5, 0, STRUCT_BINS[2] - 1)
    consec = (np.diff(c16, axis=1) == 1).sum(axis=1)
    dec = (c16 - 1) // 10
    ndec = (np.diff(dec, axis=1) > 0).sum(axis=1)  # 0..5 = số nhóm chục khác nhau - 1
    return [odd, sum_bin, span_bin, consec, ndec]


def uniform_struct_pmf(seed: int = 12345, n: int = 200_000) -> List[np.ndarray]:
    if seed not in _UNIFORM_CACHE:
        combos = random_combinations(n, np.random.default_rng(seed))
        _UNIFORM_CACHE[seed] = [
            (np.bincount(idx, minlength=nb) + 1.0) / (n + nb)
            for idx, nb in zip(struct_indices(combos), STRUCT_BINS)
        ]
    return _UNIFORM_CACHE[seed]


@dataclass
class ScoreContext:
    label: str
    pair_table: np.ndarray  # float32 (55*55,)
    triple_table: Optional[np.ndarray]  # float32 (55**3,) hoặc None
    struct_tables: List[np.ndarray]
    actual: Optional[np.ndarray] = None  # kết quả thật (để xếp hạng chính xác)


def build_score_context(
    tracker: HistoryTracker, number_probs: np.ndarray, cfg: Config, label: str, actual: Optional[np.ndarray] = None
) -> ScoreContext:
    """
    Điểm tổ hợp = w_number * tổng log(p_k / p0)   (xác suất từ Random Forest)
                + w_pair   * trung bình log-lift của 15 cặp
                + w_triple * trung bình log-lift của 20 bộ ba
                + w_struct * tổng log-tỉ-số-khả-dĩ của 5 đặc trưng cấu trúc (lẻ/chẵn, tổng, khoảng cách, liên tiếp, nhóm chục)
    Mọi bảng chỉ xây từ các kỳ đã thấy (tracker.t kỳ đầu tiên).
    """
    t = tracker.t
    p = np.clip(number_probs.astype(np.float64), 1e-4, 1 - 1e-4)
    s = np.log(p / P0)
    exp_pair = t / 99.0
    lift_pair = np.log((tracker.P + 1.0) / (exp_pair + 1.0))
    # Gộp phần số đơn vào ma trận cặp: mỗi số nằm trong 5 cặp
    w_mat = cfg.w_number * (s[:, None] + s[None, :]) / (PICK - 1) + cfg.w_pair * lift_pair / len(PAIR_POS)
    triple_table = None
    if cfg.w_triple > 0:
        exp_tri = t * math.comb(MAX_NUM - 3, PICK - 3) / TOTAL_COMBOS
        triple_table = (np.log((tracker.T + 1.0) / (exp_tri + 1.0)) * (cfg.w_triple / len(TRIPLE_POS))).astype(
            np.float32
        ).ravel()
    ref = uniform_struct_pmf()
    tables = []
    for idx, nb, ref_pmf in zip(struct_indices(tracker.history_numbers()), STRUCT_BINS, ref):
        hc = np.bincount(idx, minlength=nb) + 1.0
        tables.append(((np.log(hc / hc.sum()) - np.log(ref_pmf)) * cfg.w_struct).astype(np.float32))
    return ScoreContext(label, w_mat.astype(np.float32).ravel(), triple_table, tables, actual)


def score_batch(c: np.ndarray, ctx: ScoreContext) -> np.ndarray:
    """Chấm điểm vectorized một lô tổ hợp (B, 6) int8, số 1..55 tăng dần -> float32 (B,)."""
    z = c.astype(np.int32) - 1
    z55 = z * MAX_NUM
    score = np.zeros(c.shape[0], dtype=np.float32)
    for i, j in PAIR_POS:
        score += ctx.pair_table[z55[:, i] + z[:, j]]
    if ctx.triple_table is not None:
        z3025 = z55 * MAX_NUM
        for i, j, k in TRIPLE_POS:
            score += ctx.triple_table[z3025[:, i] + z55[:, j] + z[:, k]]
    for table, idx in zip(ctx.struct_tables, struct_indices(c)):
        score += table[idx]
    return score


def iter_combination_batches(batch_size: int, limit: int = 0) -> Iterator[np.ndarray]:
    """Sinh tổ hợp 6/55 bằng itertools.combinations theo lô; mỗi lô là mảng int8 (B, 6)."""
    total = TOTAL_COMBOS if limit <= 0 else min(limit, TOTAL_COMBOS)
    iterator = itertools.combinations(range(1, MAX_NUM + 1), PICK)
    done = 0
    while done < total:
        n = min(batch_size, total - done)
        flat = np.fromiter(
            itertools.chain.from_iterable(itertools.islice(iterator, n)), dtype=np.int8, count=n * PICK
        )
        yield flat.reshape(n, PICK)
        done += n


class TopK:
    def __init__(self, k: int) -> None:
        self.k = k
        self.scores = np.empty(0, dtype=np.float32)
        self.combos = np.empty((0, PICK), dtype=np.int8)

    def push(self, scores: np.ndarray, combos: np.ndarray) -> None:
        if self.scores.size >= self.k:  # chỉ xét phần tử có thể vào Top-K
            keep = scores > self.scores.min()
            if not keep.any():
                return
            scores, combos = scores[keep], combos[keep]
        if scores.size > self.k:
            idx = np.argpartition(scores, -self.k)[-self.k :]
            scores, combos = scores[idx], combos[idx]
        self.scores = np.concatenate([self.scores, scores])
        self.combos = np.concatenate([self.combos, combos])
        if self.scores.size > self.k:
            idx = np.argpartition(self.scores, -self.k)[-self.k :]
            self.scores, self.combos = self.scores[idx], self.combos[idx]

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(-self.scores, kind="stable")
        return self.scores[order], self.combos[order]


def run_combination_engine(contexts: List[ScoreContext], cfg: Config) -> List[dict]:
    """Quét toàn bộ không gian tổ hợp một lần cho nhiều ngữ cảnh chấm điểm; giữ Top-K mỗi ngữ cảnh."""
    total = TOTAL_COMBOS if cfg.combo_limit <= 0 else min(cfg.combo_limit, TOTAL_COMBOS)
    heaps = [TopK(cfg.top_k) for _ in contexts]
    actual_scores = [
        float(score_batch(ctx.actual[None, :].astype(np.int8), ctx)[0]) if ctx.actual is not None else None
        for ctx in contexts
    ]
    greater = [0] * len(contexts)
    equal = [0] * len(contexts)
    approx_mb = cfg.batch_size * 110 / 1e6
    log("COMBO", f"Bắt đầu quét {total:,} tổ hợp | lô {cfg.batch_size:,} (~{approx_mb:.0f} MB/lô) | Top-K = {cfg.top_k:,} | {len(contexts)} ngữ cảnh")
    if cfg.combo_limit > 0:
        log("CẢNH BÁO", f"Chế độ thử nghiệm: chỉ quét {total:,} tổ hợp đầu tiên (theo thứ tự từ điển), KHÔNG phải toàn bộ.")
    start = time.time()
    last_print = start
    done = 0
    for batch in iter_combination_batches(cfg.batch_size, cfg.combo_limit):
        for i, ctx in enumerate(contexts):
            scores = score_batch(batch, ctx)
            if actual_scores[i] is not None:
                greater[i] += int(np.count_nonzero(scores > actual_scores[i]))
                equal[i] += int(np.count_nonzero(scores == actual_scores[i]))
            heaps[i].push(scores, batch)
        done += batch.shape[0]
        now = time.time()
        if now - last_print >= 5 or done >= total:
            speed = done / max(now - start, 1e-9)
            eta = (total - done) / max(speed, 1e-9)
            log("COMBO", f"Đã xử lý {done:,}/{total:,} ({100 * done / total:.1f}%) | {speed:,.0f} tổ hợp/giây | còn ~{eta:.0f}s")
            last_print = now
    elapsed = time.time() - start
    log("COMBO", f"Hoàn thành quét {done:,} tổ hợp trong {elapsed:.1f}s")
    results = []
    for i, ctx in enumerate(contexts):
        scores, combos = heaps[i].result()
        item = {"label": ctx.label, "scores": scores, "combos": combos, "seconds": elapsed, "scanned": done}
        if actual_scores[i] is not None:
            rank = greater[i] + 1 + equal[i] / 2.0
            item.update(
                {
                    "actual_score": actual_scores[i],
                    "actual_rank": rank,
                    "actual_percentile": rank / done,
                }
            )
        results.append(item)
    return results


# ----------------------------------------------------------------------------
# 5) Đánh giá: so khớp, baseline ngẫu nhiên, kiểm định
# ----------------------------------------------------------------------------
def actual_mask(actual: Sequence[int]) -> np.ndarray:
    mask = np.zeros(MAX_NUM + 1, dtype=bool)
    mask[np.asarray(actual, dtype=np.int64)] = True
    return mask


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def match_distribution_table(counts: np.ndarray) -> List[dict]:
    total = int(counts.sum())
    return [
        {
            "so_khop": f"{m}/6",
            "so_lan": int(counts[m]),
            "ty_le": float(counts[m] / total) if total else None,
            "ty_le_ngau_nhien_ly_thuyet": float(HYPER_PMF[m]),
        }
        for m in range(PICK + 1)
    ]


def strategy_stats(match_list: List[int]) -> dict:
    arr = np.asarray(match_list, dtype=np.int64)
    n = len(arr)
    counts = np.bincount(arr, minlength=PICK + 1)
    mean = float(arr.mean()) if n else float("nan")
    se = math.sqrt(HYPER_VAR / n) if n else float("nan")
    z = (mean - HYPER_MEAN) / se if n else float("nan")
    return {
        "so_ky": n,
        "khop_trung_binh": mean,
        "khop_ky_vong_ngau_nhien": HYPER_MEAN,
        "z": z,
        "p_value_mot_phia": normal_sf(z) if n else None,
        "phan_bo": match_distribution_table(counts),
        "so_ky_khop_tu_3": int(counts[3:].sum()),
        "counts": counts.tolist(),
    }


# ----------------------------------------------------------------------------
# 6) Walk-forward
# ----------------------------------------------------------------------------
def run_walk_forward(data: DrawData, cfg: Config) -> dict:
    n = data.n
    eval_start = max(cfg.warmup + 30, n - cfg.eval_steps)
    steps = n - eval_start
    if steps < 20:
        raise DataError(
            f"Không đủ dữ liệu cho walk-forward: chỉ có {steps} kỳ để đánh giá "
            f"(cần ≥ 20). Hãy giảm --warmup hoặc bổ sung dữ liệu."
        )
    labels = build_labels(data.draws)
    rng = np.random.default_rng(cfg.seed)
    sample = random_combinations(cfg.combo_sample, rng) if cfg.combo_sample > 0 else None
    log("WALK-FORWARD", f"Đang đánh giá theo thời gian: {steps:,} kỳ (từ kỳ #{eval_start + 1:,} đến #{n:,}), huấn luyện lại mỗi {cfg.retrain_every} kỳ")
    log("WALK-FORWARD", "Mỗi kỳ chỉ dùng dữ liệu của các kỳ TRƯỚC nó (không dùng dữ liệu tương lai)")

    rows: List[dict] = []
    matches: Dict[str, List[int]] = {s: [] for s in STRATEGIES}
    mc_counts = np.zeros(PICK + 1, dtype=np.int64)
    auc = {m: [] for m in MODEL_NAMES}
    brier = {m: [] for m in MODEL_NAMES}
    brier_const: List[float] = []
    percentiles: List[float] = []
    models = None
    t_start = time.time()
    log_every = max(1, steps // 10)

    for t, tracker, feats in iter_history(data):
        if t < eval_start:
            continue
        if t >= n:
            break
        step = t - eval_start
        if models is None or step % cfg.retrain_every == 0:
            lo = max(cfg.warmup, t - cfg.train_window)
            log("ML", f"Đang huấn luyện mô hình (kỳ #{t + 1:,}, dùng kỳ #{lo + 1:,}..#{t:,} = {t - lo:,} kỳ)")
            models = fit_models(feats, labels, lo, t, cfg)
        f_t = feats[t]
        probs = predict_probabilities(models, f_t)
        actual = data.draws[t]
        mask = actual_mask(actual)
        y = labels[t].astype(np.float64)

        signal_bundle = build_strategy_signal_bundle(probs, f_t)
        tickets = {
            "random_forest": top6(probs["random_forest"]),
            "logistic_regression": top6(probs["logistic_regression"]),
            "frequency_100": top6(f_t[:, FI["freq_w100"]]),
            "overdue": top6(f_t[:, FI["delay"]]),
            "balanced_ensemble": top6(signal_bundle["balanced_ensemble"]),
            "ml_focus": top6(signal_bundle["ml_focus"]),
            "statistical_focus": top6(signal_bundle["statistical_focus"]),
            "diversified": top6(signal_bundle["diversified"]),
            "alternative": top6(signal_bundle["alternative"]),
            "random": np.sort(rng.choice(MAX_NUM, PICK, replace=False)) + 1,
        }
        row = {"draw_index": t + 1, "date": data.dates.iloc[t] if data.dates is not None else None, "actual": fmt_ticket(actual)}
        for name, ticket in tickets.items():
            m = int(mask[ticket].sum())
            matches[name].append(m)
            row[f"{name}_ticket"] = fmt_ticket(ticket)
            row[f"{name}_matches"] = m
        # baseline ngẫu nhiên mô phỏng (nhiều vé)
        mc = random_combinations(cfg.mc_tickets, rng).astype(np.int64)
        mc_counts += np.bincount(mask[mc].sum(axis=1), minlength=PICK + 1)
        for name in MODEL_NAMES:
            p = probs[name]
            auc[name].append(float(roc_auc_score(y, p)))
            brier[name].append(float(np.mean((p - y) ** 2)))
            row[f"auc_{name}"] = auc[name][-1]
        brier_const.append(float(np.mean((P0 - y) ** 2)))
        if sample is not None:
            ctx = build_score_context(tracker, probs["random_forest"], cfg, "wf")
            sc = score_batch(sample, ctx)
            a_sc = score_batch(actual[None, :].astype(np.int8), ctx)[0]
            pct = float((np.count_nonzero(sc > a_sc) + 0.5 * np.count_nonzero(sc == a_sc)) / sc.size)
            percentiles.append(pct)
            row["combo_percentile"] = pct
        rows.append(row)
        if (step + 1) % log_every == 0 or step + 1 == steps:
            log("WALK-FORWARD", f"Tiến độ {step + 1:,}/{steps:,} kỳ ({100 * (step + 1) / steps:.0f}%) | {time.time() - t_start:.0f}s")

    stats = {name: strategy_stats(matches[name]) for name in STRATEGIES}
    mc_dist = match_distribution_table(mc_counts)
    model_quality = {}
    for name in MODEL_NAMES:
        b_model, b_const = float(np.mean(brier[name])), float(np.mean(brier_const))
        se_auc = float(np.std(auc[name], ddof=1) / math.sqrt(len(auc[name])))
        model_quality[name] = {
            "auc_trung_binh": float(np.mean(auc[name])),
            "auc_ngau_nhien": 0.5,
            "auc_z": (float(np.mean(auc[name])) - 0.5) / se_auc if se_auc > 0 else None,
            "brier_mo_hinh": b_model,
            "brier_hang_so": b_const,
            "cai_thien_brier": 1.0 - b_model / b_const,
        }
    combo_summary = None
    if percentiles:
        mean_pct = float(np.mean(percentiles))
        z = (mean_pct - 0.5) / math.sqrt(1 / 12 / len(percentiles))
        combo_summary = {
            "so_ky": len(percentiles),
            "phan_vi_trung_binh": mean_pct,
            "phan_vi_ngau_nhien": 0.5,
            "z": z,
            "p_value_mot_phia": 1.0 - normal_sf(z),
            "giai_thich": (
                "Phân vị = tỉ lệ tổ hợp ngẫu nhiên có điểm CAO hơn tổ hợp thực tế; "
                "nhỏ hơn 0.5 nghĩa là điểm số xếp kết quả thật cao hơn ngẫu nhiên."
            ),
            "so_mau_moi_ky": cfg.combo_sample,
        }
    conclusion = build_conclusion(stats, model_quality, combo_summary)
    summary = {
        "thoi_diem": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "canh_bao": WARNING_VI,
        "so_ky_du_lieu": n,
        "so_ky_danh_gia": steps,
        "cau_hinh": asdict(cfg),
        "chien_luoc": {STRATEGY_VI[k]: v for k, v in stats.items()},
        "baseline_ngau_nhien_mo_phong": {"so_ve_moi_ky": cfg.mc_tickets, "phan_bo": mc_dist},
        "chat_luong_mo_hinh": {STRATEGY_VI[k]: v for k, v in model_quality.items()},
        "xep_hang_to_hop": combo_summary,
        "ket_luan": conclusion,
    }
    return {"summary": summary, "rows": rows, "stats": stats, "mc_dist": mc_dist, "mc_counts": mc_counts}


def fmt_optional_float(value: Optional[float], fmt: str = ".4f", default: str = "n/a") -> str:
    if value is None or not isinstance(value, (int, float, np.integer, np.floating)):
        return default
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return default
    return format(float(value), fmt)


def build_conclusion(stats: dict, quality: dict, combo: Optional[dict]) -> str:
    alpha = 0.05 / max(1, len(stats) - 1)  # hiệu chỉnh Bonferroni (bỏ chiến lược ngẫu nhiên)
    better = [
        STRATEGY_VI[k]
        for k, v in stats.items()
        if k != "random" and v["p_value_mot_phia"] is not None and v["p_value_mot_phia"] < alpha
    ]
    if not better:
        text = (
            "Không có chiến lược nào vượt ngẫu nhiên một cách có ý nghĩa thống kê (sau hiệu chỉnh nhiều phép thử). "
            "Điều này phù hợp với bản chất ngẫu nhiên của xổ số."
        )
    else:
        text = (
            f"Có tín hiệu cao hơn ngẫu nhiên ở: {', '.join(better)}. Cần rất thận trọng: nhiều mô hình được thử, "
            "các kỳ liên tiếp không hoàn toàn độc lập; hãy kiểm chứng lại trên dữ liệu mới trước khi tin kết quả này."
        )
    auc_txt = "; ".join(
        f"{STRATEGY_VI[k]} AUC={fmt_optional_float(v.get('auc_trung_binh'), '.4f')}"
        for k, v in quality.items()
    )
    text += f" Chất lượng xác suất từng số: {auc_txt} (0.5 = ngẫu nhiên)."
    if combo:
        text += f" Phân vị trung bình của kết quả thật theo điểm tổ hợp: {fmt_optional_float(combo.get('phan_vi_trung_binh'), '.3f')} (0.5 = ngẫu nhiên)."
    return text


def print_walk_forward_report(result: dict) -> None:
    summary = result["summary"]
    print()
    log("KẾT QUẢ", f"Walk-forward trên {summary['so_ky_danh_gia']:,} kỳ. Số khớp trung bình kỳ vọng nếu chọn ngẫu nhiên = {HYPER_MEAN:.4f}")
    for name in STRATEGIES:
        st = result["stats"][name]
        z_text = "n/a" if st.get("z") is None else f"{st.get('z'):+.2f}"
        p_text = "n/a" if st.get("p_value_mot_phia") is None else f"{st.get('p_value_mot_phia'):.3f}"
        log(
            "KẾT QUẢ",
            f"{STRATEGY_VI[name]:<22} khớp TB = {fmt_optional_float(st.get('khop_trung_binh'), '.4f')} | z = {z_text} | p(một phía) = {p_text} | ≥3/6: {st.get('so_ky_khop_tu_3', 0)} kỳ",
        )
    header = "Số khớp | " + " | ".join(f"{STRATEGY_VI[s][:14]:>14}" for s in STRATEGIES) + " | Ngẫu nhiên mô phỏng | Lý thuyết"
    log("KẾT QUẢ", "Phân bố số khớp (số kỳ):")
    print("          " + header)
    mc_total = max(int(result["mc_counts"].sum()), 1)
    for m in range(PICK + 1):
        cells = " | ".join(f"{result['stats'][s]['counts'][m]:>14,}" for s in STRATEGIES)
        print(f"          {m}/6     | {cells} | {result['mc_counts'][m] / mc_total:>19.4%} | {HYPER_PMF[m]:.4%}")
    for name, q in summary["chat_luong_mo_hinh"].items():
        auc_z = q.get("auc_z")
        z_text = "n/a" if auc_z is None else f"{auc_z:+.2f}"
        improvement = q.get("cai_thien_brier")
        improvement_text = "n/a" if improvement is None else f"{improvement:+.4%}"
        log("KẾT QUẢ", f"{name}: AUC = {fmt_optional_float(q.get('auc_trung_binh'), '.4f')} (z={z_text}), cải thiện Brier so với hằng số = {improvement_text}")
    if summary["xep_hang_to_hop"]:
        c = summary["xep_hang_to_hop"]
        z_text = "n/a" if c.get("z") is None else f"{c.get('z'):+.2f}"
        log("KẾT QUẢ", f"Phân vị tổ hợp thực tế (thấp = tốt): {fmt_optional_float(c.get('phan_vi_trung_binh'), '.3f')} (ngẫu nhiên = 0.5, z = {z_text})")
    log("KẾT QUẢ", summary["ket_luan"])
    log("LƯU Ý", WARNING_VI)


def save_walk_forward(result: dict, out: Path) -> None:
    df = pd.DataFrame(result["rows"])
    rename = {"draw_index": "Kỳ thứ", "date": "Ngày quay", "actual": "Kết quả thực tế", "combo_percentile": "Phân vị tổ hợp thực tế (thấp = tốt)"}
    for s in STRATEGIES:
        rename[f"{s}_ticket"] = f"Vé dự đoán - {STRATEGY_VI[s]}"
        rename[f"{s}_matches"] = f"Số khớp - {STRATEGY_VI[s]}"
    for m in MODEL_NAMES:
        rename[f"auc_{m}"] = f"AUC - {STRATEGY_VI[m]}"
    save_csv(df.rename(columns=rename), out / "walk_forward_chi_tiet.csv")

    dist = {"Số khớp": [f"{m}/6" for m in range(PICK + 1)]}
    for s in STRATEGIES:
        counts = result["stats"][s]["counts"]
        dist[f"Số kỳ - {STRATEGY_VI[s]}"] = counts
        dist[f"Tỷ lệ - {STRATEGY_VI[s]}"] = [c / max(sum(counts), 1) for c in counts]
    mc_total = max(int(result["mc_counts"].sum()), 1)
    dist["Tỷ lệ - Ngẫu nhiên mô phỏng"] = [c / mc_total for c in result["mc_counts"]]
    dist["Tỷ lệ - Ngẫu nhiên lý thuyết"] = HYPER_PMF.tolist()
    save_csv(pd.DataFrame(dist), out / "walk_forward_phan_bo_khop.csv")
    save_json(out / "walk_forward_tong_ket.json", result["summary"])


# ----------------------------------------------------------------------------
# 7) Dự đoán kỳ tiếp theo
# ----------------------------------------------------------------------------
def run_prediction(data: DrawData, cfg: Config, out: Path) -> dict:
    n = data.n
    labels = build_labels(data.draws)
    log("ML", "Đang huấn luyện mô hình trên toàn bộ dữ liệu lịch sử")
    models = None
    probs: Dict[str, np.ndarray] = {}
    f_next = None
    tracker_final = None
    for t, tracker, feats in iter_history(data):
        if t == n:
            lo = max(cfg.warmup, n - cfg.train_window)
            models = fit_models(feats, labels, lo, n, cfg)
            f_next = feats[n]
            probs = predict_probabilities(models, f_next)
            tracker_final = tracker
    assert models is not None and f_next is not None and tracker_final is not None

    signal_bundle = build_strategy_signal_bundle(probs, f_next)
    tickets = {
        "random_forest": top6(probs["random_forest"]),
        "logistic_regression": top6(probs["logistic_regression"]),
        "frequency_100": top6(f_next[:, FI["freq_w100"]]),
        "overdue": top6(f_next[:, FI["delay"]]),
        "balanced_ensemble": top6(signal_bundle["balanced_ensemble"]),
        "ml_focus": top6(signal_bundle["ml_focus"]),
        "statistical_focus": top6(signal_bundle["statistical_focus"]),
        "diversified": top6(signal_bundle["diversified"]),
        "alternative": top6(signal_bundle["alternative"]),
    }
    ranking = pd.DataFrame(
        {
            "number": np.arange(1, MAX_NUM + 1),
            "prob_random_forest": probs["random_forest"],
            "prob_logistic_regression": probs["logistic_regression"],
            "freq_100": f_next[:, FI["freq_w100"]],
            "delay": f_next[:, FI["delay"]],
        }
    ).sort_values("prob_random_forest", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, MAX_NUM + 1))
    log("DỰ ĐOÁN", "Top 10 số theo Random Forest: " + ", ".join(f"{int(x):02d}" for x in ranking["number"].head(10)))
    for name, tk in tickets.items():
        log("DỰ ĐOÁN", f"Vé 6 số - {STRATEGY_VI[name]}: {fmt_ticket(tk)}")

    engine_info = None
    top_combos: List[dict] = []
    if not cfg.skip_engine:
        ctx = build_score_context(tracker_final, probs["random_forest"], cfg, "next")
        res = run_combination_engine([ctx], cfg)[0]
        top_combos = [
            {"rank": i + 1, "numbers": [int(x) for x in c], "score": float(s)}
            for i, (c, s) in enumerate(zip(res["combos"], res["scores"]))
        ]
        engine_info = {
            "tong_khong_gian": TOTAL_COMBOS,
            "so_to_hop_da_quet": res["scanned"],
            "kich_thuoc_lo": cfg.batch_size,
            "top_k": cfg.top_k,
            "thoi_gian_giay": res["seconds"],
            "trong_so": {"w_number": cfg.w_number, "w_pair": cfg.w_pair, "w_triple": cfg.w_triple, "w_struct": cfg.w_struct},
            "ghi_chu": "Điểm tổ hợp = xác suất từng số (Random Forest) + lift cặp/bộ ba + cấu trúc lịch sử; chỉ để xếp hạng tham khảo.",
        }
        log("DỰ ĐOÁN", f"Top {min(5, len(top_combos))} tổ hợp: " + " | ".join(fmt_ticket(c["numbers"]) for c in top_combos[:5]))
        save_csv(
            pd.DataFrame(
                [
                    {"Hạng": c["rank"], **{f"Số {i + 1}": c["numbers"][i] for i in range(PICK)}, "Điểm": c["score"]}
                    for c in top_combos
                ]
            ),
            out / "du_doan_top_to_hop.csv",
        )
    save_csv(
        ranking.rename(
            columns={
                "rank": "Hạng", "number": "Số", "prob_random_forest": "Xác suất Random Forest",
                "prob_logistic_regression": "Xác suất Hồi quy Logistic", "freq_100": "Tần suất 100 kỳ", "delay": "Số kỳ chưa ra (delay)",
            }
        ),
        out / "du_doan_xep_hang_so.csv",
    )
    last_date = data.dates.iloc[-1] if data.dates is not None else None
    payload = {
        "thoi_diem_tao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "canh_bao": WARNING_VI,
        "based_on": {
            "draw_count": n,
            "last_draw_date": last_date.strftime("%Y-%m-%d") if last_date is not None else None,
            "last_draw_numbers": [int(x) for x in data.draws[-1]],
        },
        "ve_de_xuat_6_so": {STRATEGY_VI[k]: [int(x) for x in v] for k, v in tickets.items()},
        "suggested_tickets": {k: [int(x) for x in v] for k, v in tickets.items()},
        "xep_hang_so": ranking.to_dict(orient="records"),
        "top_combinations": top_combos,
        "dong_co_to_hop": engine_info,
        "ghi_chu": "Kết quả chỉ mang tính tham khảo thống kê, không đảm bảo trúng thưởng.",
    }
    save_json(out / "du_doan_ky_tiep.json", payload)
    if not cfg.no_save_model:
        model_path = out / "mo_hinh_ml.joblib"
        joblib.dump({"models": models, "feature_names": FEATURE_NAMES, "trained_on_draws": n}, model_path)
        log("LƯU", f"Đã lưu mô hình (joblib): {model_path}")
    return payload


# ----------------------------------------------------------------------------
# 8) Đánh giá động cơ tổ hợp trên các kỳ gần nhất (xếp hạng chính xác trên 28.989.675 tổ hợp)
# ----------------------------------------------------------------------------
def run_combos_eval(data: DrawData, cfg: Config, out: Path) -> dict:
    n = data.n
    m_draws = max(1, cfg.combos_eval_draws)
    first = n - m_draws
    if first < cfg.warmup + 30:
        raise DataError("Không đủ dữ liệu lịch sử cho --combos-eval-draws đã chọn.")
    labels = build_labels(data.draws)
    contexts: List[ScoreContext] = []
    log("COMBO-EVAL", f"Đang chuẩn bị {m_draws} kỳ gần nhất (mỗi kỳ chỉ dùng dữ liệu trước nó)")
    models = None
    for t, tracker, feats in iter_history(data):
        if t < first or t >= n:
            continue
        lo = max(cfg.warmup, t - cfg.train_window)
        log("ML", f"Đang huấn luyện mô hình cho kỳ #{t + 1:,}")
        models = fit_models(feats, labels, lo, t, cfg)
        probs = predict_probabilities(models, feats[t])
        contexts.append(build_score_context(tracker, probs["random_forest"], cfg, f"ky_{t + 1}", data.draws[t].astype(np.int8)))
    results = run_combination_engine(contexts, cfg)
    rows = []
    agg = np.zeros(PICK + 1, dtype=np.int64)
    for i, res in enumerate(results):
        t = first + i
        actual = data.draws[t]
        mask = actual_mask(actual)
        counts = np.bincount(mask[res["combos"].astype(np.int64)].sum(axis=1), minlength=PICK + 1)
        agg += counts
        k = res["combos"].shape[0]
        row = {
            "draw_index": t + 1,
            "date": data.dates.iloc[t] if data.dates is not None else None,
            "actual": fmt_ticket(actual),
            "top_k": k,
            "best_match": int(np.max(np.nonzero(counts)[0])),
            "actual_rank": res.get("actual_rank"),
            "actual_percentile": res.get("actual_percentile"),
        }
        for mm in range(PICK + 1):
            row[f"tickets_{mm}"] = int(counts[mm])
            row[f"expected_random_{mm}"] = float(k * HYPER_PMF[mm])
        rows.append(row)
        log(
            "COMBO-EVAL",
            f"Kỳ #{t + 1:,} ({fmt_ticket(actual)}): hạng thực tế {res['actual_rank']:,.0f}/{res['scanned']:,} "
            f"(phân vị {res['actual_percentile']:.4f}; ngẫu nhiên ≈ 0.5) | khớp tốt nhất trong Top-{k}: {row['best_match']}/6",
        )
    df = pd.DataFrame(rows)
    rename = {
        "draw_index": "Kỳ thứ", "date": "Ngày quay", "actual": "Kết quả thực tế", "top_k": "Số tổ hợp Top-K",
        "best_match": "Khớp tốt nhất", "actual_rank": "Hạng của kết quả thật", "actual_percentile": "Phân vị (thấp = tốt)",
    }
    for mm in range(PICK + 1):
        rename[f"tickets_{mm}"] = f"Số vé khớp {mm}/6"
        rename[f"expected_random_{mm}"] = f"Kỳ vọng ngẫu nhiên {mm}/6"
    save_csv(df.rename(columns=rename), out / "danh_gia_to_hop.csv")
    total_tickets = int(agg.sum())
    payload = {
        "thoi_diem": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "canh_bao": WARNING_VI,
        "so_ky_danh_gia": m_draws,
        "top_k_moi_ky": cfg.top_k,
        "phan_bo_khop_top_k": [
            {
                "so_khop": f"{mm}/6",
                "so_ve": int(agg[mm]),
                "ty_le": float(agg[mm] / total_tickets),
                "ty_le_ngau_nhien_ly_thuyet": float(HYPER_PMF[mm]),
                "ky_vong_ngau_nhien": float(total_tickets * HYPER_PMF[mm]),
            }
            for mm in range(PICK + 1)
        ],
        "chi_tiet_theo_ky": json.loads(json.dumps(rows, default=_to_builtin)),
        "ghi_chu": (
            "So sánh Top-K vé của động cơ với K vé ngẫu nhiên (phân phối siêu bội). "
            "Với ít kỳ, chênh lệch gần như chắc chắn chỉ là nhiễu ngẫu nhiên."
        ),
    }
    save_json(out / "danh_gia_to_hop.json", payload)
    log("KẾT QUẢ", f"Phân bố số khớp của {total_tickets:,} vé Top-K (so với ngẫu nhiên):")
    for mm in range(PICK + 1):
        log("KẾT QUẢ", f"  {mm}/6: {agg[mm]:>8,} vé ({agg[mm] / total_tickets:.4%}) | ngẫu nhiên kỳ vọng {total_tickets * HYPER_PMF[mm]:,.1f} ({HYPER_PMF[mm]:.4%})")
    return payload


# ----------------------------------------------------------------------------
# 9) Đánh giá dự đoán đã lưu với kết quả thật
# ----------------------------------------------------------------------------
def run_evaluate_prediction(data: DrawData, cfg: Config, out: Path) -> dict:
    if not cfg.prediction_file:
        raise DataError("Thiếu tham số --prediction-file (đường dẫn tới du_doan_ky_tiep.json).")
    path = Path(cfg.prediction_file)
    if not path.is_file():
        raise DataError(f"Không tìm thấy file dự đoán: {path}")
    try:
        pred = json.loads(path.read_text(encoding="utf-8"))
        base_n = int(pred["based_on"]["draw_count"])
    except (ValueError, KeyError, TypeError) as exc:
        raise DataError("File dự đoán không đúng định dạng do ml_model.py tạo ra.") from exc

    if cfg.actual:
        nums = [int(x) for x in re.findall(r"\d+", cfg.actual)][:PICK]
        if len(nums) != PICK or len(set(nums)) != PICK or not all(1 <= x <= MAX_NUM for x in nums):
            raise DataError("--actual phải gồm đúng 6 số khác nhau trong khoảng 1..55, ví dụ: \"05 12 23 34 41 52\".")
        actual, actual_date = np.sort(nums), None
    else:
        if data.n <= base_n:
            raise DataError(
                f"Chưa có kỳ quay mới sau dự đoán (dự đoán dựa trên {base_n:,} kỳ, CSV hiện có {data.n:,} kỳ). "
                "Hãy cập nhật CSV hoặc dùng --actual."
            )
        actual = data.draws[base_n]
        actual_date = data.dates.iloc[base_n] if data.dates is not None else None
        saved_last = pred["based_on"].get("last_draw_date")
        if saved_last and data.dates is not None and data.dates.iloc[base_n - 1].strftime("%Y-%m-%d") != saved_last:
            log("CẢNH BÁO", "Ngày của kỳ cuối lúc dự đoán khác với CSV hiện tại: dữ liệu có thể đã bị sắp xếp/chỉnh sửa.")
    mask = actual_mask(actual)
    log("EVALUATE", f"Kết quả thực tế: {fmt_ticket(actual)}" + (f" (ngày {actual_date:%d/%m/%Y})" if actual_date is not None else ""))

    rows = []
    for name, ticket in pred.get("suggested_tickets", {}).items():
        m = int(mask[np.asarray(ticket)].sum())
        rows.append({"loai": f"ve_{name}", "ten": STRATEGY_VI.get(name, name), "so": fmt_ticket(ticket), "so_khop": m})
        log("EVALUATE", f"Vé {STRATEGY_VI.get(name, name)}: {fmt_ticket(ticket)} -> khớp {m}/6")
    combos = pred.get("top_combinations", [])
    counts = np.zeros(PICK + 1, dtype=np.int64)
    for c in combos:
        counts[int(mask[np.asarray(c["numbers"])].sum())] += 1
    k = int(counts.sum())
    if k:
        log("EVALUATE", f"Top-{k} tổ hợp: " + ", ".join(f"{m}/6={counts[m]}" for m in range(PICK + 1)))
    payload = {
        "thoi_diem": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "canh_bao": WARNING_VI,
        "ket_qua_thuc_te": [int(x) for x in actual],
        "ngay_quay": actual_date,
        "ve_de_xuat": rows,
        "top_to_hop": [
            {
                "so_khop": f"{m}/6",
                "so_ve": int(counts[m]),
                "ky_vong_ngau_nhien": float(k * HYPER_PMF[m]),
            }
            for m in range(PICK + 1)
        ],
        "ghi_chu": "So sánh với ngẫu nhiên: kỳ vọng số khớp trung bình của một vé ngẫu nhiên là %.3f/6." % HYPER_MEAN,
    }
    save_json(out / "danh_gia_du_doan.json", payload)
    history_path = out / "lich_su_danh_gia_du_doan.csv"
    new_rows = pd.DataFrame(
        [
            {
                "Thời điểm": payload["thoi_diem"], "Kết quả thực tế": fmt_ticket(actual),
                "Loại": r["loai"], "Vé": r["so"], "Số khớp": f"{r['so_khop']}/6",
            }
            for r in rows
        ]
        + [{"Thời điểm": payload["thoi_diem"], "Kết quả thực tế": fmt_ticket(actual), "Loại": "top_to_hop", "Vé": f"Top-{k}", "Số khớp": ", ".join(f"{m}/6={counts[m]}" for m in range(PICK + 1))}]
        if k
        else []
    )
    if history_path.exists() and not new_rows.empty:
        new_rows = pd.concat([pd.read_csv(history_path, encoding="utf-8-sig", dtype=str), new_rows], ignore_index=True)
    if not new_rows.empty:
        save_csv(new_rows, history_path)
    return payload


# ----------------------------------------------------------------------------
# 10) Dòng lệnh
# ----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description="Phân tích thử nghiệm Vietlott 6/55 bằng thống kê + Machine Learning (KHÔNG đảm bảo dự đoán đúng).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", nargs="*", metavar="FILE", help="Các file CSV lịch sử (liệt kê từ cũ đến mới nếu không có cột ngày). Nếu bỏ trống, script sẽ tự dò CSV trong thư mục dự án.")
    p.add_argument("--game", choices=["power655", "mega645", "lotto535"], default="power655",
                   help="Loại game để dùng cùng engine thống kê chung. Mặc định là Power 6/55.")
    p.add_argument("--mode", choices=["validate", "walkforward", "predict", "combos-eval", "evaluate", "full"], default="full",
                   help="validate: chỉ kiểm tra dữ liệu | walkforward: đánh giá theo thời gian | predict: dự đoán kỳ tiếp | "
                        "combos-eval: xếp hạng chính xác trên toàn bộ 28.989.675 tổ hợp cho các kỳ gần nhất | "
                        "evaluate: so dự đoán đã lưu với kết quả thật | full: walkforward + predict")
    p.add_argument("--output-dir", default="output", help="Thư mục lưu kết quả")
    p.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên (tái lập kết quả)")
    p.add_argument("--warmup", type=int, default=50, help="Số kỳ đầu chỉ để tích lũy thống kê, không dùng làm dữ liệu huấn luyện")
    p.add_argument("--eval-steps", type=int, default=200, help="Số kỳ gần nhất được đánh giá walk-forward")
    p.add_argument("--retrain-every", type=int, default=10, help="Huấn luyện lại mô hình sau mỗi N kỳ khi walk-forward")
    p.add_argument("--train-window", type=int, default=600, help="Số kỳ gần nhất dùng để huấn luyện (giới hạn RAM/thời gian)")
    p.add_argument("--n-estimators", type=int, default=150, help="Số cây Random Forest")
    p.add_argument("--max-depth", type=int, default=10, help="Độ sâu tối đa mỗi cây")
    p.add_argument("--min-samples-leaf", type=int, default=40, help="Số mẫu tối thiểu ở lá (chống overfit)")
    p.add_argument("--n-jobs", type=int, default=-1, help="Số luồng CPU (-1 = tất cả)")
    p.add_argument("--top-k", type=int, default=100, help="Số tổ hợp tốt nhất được giữ lại")
    p.add_argument("--batch-size", type=int, default=500_000, help="Số tổ hợp mỗi lô (giảm nếu thiếu RAM)")
    p.add_argument("--combo-limit", type=int, default=0, help="CHỈ ĐỂ THỬ: giới hạn số tổ hợp quét (0 = quét toàn bộ)")
    p.add_argument("--combo-sample", type=int, default=20_000, help="Số tổ hợp ngẫu nhiên mỗi kỳ để ước lượng phân vị trong walk-forward (0 = tắt)")
    p.add_argument("--mc-tickets", type=int, default=200, help="Số vé ngẫu nhiên mô phỏng mỗi kỳ làm baseline")
    p.add_argument("--combos-eval-draws", type=int, default=3, help="Số kỳ gần nhất cho chế độ combos-eval")
    p.add_argument("--w-number", type=float, default=1.0, help="Trọng số xác suất từng số (ML)")
    p.add_argument("--w-pair", type=float, default=0.5, help="Trọng số cặp số")
    p.add_argument("--w-triple", type=float, default=0.25, help="Trọng số bộ ba (0 = tắt, nhanh hơn)")
    p.add_argument("--w-struct", type=float, default=0.5, help="Trọng số cấu trúc (lẻ/chẵn, tổng, khoảng cách...)")
    p.add_argument("--skip-engine", action="store_true", help="Bỏ qua quét 28.989.675 tổ hợp khi predict")
    p.add_argument("--prediction-file", default="", help="File du_doan_ky_tiep.json để đánh giá (mode evaluate)")
    p.add_argument("--actual", default="", help='Kết quả thật nhập tay, ví dụ "05 12 23 34 41 52" (mode evaluate)')
    p.add_argument("--no-save-model", action="store_true", help="Không lưu mô hình joblib")
    p.add_argument("--debug", action="store_true", help="In chi tiết lỗi (traceback)")
    ns = p.parse_args(argv)
    resolved_csv = resolve_project_csv_files(ns.csv)
    cfg = Config(csv_files=resolved_csv, **{k: v for k, v in vars(ns).items() if k != "csv"})
    cfg.game = resolve_game_spec(cfg.game, cfg.csv_files).name
    if cfg.batch_size < 1000:
        raise DataError("--batch-size phải từ 1.000 trở lên.")
    if cfg.top_k < 1 or cfg.retrain_every < 1 or cfg.eval_steps < 1 or cfg.warmup < 2:
        raise DataError("Tham số không hợp lệ: --top-k, --retrain-every, --eval-steps phải ≥ 1 và --warmup ≥ 2.")
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
        game_spec = resolve_game_spec(cfg.game, cfg.csv_files)
        apply_game_spec(game_spec)
        log("HỆ THỐNG", f"Công cụ thống kê/ML chung cho {game_spec.short_name or game_spec.name} - " + WARNING_VI)
        data = load_and_validate(cfg.csv_files, game_spec)
        stats_summary = print_statistics_summary(data)
        reference_stats = compute_reference_stats(data)
        save_json(out / "bao_cao_du_lieu.json", {"canh_bao": WARNING_VI, "kiem_tra_du_lieu": data.report, "thong_ke": stats_summary, "reference_stats": reference_stats})
        walk_result = None
        pred_payload = None
        if cfg.mode == "validate":
            log("RESULT", "Đã hoàn thành kiểm tra dữ liệu")
            return 0
        if cfg.mode in ("walkforward", "full"):
            walk_result = run_walk_forward(data, cfg)
            print_walk_forward_report(walk_result)
            save_walk_forward(walk_result, out)
        if cfg.mode in ("predict", "full"):
            pred_payload = run_prediction(data, cfg, out)
        if cfg.mode == "combos-eval":
            run_combos_eval(data, cfg, out)
        if cfg.mode == "evaluate":
            run_evaluate_prediction(data, cfg, out)
        build_game_outputs(out, walk_result, pred_payload, stats_summary, reference_stats, game_spec, PROJECT_ROOT)
        generate_combined_report(out, PROJECT_ROOT)
        log("RESULT", f"Đã hoàn thành phân tích trong {time.time() - t0:.1f}s. Kết quả nằm trong thư mục: {out.resolve()}")
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
        log("LỖI", "Hết bộ nhớ RAM. Hãy giảm --batch-size (ví dụ 200000) hoặc --train-window.")
        return 3
    except Exception as exc:  # noqa: BLE001
        log("LỖI", f"Lỗi không mong đợi: {exc}. Chạy lại với --debug để xem chi tiết.")
        if debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())