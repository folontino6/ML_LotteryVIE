#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phan tich thong ke Vietlott Power 6/55 (khong sua CSV goc).

Cau truc: (1) nap + kiem tra du lieu, (2) dac trung, (3) mo hinh / kiem dinh,
(4) walk-forward, (5) ensemble + to hop ung vien, (6) bao cao.
Code/identifier: tieng Anh. Log, bao cao: tieng Viet.
"""
import argparse
import itertools
import json
import math
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_NUM, K = 55, 6
P_NUM = K / N_NUM
P_PAIR = 30 / (55 * 54)
SEED = 20261005
MAIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
TOTAL_COMBOS = comb(55, 6)
HALF_LIVES = {"ew10": 10, "ew30": 30, "ew100": 100, "ew300": 300}
WINDOWS = [20, 50, 100, 200]


# ------------------------------------------------------------------ 1. DATA
def find_csv(arg):
    if arg:
        return Path(arg)
    for base in [Path.cwd(), Path("/mnt/user-data/uploads")]:
        hits = sorted(base.rglob("power655*.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError("Khong tim thay file CSV Power 6/55")


def load_and_validate(path):
    raw = pd.read_csv(path, dtype={"draw_number": str})
    issues = []
    info = {"columns": list(raw.columns), "rows_raw": int(len(raw))}
    for c in MAIN_COLS + ["special_number"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["draw_no"] = pd.to_numeric(raw["draw_number"], errors="coerce")
    raw["date"] = pd.to_datetime(raw["draw_datetime"], errors="coerce")

    malformed = raw[MAIN_COLS + ["special_number", "draw_no"]].isna().any(axis=1) | raw["date"].isna()
    in_range = ((raw[MAIN_COLS] >= 1) & (raw[MAIN_COLS] <= 55)).all(axis=1)
    unique6 = raw[MAIN_COLS].nunique(axis=1) == 6
    dup_no = raw["draw_no"].duplicated(keep="first")
    valid = ~malformed & in_range & unique6 & ~dup_no
    if malformed.sum():
        issues.append(f"{int(malformed.sum())} dong thieu/sai dinh dang")
    if (~in_range & ~malformed).sum():
        issues.append(f"{int((~in_range & ~malformed).sum())} dong co so ngoai 1-55")
    if (~unique6 & ~malformed).sum():
        issues.append(f"{int((~unique6 & ~malformed).sum())} dong khong du 6 so khac nhau")
    if dup_no.sum():
        issues.append(f"{int(dup_no.sum())} so ky quay bi trung")

    df = raw[valid].sort_values("draw_no").reset_index(drop=True)
    nums = np.sort(df[MAIN_COLS].to_numpy(dtype=int), axis=1)
    info["not_sorted_rows"] = int((raw[MAIN_COLS].diff(axis=1).iloc[:, 1:] <= 0).any(axis=1).sum())

    dn = df["draw_no"].astype(int)
    missing = sorted(set(range(dn.min(), dn.max() + 1)) - set(dn))
    if missing:
        issues.append(f"Thieu {len(missing)} so ky: {missing[:20]}")
    info["draw_no_min"], info["draw_no_max"] = int(dn.min()), int(dn.max())
    info["contiguous"] = len(missing) == 0
    info["date_min"], info["date_max"] = str(df["date"].min().date()), str(df["date"].max().date())
    if (df["date"].diff().dt.days < 0).any():
        issues.append("Ngay quay khong tang dan theo so ky")
    gaps = df["date"].diff().dt.days.dropna()
    info["date_gap_days_counts"] = {int(k): int(v) for k, v in gaps.value_counts().sort_index().items()}
    big = df.loc[gaps[gaps > 7].index, ["draw_number", "draw_datetime"]]
    if len(big):
        issues.append(f"{len(big)} khoang nghi > 7 ngay giua cac ky")
    info["game_type_values"] = sorted(map(str, raw["game_type"].dropna().unique()))

    keys = [tuple(r) for r in nums]
    dup_combo = len(keys) - len(set(keys))
    info["duplicate_combinations"] = dup_combo
    if dup_combo:
        issues.append(f"{dup_combo} to hop 6 so lap lai (thong tin, co the xay ra ngau nhien)")

    sp = df["special_number"].to_numpy(dtype=int)
    info["special_out_of_range"] = int(((sp < 1) | (sp > 55)).sum())
    info["special_in_main"] = int(sum(sp[i] in nums[i] for i in range(len(df))))
    if info["special_out_of_range"]:
        issues.append(f"{info['special_out_of_range']} so dac biet ngoai 1-55")
    if info["special_in_main"]:
        issues.append(f"{info['special_in_main']} ky co so dac biet trung so chinh")
    info["n_valid"] = int(len(df))
    info["issues"] = issues
    return df, nums, sp, info


def to_matrix(nums):
    T = len(nums)
    X = np.zeros((T, N_NUM))
    X[np.arange(T)[:, None], nums - 1] = 1
    return X


# ------------------------------------------------------------ 2. UTILITIES
def bh_adjust(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        run = min(run, p[i] * n / rank)
        adj[i] = run
    return adj


def binom_two_sided(k, n, p):
    return np.minimum(1.0, 2 * np.minimum(stats.binom.cdf(k, n, p), stats.binom.sf(k - 1, n, p)))


def sim_draws(rng, T):
    r = rng.random((T, N_NUM))
    return np.sort(np.argpartition(r, K, axis=1)[:, :K], axis=1)


TRIPLE_POS = list(itertools.combinations(range(6), 3))


def calc_stats(idx):
    """Thong ke tong hop (dung chung cho du lieu that va mo phong)."""
    T = idx.shape[0]
    X = np.zeros((T, N_NUM))
    X[np.arange(T)[:, None], idx] = 1
    cnt = X.sum(0)
    E, sd = T * P_NUM, math.sqrt(T * P_NUM * (1 - P_NUM))
    out = {"freq_chi": float(((cnt - E) ** 2).sum() / sd ** 2), "freq_maxz": float(np.abs(cnt - E).max() / sd)}
    C = X.T @ X
    pc = C[np.triu_indices(N_NUM, 1)]
    Ep, sdp = T * P_PAIR, math.sqrt(T * P_PAIR * (1 - P_PAIR))
    z = (pc - Ep) / sdp
    out["pair_chi"], out["pair_maxz"] = float((z ** 2).sum()), float(np.abs(z).max())
    mx = 0
    codes = np.concatenate([idx[:, a] * 3025 + idx[:, b] * 55 + idx[:, c] for a, b, c in TRIPLE_POS])
    mx = int(np.bincount(codes).max())
    out["triple_max"] = mx
    prev, cur = X[:-1], X[1:]
    out["overlap_mean"] = float((prev * cur).sum(1).mean())
    n1, n0 = prev.sum(0), (1 - prev).sum(0)
    n11, n01 = (prev * cur).sum(0), ((1 - prev) * cur).sum(0)
    pool = (n11 + n01) / (n1 + n0)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n0))
    zz = (n11 / n1 - n01 / n0) / se
    out["markov_chi"] = float((zz ** 2).sum())
    s = idx.sum(1) + 6.0
    s = s - s.mean()
    out["sum_acf1"] = float((s[1:] * s[:-1]).sum() / (s ** 2).sum())
    return out


def mc_window_chi(rng, w):
    c = np.bincount(sim_draws(rng, w).ravel(), minlength=N_NUM)
    return ((c - w * P_NUM) ** 2).sum() / (w * P_NUM * (1 - P_NUM))


def draw_metrics(nums):
    d = {}
    d["sum"] = nums.sum(1)
    d["odd"] = (nums % 2 == 1).sum(1)
    d["low"] = (nums <= 27).sum(1)
    d["range"] = nums[:, 5] - nums[:, 0]
    d["std"] = nums.std(1)
    d["consec"] = (np.diff(nums, axis=1) == 1).sum(1)
    dec = np.sort(nums // 10, axis=1)
    d["ndecades"] = (np.diff(dec, axis=1) > 0).sum(1) + 1
    return d


def cat_test(obs_vals, exp_probs, maxcat):
    n = len(obs_vals)
    o = np.bincount(np.clip(obs_vals, 0, maxcat), minlength=maxcat + 1).astype(float)
    e = np.asarray(exp_probs, float) * n
    labels = [str(i) for i in range(maxcat + 1)]
    o, e = list(o), list(e)
    while len(e) > 2 and e[0] < 5:
        o[1] += o[0]; e[1] += e[0]; o.pop(0); e.pop(0); labels.pop(0)
    while len(e) > 2 and e[-1] < 5:
        o[-2] += o[-1]; e[-2] += e[-1]; o.pop(); e.pop(); labels.pop()
    o, e = np.array(o), np.array(e)
    chi = float(((o - e) ** 2 / e).sum())
    df = len(e) - 1
    return {"chi2": chi, "df": df, "p": float(stats.chi2.sf(chi, df)), "labels": labels,
            "observed": o.tolist(), "expected": [round(x, 1) for x in e]}


def hyper_overlap_probs():
    return np.array([comb(6, k) * comb(49, 6 - k) / TOTAL_COMBOS for k in range(7)])


# --------------------------------------------------------- 3. FEATURE STATE
class Tracker:
    """Trang thai tang dan: chi dung cac ky < t (khong ro ri du lieu)."""

    def __init__(self, X, shrink=50.0):
        self.X = X
        self.T = X.shape[0]
        self.cs = np.vstack([np.zeros((1, N_NUM)), np.cumsum(X, 0)])
        self.t = 0
        self.lam = {k: 0.5 ** (1.0 / h) for k, h in HALF_LIVES.items()}
        self.num = {k: np.zeros(N_NUM) for k in HALF_LIVES}
        self.den = {k: 0.0 for k in HALF_LIVES}
        self.last = -np.ones(N_NUM, dtype=int)
        self.gaps = [[] for _ in range(N_NUM)]
        self.a1 = np.zeros(N_NUM); self.a11 = np.zeros(N_NUM)
        self.a0 = np.zeros(N_NUM); self.a01 = np.zeros(N_NUM)
        self.C = np.zeros((N_NUM, N_NUM))
        self.shrink = shrink

    def components(self):
        t = self.t
        c = {"f_all": self.cs[t] / t}
        for w in WINDOWS:
            lo = max(0, t - w)
            c[f"f_{w}"] = (self.cs[t] - self.cs[lo]) / (t - lo)
        for k in HALF_LIVES:
            c[k] = self.num[k] / self.den[k]
        gap_now = np.where(self.last >= 0, t - 1 - self.last, t)
        pct = np.empty(N_NUM)
        for i in range(N_NUM):
            g = self.gaps[i]
            pct[i] = (np.searchsorted(np.sort(g), gap_now[i], side="right") / len(g)) if len(g) >= 3 else 0.5
        c["gap"] = pct
        c["gap_raw"] = gap_now.astype(float)
        prev = self.X[t - 1]
        k = self.shrink
        p1 = (self.a11 + k * P_NUM) / (self.a1 + k)
        p0 = (self.a01 + k * P_NUM) / (self.a0 + k)
        c["markov"] = np.where(prev == 1, p1, p0)
        E = t * P_PAIR
        dev = (self.C - E) / math.sqrt(E)
        np.fill_diagonal(dev, 0.0)
        c["pair"] = dev[:, prev == 1].sum(1)
        return c

    def update(self):
        t = self.t
        x = self.X[t]
        if t >= 1:
            prev = self.X[t - 1]
            self.a1 += prev; self.a11 += prev * x
            self.a0 += 1 - prev; self.a01 += (1 - prev) * x
        for kname in HALF_LIVES:
            self.num[kname] = self.lam[kname] * self.num[kname] + x
            self.den[kname] = self.lam[kname] * self.den[kname] + 1
        for i in np.where(x == 1)[0]:
            if self.last[i] >= 0:
                self.gaps[i].append(t - self.last[i] - 1)
            self.last[i] = t
        self.C += np.outer(x, x)
        self.t += 1


def zrow(M):
    M = np.atleast_2d(M)
    sd = M.std(1, keepdims=True)
    sd[sd == 0] = 1
    return (M - M.mean(1, keepdims=True)) / sd


ENS_COMPONENTS = ["f_all", "f_100", "f_20", "ew30", "gap", "markov", "pair"]
PROB_MODELS = ["f_all", "f_200", "ew30", "markov"]


# ------------------------------------------------------------ 4. BACKTEST
HYP = hyper_overlap_probs()
MU_MATCH = K * K / N_NUM
VAR_MATCH = K * (K / N_NUM) * (1 - K / N_NUM) * (N_NUM - K) / (N_NUM - 1)


def rank_pct(S):
    return (stats.rankdata(S, axis=1) - 1) / (N_NUM - 1)


def evaluate(S, Xs, rng, boot=2000):
    n = len(S)
    top = np.argsort(-(S + 1e-9 * rng.random(S.shape)), axis=1)[:, :K]
    m = Xs[np.arange(n)[:, None], top].sum(1)
    z = (m.mean() - MU_MATCH) / math.sqrt(VAR_MATCH / n)
    obs = np.array([(m == 0).sum(), (m == 1).sum(), (m == 2).sum(), (m >= 3).sum()], float)
    pe = np.array([HYP[0], HYP[1], HYP[2], HYP[3:].sum()]) * n
    chi = float(((obs - pe) ** 2 / pe).sum())
    bs = np.array([rng.choice(m, n).mean() for _ in range(boot)])
    rp = rank_pct(S)[Xs == 1].reshape(n, K).mean(1)
    tt = stats.ttest_1samp(rp, 0.5)
    return {"n": n, "mean_matches": float(m.mean()), "expected": MU_MATCH,
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "z": float(z), "p_matches": float(2 * stats.norm.sf(abs(z))),
            "dist_obs": (obs / n).round(4).tolist(), "dist_exp": (pe / n).round(4).tolist(),
            "chi2_dist_p": float(stats.chi2.sf(chi, 3)),
            "mean_rank_pct": float(rp.mean()), "rank_t_p": float(tt.pvalue),
            "rank_effect": float(rp.mean() - 0.5), "hit3plus": float((m >= 3).mean())}


def brier_eval(P, Xs):
    P = np.clip(P * (K / P.sum(1, keepdims=True)), 0.005, 0.995)
    b = ((P - Xs) ** 2).mean(1)
    b0 = ((P_NUM - Xs) ** 2).mean(1)
    d = b0 - b
    tt = stats.ttest_1samp(d, 0.0)
    bins = np.quantile(P, [0, .2, .4, .6, .8, 1.0])
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (P >= lo) & (P <= hi)
        cal.append([round(float(P[mk].mean()), 4), round(float(Xs[mk].mean()), 4), int(mk.sum())])
    return {"brier": float(b.mean()), "brier_base": float(b0.mean()),
            "bss": float(1 - b.mean() / b0.mean()), "p": float(tt.pvalue), "calibration": cal}


def walk_forward(X, t0, rng):
    T = X.shape[0]
    tr = Tracker(X)
    comps = {}
    for t in range(T):
        if t >= t0:
            for k, v in tr.components().items():
                comps.setdefault(k, []).append(v)
        tr.update()
    comps = {k: np.array(v) for k, v in comps.items()}
    Xs = X[t0:]
    S = len(Xs)
    models = {k: comps[k] for k in ["f_all", "f_200", "f_100", "f_50", "f_20",
                                    "ew10", "ew30", "ew100", "ew300", "gap", "markov", "pair"]}
    Z = {k: zrow(comps[k]) for k in ENS_COMPONENTS}
    models["ens_equal"] = np.mean([Z[k] for k in ENS_COMPONENTS], axis=0)
    res = {name: evaluate(M, Xs, rng) for name, M in models.items()}

    # trong so hoc tu nua dau, kiem tra tren nua sau (co shrink 50% ve trong so deu)
    h = S // 2
    edge = np.array([rank_pct(Z[k][:h])[Xs[:h] == 1].reshape(h, K).mean() - 0.5 for k in ENS_COMPONENTS])
    w = np.clip(edge, 0, None)
    w = np.ones(len(w)) / len(w) if w.sum() == 0 else 0.5 / len(w) + 0.5 * w / w.sum()
    ens_w = sum(wi * Z[k] for wi, k in zip(w, ENS_COMPONENTS))
    ens_eq = models["ens_equal"]
    res["_weights_check"] = {
        "weights": dict(zip(ENS_COMPONENTS, np.round(w, 3).tolist())),
        "edge_first_half": dict(zip(ENS_COMPONENTS, np.round(edge, 4).tolist())),
        "valweighted_second_half": evaluate(ens_w[h:], Xs[h:], rng),
        "equal_second_half": evaluate(ens_eq[h:], Xs[h:], rng)}

    # nen ngau nhien (mo phong) tren cung cac ky
    rm = []
    for _ in range(200):
        pick = np.argsort(rng.random((S, N_NUM)), axis=1)[:, :K]
        rm.append(Xs[np.arange(S)[:, None], pick].sum(1).mean())
    res["_random_sim"] = {"mean": float(np.mean(rm)), "sd": float(np.std(rm)),
                          "range95": [float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))]}
    prob = {}
    for k in PROB_MODELS:
        prob[k] = brier_eval(comps[k], Xs)
    res["_brier"] = prob
    res["_steps"] = S
    return res, comps


# ------------------------------------------------------------------- MAIN
def main():
    BASE_DIR = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", default=str(BASE_DIR))
    ap.add_argument("--analysis-date", default=None)
    ap.add_argument("--t0", type=int, default=200)
    ap.add_argument("--mc-reps", type=int, default=1000)
    ap.add_argument("--mc-draws", type=int, default=300000)
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    path = find_csv(a.csv)
    adate = a.analysis_date or str(pd.Timestamp.today().date())

    print(f"[Du lieu] Duong dan CSV: {path}")
    df, nums, special, info = load_and_validate(path)
    T = len(df)
    X = to_matrix(nums)
    print(f"[Du lieu] So ky hop le: {T} | Ngay phan tich: {adate}")
    print(f"[Du lieu] Khoang ky: {info['draw_no_min']}-{info['draw_no_max']} | Ngay: {info['date_min']} -> {info['date_max']}")
    print(f"[Tham so] seed={SEED}, t0={a.t0}, mc_reps={a.mc_reps}, mc_draws={a.mc_draws}, "
          f"half_lives={HALF_LIVES}, windows={WINDOWS}, markov_shrink=50")
    R = {"meta": {"csv": str(path), "analysis_date": adate, "seed": SEED, "n_draws": T,
                  "latest_draw": str(df['draw_number'].iloc[-1]),
                  "latest_date": str(df['date'].iloc[-1].date()),
                  "latest_numbers": nums[-1].tolist(), "latest_special": int(special[-1])},
         "data": info}
    tests = []  # (ten, p)

    # ---- A. Baseline
    R["baseline"] = {"p_number": P_NUM, "p_pair": P_PAIR,
                     "p_triple": comb(52, 3) / TOTAL_COMBOS, "combos": TOTAL_COMBOS,
                     "p_combo": 1 / TOTAL_COMBOS, "expected_matches_top6": MU_MATCH,
                     "hyper_matches": {str(k): float(HYP[k]) for k in range(7)}}

    # ---- Monte Carlo cho cac thong ke tong hop
    print("[MC] Mo phong thong ke tong hop ...")
    obs = calc_stats(nums - 1)
    sims = {k: [] for k in obs}
    for _ in range(a.mc_reps):
        s = calc_stats(sim_draws(rng, T))
        for k, v in s.items():
            sims[k].append(v)
    mc = {}
    for k, v in obs.items():
        arr = np.array(sims[k])
        two = k in ("overlap_mean", "sum_acf1")
        if two:
            p = (1 + (np.abs(arr - arr.mean()) >= abs(v - arr.mean())).sum()) / (1 + len(arr))
        else:
            p = (1 + (arr >= v).sum()) / (1 + len(arr))
        mc[k] = {"observed": v, "sim_mean": float(arr.mean()), "sim_sd": float(arr.std()),
                 "sim_p95": float(np.percentile(arr, 95)), "p_mc": float(p)}
        tests.append((f"MC:{k}", float(p)))
    R["mc_global"] = mc

    # ---- B. Tan suat
    cnt_all = X.sum(0)
    freq_tab = {}
    for w in [20, 50, 100, 200, T]:
        c = X[-w:].sum(0)
        pv = binom_two_sided(c, w, P_NUM)
        z = (c - w * P_NUM) / math.sqrt(w * P_NUM * (1 - P_NUM))
        chi = float((z ** 2).sum())
        if w == T:
            pchi = mc["freq_chi"]["p_mc"]
        else:
            sm = np.array([mc_window_chi(rng, w) for _ in range(2000)])
            pchi = float((1 + (sm >= chi).sum()) / 2001)
        adj = bh_adjust(pv)
        freq_tab[w] = {"count": c.tolist(), "z": z.tolist(), "p": pv.tolist(), "p_bh": adj.tolist(),
                       "chi2_stat": chi, "chi2_p_mc": pchi,
                       "n_sig_raw": int((pv < 0.05).sum()), "n_sig_bh": int((adj < 0.05).sum()),
                       "max_abs_z": float(np.abs(z).max())}
        tests.append((f"Tan suat chi2 cua so {w}", pchi))
    R["freq"] = freq_tab

    # ---- C. Khoang cach (gap)
    gap_rows = []
    tr_full = Tracker(X)
    for _ in range(T):
        tr_full.update()
    comp_next = tr_full.components()
    gap_now = comp_next["gap_raw"]
    all_gaps = []
    for i in range(N_NUM):
        g = np.array(tr_full.gaps[i])
        all_gaps.extend(g.tolist())
        pct = float((g <= gap_now[i]).mean()) if len(g) else float("nan")
        tail_geo = float((1 - P_NUM) ** gap_now[i])
        gap_rows.append({"number": i + 1, "appearances": int(cnt_all[i]), "gap_now": int(gap_now[i]),
                         "gap_mean": float(g.mean()), "gap_median": float(np.median(g)), "gap_max": int(g.max()),
                         "gap_percentile": pct, "geom_tail_p": tail_geo})
    all_gaps = np.array(all_gaps)
    geo_p = P_NUM
    ks = stats.kstest(all_gaps + 1, stats.geom(geo_p).cdf)
    R["gap"] = {"rows": gap_rows, "pooled_mean_gap": float(all_gaps.mean()),
                "theory_mean_gap": (1 - P_NUM) / P_NUM, "pooled_var": float(all_gaps.var()),
                "theory_var": (1 - P_NUM) / P_NUM ** 2,
                "n_tail_p_lt_005": int(sum(r["geom_tail_p"] < 0.05 for r in gap_rows)),
                "min_geom_tail_p": float(min(r["geom_tail_p"] for r in gap_rows)),
                "ks_geom_stat": float(ks.statistic), "ks_geom_p": float(ks.pvalue)}
    tests.append(("Phan phoi gap ~ hinh hoc (KS)", float(ks.pvalue)))
    # kiem tra "khong nho": xac suat xuat hien theo gap hien tai
    last = -np.ones(N_NUM, dtype=int)
    buckets = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 11), (12, 17), (18, 10 ** 6)]
    hit = np.zeros(len(buckets)); tot = np.zeros(len(buckets))
    gx, gy = [], []
    for t in range(T):
        ok = last >= 0
        g = t - 1 - last
        for bi, (lo, hi) in enumerate(buckets):
            mk = ok & (g >= lo) & (g <= hi)
            tot[bi] += mk.sum(); hit[bi] += X[t][mk].sum()
        gx.append(np.log1p(g[ok])); gy.append(X[t][ok])
        last[X[t] == 1] = t
    exp_h = tot * P_NUM
    chi_h = float(((hit - exp_h) ** 2 / (tot * P_NUM * (1 - P_NUM))).sum())
    pr = stats.pearsonr(np.concatenate(gx), np.concatenate(gy))
    R["gap"]["hazard"] = {"buckets": [f"{lo}-{hi if hi < 10**6 else '+'}" for lo, hi in buckets],
                          "n": tot.tolist(), "rate": (hit / tot).round(4).tolist(), "expected_rate": P_NUM,
                          "chi2": chi_h, "p": float(stats.chi2.sf(chi_h, len(buckets))),
                          "corr_gap_outcome": float(pr[0]), "corr_p": float(pr[1])}
    tests.append(("Hazard theo gap (chi2)", R["gap"]["hazard"]["p"]))

    # ---- D. Trong so thoi gian
    ew = {k: comp_next[k] for k in HALF_LIVES}
    ew["f_all"] = comp_next["f_all"]
    names = list(ew)
    rho = np.array([[stats.spearmanr(ew[x], ew[y])[0] for y in names] for x in names])
    top10 = {k: set(np.argsort(-ew[k])[:10] + 1) for k in names}
    appear = {n + 1: sum((n + 1) in top10[k] for k in names) for n in range(N_NUM)}
    R["weighted"] = {"schemes": names, "spearman": rho.round(3).tolist(),
                     "top10": {k: sorted(map(int, v)) for k, v in top10.items()},
                     "top10_agreement": {str(k): v for k, v in appear.items() if v >= 1}}

    # ---- E. Markov / chuyen tiep
    prev, cur = X[:-1], X[1:]
    p11 = float((prev * cur).sum() / prev.sum())
    p01 = float(((1 - prev) * cur).sum() / (1 - prev).sum())
    ov = (prev * cur).sum(1)
    ov_probs = HYP
    ct = cat_test(ov.astype(int), ov_probs, 6)
    shared_pairs = (ov * (ov - 1) / 2)
    exp_sp = float((ov_probs * np.array([k * (k - 1) / 2 for k in range(7)])).sum())
    var_sp = float((ov_probs * np.array([(k * (k - 1) / 2) ** 2 for k in range(7)])).sum() - exp_sp ** 2)
    z_sp = (shared_pairs.mean() - exp_sp) / math.sqrt(var_sp / len(ov))
    R["markov"] = {"p_appear_given_prev1": p11, "p_appear_given_prev0": p01, "p_base": P_NUM,
                   "overlap_mean": float(ov.mean()), "overlap_expected": 36 / 55,
                   "overlap_dist_test": ct, "shared_pairs_mean": float(shared_pairs.mean()),
                   "shared_pairs_expected": exp_sp, "shared_pairs_z": float(z_sp),
                   "markov_chi_obs": obs["markov_chi"], "markov_chi_sim_mean": mc["markov_chi"]["sim_mean"],
                   "markov_p_mc": mc["markov_chi"]["p_mc"]}
    tests += [("Overlap lien ky (phan phoi sieu boi)", ct["p"]),
              ("Cap chung lien ky (z)", float(2 * stats.norm.sf(abs(z_sp))))]
    s_series = nums.sum(1).astype(float)
    sc = s_series - s_series.mean()
    acf = [float((sc[l:] * sc[:-l]).sum() / (sc ** 2).sum()) for l in range(1, 11)]
    lb = T * (T + 2) * sum(acf[l - 1] ** 2 / (T - l) for l in range(1, 11))
    R["markov"]["sum_acf"] = acf
    R["markov"]["ljung_box_p"] = float(stats.chi2.sf(lb, 10))
    R["markov"]["acf_bound95"] = 1.96 / math.sqrt(T)
    tests.append(("Ljung-Box tong cac so (lag 1-10)", R["markov"]["ljung_box_p"]))

    # ---- F. Cap / bo ba
    C = X.T @ X
    iu = np.triu_indices(N_NUM, 1)
    pc = C[iu]
    Ep = T * P_PAIR
    zp = (pc - Ep) / math.sqrt(T * P_PAIR * (1 - P_PAIR))
    pp = binom_two_sided(pc, T, P_PAIR)
    ppa = bh_adjust(pp)
    order = np.argsort(-np.abs(zp))[:10]
    R["pairs"] = {"expected": Ep, "n_pairs": len(pc), "chi2": obs["pair_chi"], "chi2_p_mc": mc["pair_chi"]["p_mc"],
                  "maxz_obs": obs["pair_maxz"], "maxz_sim_mean": mc["pair_maxz"]["sim_mean"],
                  "maxz_p_mc": mc["pair_maxz"]["p_mc"], "n_sig_raw": int((pp < 0.05).sum()),
                  "n_sig_bh": int((ppa < 0.05).sum()),
                  "top": [{"pair": [int(iu[0][i] + 1), int(iu[1][i] + 1)], "count": int(pc[i]),
                           "z": float(zp[i]), "p_bh": float(ppa[i])} for i in order],
                  "triple_expected": T * comb(52, 3) / TOTAL_COMBOS,
                  "triple_max_obs": obs["triple_max"], "triple_max_sim_mean": mc["triple_max"]["sim_mean"],
                  "triple_max_p_mc": mc["triple_max"]["p_mc"]}
    tests += [("Cap: chi2 tong the", mc["pair_chi"]["p_mc"]), ("Cap: max |z|", mc["pair_maxz"]["p_mc"]),
              ("Bo ba: so lan lon nhat", mc["triple_max"]["p_mc"])]
    pair_last = {}
    # ---- G. Cau truc
    print("[MC] Mo phong phan phoi cau truc ...")
    chunks = []
    for _ in range(math.ceil(a.mc_draws / 100000)):
        chunks.append(sim_draws(rng, 100000) + 1)
    mcn = np.vstack(chunks)
    md, od = draw_metrics(mcn), draw_metrics(nums)
    struct = {}
    for key, mxc in [("odd", 6), ("low", 6), ("consec", 5), ("ndecades", 6)]:
        pr_ = np.bincount(np.clip(md[key], 0, mxc), minlength=mxc + 1) / len(mcn)
        r = cat_test(od[key], pr_, mxc)
        struct[key] = r
        tests.append((f"Cau truc: {key}", r["p"]))
    ref = rng.choice(len(mcn), 50000, replace=False)
    for key in ["sum", "range", "std"]:
        ksr = stats.ks_2samp(od[key], md[key][ref])
        struct[key] = {"obs_mean": float(od[key].mean()), "sim_mean": float(md[key].mean()),
                       "obs_sd": float(od[key].std()), "sim_sd": float(md[key].std()),
                       "cohen_d": float((od[key].mean() - md[key].mean()) / md[key].std()),
                       "ks": float(ksr.statistic), "p": float(ksr.pvalue)}
        tests.append((f"Cau truc: {key} (KS)", float(ksr.pvalue)))
    R["structure"] = struct
    sum_lo, sum_hi = np.percentile(md["sum"], [5, 95])
    R["mc_percentiles"] = {"sum_p5": float(sum_lo), "sum_p95": float(sum_hi), "sum_mean": float(md["sum"].mean())}

    # ---- So dac biet
    spc = np.bincount(special, minlength=56)[1:]
    chi_sp = float(((spc - T / 55) ** 2 / (T / 55)).sum())
    R["special"] = {"chi2": chi_sp, "df": 54, "p": float(stats.chi2.sf(chi_sp, 54)),
                    "min": int(special.min()), "max": int(special.max()),
                    "in_main": info["special_in_main"]}
    tests.append(("So dac biet: chi2 dong deu", R["special"]["p"]))

    # ---- Walk-forward
    print(f"[Walk-forward] t0={a.t0}, so buoc={T - a.t0} ...")
    bt, comps_bt = walk_forward(X, a.t0, rng)
    R["backtest"] = bt
    model_names = [k for k in bt if not k.startswith("_")]
    for k in model_names:
        tests.append((f"Backtest top-6 so khop: {k}", bt[k]["p_matches"]))
    R["backtest_bonferroni_m"] = len(model_names)

    # ---- Tong hop kiem dinh
    pv = np.array([p for _, p in tests])
    R["multiplicity"] = {"n_tests": len(pv), "n_p_lt_005": int((pv < 0.05).sum()),
                         "expected_false_pos": 0.05 * len(pv),
                         "n_bh_lt_005": int((bh_adjust(pv) < 0.05).sum()),
                         "list": [{"test": n, "p": p, "p_bh": float(q)} for (n, p), q in zip(tests, bh_adjust(pv))]}

    # ---- Ensemble cho ky tiep theo
    comp = comp_next
    Zc = {k: zrow(comp[k])[0] for k in ["f_all", "f_200", "f_100", "f_50", "f_20", "ew10", "ew30", "ew100", "ew300",
                                         "gap", "markov", "pair"]}
    ens = np.mean([Zc[k] for k in ENS_COMPONENTS], axis=0)
    rk = {k: stats.rankdata(-Zc[k], method="min") for k in ENS_COMPONENTS}
    agree = sum((rk[k] <= 15).astype(int) for k in ENS_COMPONENTS)
    pct_rank = np.mean([stats.rankdata(Zc[k]) / N_NUM for k in ENS_COMPONENTS], axis=0)
    std_rank = np.std([stats.rankdata(Zc[k]) / N_NUM for k in ENS_COMPONENTS], axis=0)
    cons_score = zrow(pct_rank - 0.5 * std_rank)[0]
    rec_score = zrow(0.4 * Zc["ew10"] + 0.3 * Zc["f_20"] + 0.3 * Zc["f_50"])[0]
    long_score = zrow(0.4 * Zc["f_all"] + 0.3 * Zc["f_200"] + 0.3 * Zc["ew300"])[0]

    # rang buoc cau truc tu MC
    combos = np.array(list(itertools.combinations(range(1, 56), 6)), dtype=np.int16) if False else None

    def structural_ok(arr):  # arr (M,6) gia tri 1..55 da sap xep
        m = draw_metrics(arr)
        return ((m["sum"] >= sum_lo) & (m["sum"] <= sum_hi) & (m["odd"] >= 2) & (m["odd"] <= 4) &
                (m["low"] >= 2) & (m["low"] <= 4) & (m["consec"] <= 1) & (m["ndecades"] >= 4))

    def best_combo(score, pool=20, forbid=None, max_overlap=2):
        pidx = np.sort(np.argsort(-score)[:pool])
        cb = np.array(list(itertools.combinations(pidx, 6)))
        vals = cb + 1
        ok = structural_ok(vals)
        if forbid:
            for f in forbid:
                ok &= np.isin(vals, f).sum(1) <= max_overlap
        if not ok.any():
            return None
        sc = score[cb].sum(1)
        sc[~ok] = -1e9
        return sorted((cb[np.argmax(sc)] + 1).tolist())

    cand = {}
    cand["A"] = ("Ensemble cân bằng", best_combo(ens), ens)
    cand["B"] = ("Thận trọng (đồng thuận)", best_combo(cons_score), cons_score)
    cand["C"] = ("Thiên về gần đây", best_combo(rec_score), rec_score)
    cand["D"] = ("Thiên về dài hạn", best_combo(long_score), long_score)
    prior = [c[1] for c in cand.values()]
    cand["E"] = ("Đa dạng hóa", best_combo(ens, pool=28, forbid=prior, max_overlap=2), ens)
    cand_out = {}
    for key, (label, combo, sc) in cand.items():
        arr = np.array([combo])
        m = draw_metrics(arr)
        cand_out[key] = {"label": label, "numbers": combo, "ensemble_score": float(ens[np.array(combo) - 1].mean()),
                         "sum": int(m["sum"][0]), "odd": int(m["odd"][0]), "low": int(m["low"][0]),
                         "sum_percentile_mc": float((md["sum"] <= m["sum"][0]).mean() * 100),
                         "p_combo": 1 / TOTAL_COMBOS}
    R["candidates"] = cand_out

    rows = []
    for i in range(N_NUM):
        rows.append({"number": i + 1, "ensemble_z": float(ens[i]),
                     "z_all": float(freq_tab[T]["z"][i]), "z_200": float(freq_tab[200]["z"][i]),
                     "z_50": float(freq_tab[50]["z"][i]), "ew30_z": float(Zc["ew30"][i]),
                     "gap_now": int(gap_now[i]), "gap_pct": gap_rows[i]["gap_percentile"],
                     "markov_p": float(comp["markov"][i]), "pair_z": float(Zc["pair"][i]),
                     "agreement_top15": int(agree[i]),
                     "top10_weight_schemes": int(appear[i + 1])})
    rank_df = pd.DataFrame(rows).sort_values("ensemble_z", ascending=False).reset_index(drop=True)
    rank_df["confidence"] = np.where(rank_df["agreement_top15"] >= 5, "Thấp", "Rất thấp")
    rank_df.insert(0, "rank", rank_df.index + 1)
    rank_df.to_csv(out / "power655_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gap_rows).to_csv(out / "power655_gap_stats.csv", index=False, encoding="utf-8-sig")
    R["ranking_top20"] = rank_df.head(20).to_dict("records")

    with open(out / "power655_results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(R, rank_df, out / "power655_report.md")
    print(f"[Hoan tat] {time.time() - t_start:.0f}s | Da luu bao cao + JSON + CSV vao {out}")


# ------------------------------------------------------------------ REPORT
def md_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_report(R, rank_df, path):
    m, d, bt = R["meta"], R["data"], R["backtest"]
    L = []
    L.append("# Phân tích thống kê Vietlott Power 6/55\n")
    L.append(f"*Ngày phân tích: {m['analysis_date']} · Seed: {m['seed']} · CSV: `{m['csv']}`*\n")
    L.append("> **Lưu ý:** Power 6/55 được thiết kế để ngẫu nhiên. Báo cáo này là xếp hạng thống kê, "
             "KHÔNG phải dự báo chắc chắn. Mọi tổ hợp 6 số đều có xác suất như nhau.\n")
    L.append("## 1. Dữ liệu\n")
    L.append(md_table(["Mục", "Giá trị"], [
        ["Số kỳ phân tích", f"{m['n_draws']} (trên {d['rows_raw']} dòng)"],
        ["Khoảng kỳ", f"{d['draw_no_min']} – {d['draw_no_max']} (liên tục: {'có' if d['contiguous'] else 'không'})"],
        ["Khoảng ngày", f"{d['date_min']} → {d['date_max']}"],
        ["Kỳ mới nhất", f"{m['latest_draw']} ({m['latest_date']}): {m['latest_numbers']} + đặc biệt {m['latest_special']}"],
        ["Số đặc biệt", f"trong 1–55; trùng số chính: {d['special_in_main']} kỳ"],
        ["Tổ hợp 6 số lặp", d["duplicate_combinations"]],
        ["Vấn đề dữ liệu", "; ".join(d["issues"]) if d["issues"] else "Không phát hiện"]]))
    b = R["baseline"]
    L.append("\n## 2. Mô hình cơ sở (lý thuyết)\n")
    L.append(f"- P(một số xuất hiện) = 6/55 = {b['p_number']:.4f}\n- P(một cặp) = 30/2970 = {b['p_pair']:.5f}\n"
             f"- P(một bộ ba) = {b['p_triple']:.6f}\n- P(một tổ hợp 6 số) = 1/{b['combos']:,}\n"
             f"- Số trùng kỳ vọng khi chọn 6 số bất kỳ: {b['expected_matches_top6']:.4f}\n")
    L.append("\n## 3. Phát hiện thống kê\n")
    f = R["freq"]
    L.append("**Tần suất** (kiểm định chi bình phương, p theo Monte Carlo; số có |z| bất thường sau hiệu chỉnh BH):\n")
    L.append(md_table(["Cửa sổ", "chi²", "p (MC)", "Số p<0.05 (thô)", "Kỳ vọng ngẫu nhiên", "Sau BH", "max |z|"],
                      [[w if w != str(m['n_draws']) else f"toàn bộ ({w})", f"{v['chi2_stat']:.1f}", fp(v["chi2_p_mc"]),
                        v["n_sig_raw"], "≈2.75", v["n_sig_bh"], f"{v['max_abs_z']:.2f}"] for w, v in f.items()]))
    g = R["gap"]
    h = g["hazard"]
    L.append(f"\n**Khoảng cách (gap):** trung bình gộp {g['pooled_mean_gap']:.2f} (lý thuyết {g['theory_mean_gap']:.2f}); "
             f"KS so với phân phối hình học: p = {fp(g['ks_geom_p'])}. "
             f"Kiểm tra \"số lâu chưa ra\": tỉ lệ xuất hiện theo gap hiện tại\n")
    L.append(md_table(["Gap"] + h["buckets"], [["Số quan sát"] + [int(x) for x in h["n"]],
                                                ["Tỉ lệ xuất hiện"] + h["rate"]]))
    L.append(f"\nKỳ vọng {h['expected_rate']:.4f} ở mọi nhóm; chi² p = {fp(h['p'])}; "
             f"tương quan gap–kết quả r = {h['corr_gap_outcome']:.4f} (p = {fp(h['corr_p'])}). "
             "→ Không có bằng chứng cho giả thuyết \"số đến hạn\".\n")
    w = R["weighted"]
    L.append("\n**Trọng số thời gian** (half-life 10/30/100/300 kỳ): hệ số Spearman giữa các bảng xếp hạng:\n")
    L.append(md_table([""] + w["schemes"], [[w["schemes"][i]] + [f"{x:.2f}" for x in row] for i, row in enumerate(w["spearman"])]))
    mk = R["markov"]
    L.append(f"\n**Markov / phụ thuộc giữa các kỳ:** P(ra | kỳ trước ra) = {mk['p_appear_given_prev1']:.4f} "
             f"so với P(ra | kỳ trước không ra) = {mk['p_appear_given_prev0']:.4f} (cơ sở {mk['p_base']:.4f}). "
             f"Số trùng trung bình giữa 2 kỳ liên tiếp {mk['overlap_mean']:.3f} (kỳ vọng {mk['overlap_expected']:.3f}); "
             f"phân phối trùng chi² p = {fp(mk['overlap_dist_test']['p'])}; "
             f"thống kê Markov gộp {mk['markov_chi_obs']:.1f} vs mô phỏng {mk['markov_chi_sim_mean']:.1f} (p MC = {fp(mk['markov_p_mc'])}); "
             f"Ljung–Box tổng các số p = {fp(mk['ljung_box_p'])}.\n")
    pr = R["pairs"]
    L.append(f"\n**Cặp / bộ ba:** kỳ vọng mỗi cặp ≈ {pr['expected']:.1f} lần. chi² tổng thể p(MC) = {fp(pr['chi2_p_mc'])}; "
             f"max|z| = {pr['maxz_obs']:.2f} (mô phỏng trung bình {pr['maxz_sim_mean']:.2f}, p = {fp(pr['maxz_p_mc'])}); "
             f"số cặp p<0.05 thô: {pr['n_sig_raw']}/{pr['n_pairs']} (≈{0.05 * pr['n_pairs']:.0f} do ngẫu nhiên), sau BH: {pr['n_sig_bh']}. "
             f"Bộ ba: kỳ vọng ≈{pr['triple_expected']:.2f} lần/bộ ba (mẫu quá nhỏ để suy luận); lần lặp lớn nhất {pr['triple_max_obs']} "
             f"(mô phỏng {pr['triple_max_sim_mean']:.1f}, p = {fp(pr['triple_max_p_mc'])}).\n")
    s = R["structure"]
    L.append("\n**Cấu trúc kỳ quay vs mô phỏng ngẫu nhiên:**\n")
    rows = [[k, f"{s[k]['obs_mean']:.2f}", f"{s[k]['sim_mean']:.2f}", f"{s[k]['cohen_d']:.3f}", fp(s[k]['p'])]
            for k in ["sum", "range", "std"]]
    L.append(md_table(["Chỉ số", "Quan sát (TB)", "Mô phỏng (TB)", "Cohen d", "p (KS)"], rows))
    L.append(md_table(["Chỉ số", "p (chi²)"], [[k, fp(s[k]["p"])] for k in ["odd", "low", "consec", "ndecades"]]))
    sp = R["special"]
    L.append(f"\n**Số đặc biệt:** phân phối đều 1–55: chi² p = {fp(sp['p'])}.\n")
    mu = R["multiplicity"]
    L.append(f"\n**Tổng hợp {mu['n_tests']} kiểm định:** {mu['n_p_lt_005']} có p<0.05 (kỳ vọng do ngẫu nhiên ≈ {mu['expected_false_pos']:.1f}); "
             f"sau hiệu chỉnh Benjamini–Hochberg còn {mu['n_bh_lt_005']}.\n")
    near = [x for x in mu["list"] if x["p"] < 0.05]
    if near:
        L.append(md_table(["Kiểm định", "p thô", "p BH"], [[x["test"], fp(x["p"]), fp(x["p_bh"])] for x in near]))
    L.append("\n## 4. So sánh mô hình & Backtest (walk-forward)\n")
    L.append(f"Huấn luyện bằng các kỳ cũ, dự đoán kỳ kế tiếp, trượt cửa sổ qua **{bt['_steps']} kỳ** (bắt đầu từ kỳ thứ 200). "
             f"Mỗi mô hình chọn top-6; so với kỳ vọng ngẫu nhiên {MU_MATCH:.4f} số trùng. "
             f"Nền ngẫu nhiên mô phỏng: TB {bt['_random_sim']['mean']:.4f}, khoảng 95% [{bt['_random_sim']['range95'][0]:.3f}; {bt['_random_sim']['range95'][1]:.3f}].\n")
    rows = []
    for k in [x for x in bt if not x.startswith("_")]:
        v = bt[k]
        rows.append([k, f"{v['mean_matches']:.4f}", f"[{v['ci95'][0]:.3f}; {v['ci95'][1]:.3f}]", fp(v["p_matches"]),
                     "/".join(f"{x*100:.1f}" for x in v["dist_obs"]), f"{v['mean_rank_pct']:.4f}", fp(v["rank_t_p"])])
    exp_d = bt["f_all"]["dist_exp"]
    L.append(md_table(["Mô hình", "TB số trùng", "CI95 (bootstrap)", "p (so với ngẫu nhiên)",
                       "% trùng 0/1/2/3+", "Hạng TB của số trúng (kỳ vọng 0.5)", "p (t-test hạng)"], rows))
    L.append(f"\nPhân phối kỳ vọng ngẫu nhiên (% trùng 0/1/2/3+): {'/'.join(f'{x*100:.1f}' for x in exp_d)}. "
             f"Bonferroni cho {R['backtest_bonferroni_m']} mô hình: ngưỡng p = {0.05 / R['backtest_bonferroni_m']:.4f}.\n")
    wc = bt["_weights_check"]
    L.append(f"\n**Kiểm tra trọng số ensemble** (học trọng số ở nửa đầu, đánh giá ở nửa sau): trọng số {wc['weights']}; "
             f"nửa sau – trọng số học được: {wc['valweighted_second_half']['mean_matches']:.4f} số trùng "
             f"(p = {fp(wc['valweighted_second_half']['p_matches'])}); trọng số đều: {wc['equal_second_half']['mean_matches']:.4f} "
             f"(p = {fp(wc['equal_second_half']['p_matches'])}).\n")
    L.append("\n**Mô hình xác suất (Brier score so với 6/55 cố định):**\n")
    L.append(md_table(["Mô hình", "Brier", "Brier cơ sở", "BSS", "p"],
                      [[k, f"{v['brier']:.6f}", f"{v['brier_base']:.6f}", f"{v['bss']:.5f}", fp(v["p"])]
                       for k, v in bt["_brier"].items()]))
    L.append("\n## 5. Xếp hạng số (top 20, cho kỳ kế tiếp)\n")
    L.append("Điểm ensemble = trung bình z-score chuẩn hóa của 7 thành phần (tần suất toàn bộ/100/20 kỳ, EWMA, gap, Markov, cặp), trọng số đều.\n")
    top = rank_df.head(20)
    L.append(md_table(["Hạng", "Số", "Điểm ensemble", "z tần suất (toàn bộ / 200 / 50)", "z gần đây (EWMA30)",
                       "Gap hiện tại (phân vị)", "Đồng thuận (top15 /7)", "Độ tin cậy"],
                      [[r["rank"], int(r["number"]), f"{r['ensemble_z']:.2f}",
                        f"{r['z_all']:.2f} / {r['z_200']:.2f} / {r['z_50']:.2f}", f"{r['ew30_z']:.2f}",
                        f"{int(r['gap_now'])} ({r['gap_pct']*100:.0f}%)", int(r["agreement_top15"]), r["confidence"]]
                       for _, r in top.iterrows()]))
    L.append("\n## 6. Tổ hợp đề xuất\n")
    L.append("Chọn từ nhóm số điểm cao nhất, lọc theo cấu trúc \"bình thường\" của mô phỏng (tổng trong P5–P95, "
             "2–4 lẻ, 2–4 thấp, ≤1 cặp liên tiếp, ≥4 nhóm chục). **Xác suất trúng jackpot của mọi tổ hợp đều bằng nhau: "
             f"1/{TOTAL_COMBOS:,}.**\n")
    L.append(md_table(["Loại", "Mô tả", "6 số", "Điểm TB", "Tổng (phân vị MC)", "Lẻ/Chẵn"],
                      [[k, v["label"], " – ".join(f"{x:02d}" for x in v["numbers"]), f"{v['ensemble_score']:.2f}",
                        f"{v['sum']} ({v['sum_percentile_mc']:.0f}%)", f"{v['odd']}/{6 - v['odd']}"]
                       for k, v in R["candidates"].items()]))
    L.append("\n## 7. Kết luận quan trọng\n")
    best = min([(bt[k]["p_matches"], k) for k in bt if not k.startswith("_")])
    sig = [k for k in bt if not k.startswith("_") and bt[k]["p_matches"] < 0.05 / R["backtest_bonferroni_m"]]
    L.append(f"- Mô hình có p thấp nhất trong backtest: **{best[1]}** (p = {fp(best[0])}); "
             f"số mô hình vượt ngưỡng Bonferroni: **{len(sig)}**.\n"
             "- Dữ liệu lịch sử không cho thấy tín hiệu dự báo có ý nghĩa thống kê nếu các p ở trên không vượt hiệu chỉnh đa kiểm định; "
             "mọi lệch nhỏ được kỳ vọng xuất hiện do ngẫu nhiên.\n"
             "- Frequency, gap, Markov và cặp số không tạo ra năng lực dự báo tự động; mọi \"mẫu hình\" có thể chỉ là nhiễu lấy mẫu.\n"
             "- Bảng xếp hạng và tổ hợp ở trên là kết quả xếp hạng thống kê, không phải dự báo đảm bảo.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()