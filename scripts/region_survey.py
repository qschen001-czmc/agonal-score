# -*- coding: utf-8 -*-
"""region_survey.py —— 脑区调查、leave-one-out 与核心面板敏感性。

三件事：
  1. 逐脑区拟合 IEG ~ DTHHRDY + RIN + 缺血时间 + 年龄 + 性别，报告效应量与 p；
     并给出"剔除呼吸机组"的版本，区分「呼吸机效应」与「跨等级趋势」。
  2. leave-one-out：逐个去掉面板中的一个基因，看主效应是否稳定。
  3. 核心 6 基因面板 vs 全 18 基因面板。

判据：criteria_thresholds.json（region_min_n、面板定义）。
"""
from __future__ import annotations

import argparse
import io
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CRIT = os.path.join(ROOT, "criteria", "criteria_thresholds.json")
PROC = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "tables")


def load_cfg() -> dict:
    with io.open(CRIT, encoding="utf-8") as fh:
        return json.load(fh)


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {"beta": beta, "se": se, "p": p, "r2": float(r2), "n": n, "dof": dof}


COVS = [("RIN", "RIN"), ("ischemic_time", "ischemic_time"), ("AGE_num", "AGE_num"), ("SEX_male", "SEX_male")]


def fit_region(df: pd.DataFrame) -> dict | None:
    d = df.dropna(subset=["IEG_score", "DTHHRDY", "RIN", "ischemic_time", "AGE_num", "SEX_male"])
    if len(d) < 30:
        return None
    y = d["IEG_score"].to_numpy(float)
    cols = ["DTHHRDY"] + [c for c, _ in COVS]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    m = ols(y, X)
    # 未校正模型也存下来，保证全套图表只有一个数据源
    Xu = np.column_stack([np.ones(len(d)), d["DTHHRDY"].to_numpy(float)])
    mu = ols(y, Xu)
    rho, rho_p = stats.spearmanr(d["DTHHRDY"], d["IEG_score"])
    return {"n": len(d), "n_vent": int((d["DTHHRDY"] == 0).sum()),
            "beta_unadj": float(mu["beta"][1]), "p_unadj": float(mu["p"][1]),
            "beta": float(m["beta"][1]), "se": float(m["se"][1]), "p": float(m["p"][1]),
            "r2": m["r2"], "rho": float(rho), "rho_p": float(rho_p),
            "median_dth0": float(np.median(d.loc[d["DTHHRDY"] == 0, "IEG_score"]))
            if (d["DTHHRDY"] == 0).any() else np.nan,
            "median_rest": float(np.median(d.loc[d["DTHHRDY"] > 0, "IEG_score"]))}


def fit_region_no_vent(df: pd.DataFrame) -> dict | None:
    d = df[df["DTHHRDY"] > 0].dropna(subset=["IEG_score", "DTHHRDY", "RIN", "ischemic_time",
                                             "AGE_num", "SEX_male"])
    if len(d) < 30 or d["DTHHRDY"].nunique() < 3:
        return None
    y = d["IEG_score"].to_numpy(float)
    cols = ["DTHHRDY"] + [c for c, _ in COVS]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    m = ols(y, X)
    rho, rho_p = stats.spearmanr(d["DTHHRDY"], d["IEG_score"])
    return {"n_no_vent": len(d), "beta_no_vent": float(m["beta"][1]), "p_no_vent": float(m["p"][1]),
            "rho_no_vent": float(rho), "rho_p_no_vent": float(rho_p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    cfg = load_cfg()
    panel_all = cfg["ieg"]["panel"]
    panel_core = cfg["ieg"]["core_panel"]
    min_n = int(cfg["data"]["gtex"].get("region_min_n", 30))

    df = pd.read_csv(os.path.join(PROC, "ieg_samples.tsv"), sep="\t")
    df["SEX_male"] = (df["SEX"] == "Male").astype(float)
    raw = pd.read_csv(os.path.join(PROC, "ieg_expression.tsv"), sep="\t", index_col=0)

    print(f"样本 {len(df)}，脑区 {df['tissue'].nunique()} 个")

    # ---------- 1. 脑区调查 ----------
    rows = []
    for t, g in df.groupby("tissue"):
        r = fit_region(g)
        if r is None or r["n"] < min_n:
            continue
        nv = fit_region_no_vent(g)
        row = {"tissue": t, **r}
        if nv:
            row.update(nv)
        rows.append(row)
        print(f"  {t[:46]:46s} n={r['n']:4d} β={r['beta']:+.3f} p={r['p']:.2g} "
              f"| ρ={r['rho']:+.2f}" + (f" | 去呼吸机 β={nv['beta_no_vent']:+.3f} p={nv['p_no_vent']:.2g}" if nv else ""))
    reg = pd.DataFrame(rows).sort_values("beta")
    reg.to_csv(os.path.join(OUT, "region_survey.tsv"), sep="\t", index=False)

    # ---------- 2. leave-one-out（BA9）----------
    primary = cfg["data"]["gtex"]["tissue"]
    ba9 = df[df["tissue"] == primary].copy()
    ba9 = ba9.dropna(subset=["IEG_score", "DTHHRDY", "RIN", "ischemic_time", "AGE_num", "SEX_male"])
    cols = ["DTHHRDY"] + [c for c, _ in COVS]
    loo = []
    base = fit_region(ba9)
    sample_ids = ba9["SAMPID"].tolist()
    # ieg_expression.tsv 的行是样本、列是基因（写文件时做了 .T）
    genes_avail = [g for g in panel_all if g in raw.columns]
    sub = raw.reindex(index=sample_ids)[genes_avail].astype(float)
    logtpm = np.log2(sub + 1)          # 样本 x 基因
    logtpm = logtpm.T                  # 转成 基因 x 样本，便于按基因求均值
    for drop in [None] + list(logtpm.index):
        genes = [g for g in logtpm.index if g != drop]
        score = logtpm.loc[genes].mean(axis=0)
        d = ba9.set_index("SAMPID").copy()
        d["IEG_score"] = score.reindex(d.index).values
        r = fit_region(d.reset_index())
        loo.append({"dropped": drop or "(none)", "n_genes": len(genes),
                    "beta": r["beta"], "p": r["p"], "rho": r["rho"]})
    loo_df = pd.DataFrame(loo)
    loo_df.to_csv(os.path.join(OUT, "loo_panel.tsv"), sep="\t", index=False)
    print(f"\nleave-one-out（{primary}）：β 范围 {loo_df['beta'].min():+.3f} ~ {loo_df['beta'].max():+.3f}，"
          f"全部 p<0.05：{(loo_df['p'] < 0.05).all()}")

    # ---------- 3. 核心面板 ----------
    core = [g for g in panel_core if g in logtpm.index]
    core_score = logtpm.loc[core].mean(axis=0)
    d = ba9.set_index("SAMPID").copy()
    d["IEG_score"] = core_score.reindex(d.index).values
    rc = fit_region(d.reset_index())
    print(f"\n核心 {len(core)} 基因面板（{primary}）：β={rc['beta']:+.3f} p={rc['p']:.2g} "
          f"（全 18 基因 β={base['beta']:+.3f} p={base['p']:.2g}）")
    pd.DataFrame([{"panel": f"core{len(core)}", **rc}, {"panel": f"full{len(panel_all)}", **base}]) \
        .to_csv(os.path.join(OUT, "panel_variants.tsv"), sep="\t", index=False)

    print(f"\n写出 {OUT}\\region_survey.tsv、loo_panel.tsv、panel_variants.tsv")
    return 0


def selftest() -> int:
    rng = np.random.default_rng(0)
    n = 120
    dth = rng.integers(0, 5, n).astype(float)
    df = pd.DataFrame({
        "IEG_score": 5.0 - 0.2 * dth + rng.normal(0, 0.3, n),
        "DTHHRDY": dth, "RIN": rng.normal(6.5, 1, n),
        "ischemic_time": rng.normal(900, 300, n),
        "AGE_num": rng.normal(55, 10, n), "SEX_male": rng.integers(0, 2, n).astype(float),
    })
    r = fit_region(df)
    ok1 = abs(r["beta"] + 0.2) < 0.06 and r["p"] < 0.01
    print(f"{'OK  ' if ok1 else 'FAIL'} 逐脑区拟合 β={r['beta']:+.3f}（真值 -0.20）p={r['p']:.2g}")
    ok2 = fit_region(df.head(20)) is None
    print(f"{'OK  ' if ok2 else 'FAIL'} 样本 <30 时返回 None")
    ok = ok1 and ok2
    print("自检：", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
