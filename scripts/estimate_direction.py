# -*- coding: utf-8 -*-
"""estimate_direction.py —— 检验 IEG 评分与 GTEx 死亡过程（DTHHRDY）的关联。

预研一的正式分析。判据来自 criteria_thresholds.json，不在代码里另写阈值。

两个模型：
    单变量：IEG ~ DTHHRDY
    全协变量：IEG ~ DTHHRDY + RIN + 缺血时间 + 年龄 + 性别（+ 脑区，合并分析时）

注意：本机 statsmodels 与 numpy 2 不兼容（np.MachAr 已移除），
因此 OLS 用 numpy 直接解正规方程，t 检验用 scipy.stats.t 自行计算。

用法：
    python estimate_direction.py
    python estimate_direction.py --selftest
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CRIT = os.path.join(ROOT, "criteria", "criteria_thresholds.json")
PROC = os.path.join(ROOT, "data", "derived")


def load_cfg() -> dict:
    with io.open(CRIT, encoding="utf-8") as fh:
        return json.load(fh)


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """最小二乘 + 经典标准误。X 需已含截距列。"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {"beta": beta, "se": se, "t": t, "p": p, "dof": dof, "r2": float(r2), "n": n}


def design(df: pd.DataFrame, cols: list[str], with_intercept: bool = True) -> np.ndarray:
    mats = [np.ones((len(df), 1))] if with_intercept else []
    for c in cols:
        v = df[c].astype(float).to_numpy()
        mats.append(v.reshape(-1, 1))
    return np.hstack(mats)


def vif(df: pd.DataFrame, target: str, others: list[str]) -> float:
    """target 对其余变量的 R^2 → VIF。用于检查 IEG 与 RIN 的共线性。"""
    y = df[target].astype(float).to_numpy()
    X = design(df, others)
    r = ols(y, X)
    return 1.0 / max(1e-12, 1.0 - r["r2"])


def analyse_block(df: pd.DataFrame, label: str, cfg: dict) -> list[dict]:
    # 判据里写的是 age/sex，数据列名是 AGE_num/SEX_male，必须显式映射——
    # 早先直接用白名单过滤，导致 age 与 sex 被静默丢弃，"全协变量"实为两协变量。
    name_map = {"age": "AGE_num", "sex": "SEX_male"}
    wanted = [name_map.get(c, c) for c in cfg["validation"]["covariates"]]
    covs = [c for c in wanted if c in df.columns and df[c].notna().all()]
    df = df.dropna(subset=["IEG_score", "DTHHRDY"] + covs).copy()
    if len(df) < 30:
        print(f"  [skip] {label}: 样本仅 {len(df)}")
        return []

    rows = []
    y = df["IEG_score"].to_numpy(float)
    dth = df["DTHHRDY"].to_numpy(float)

    # 组间差异与单调性
    groups = [g["IEG_score"].to_numpy(float) for _, g in df.groupby("DTHHRDY")]
    kw = stats.kruskal(*groups)
    rho, rho_p = stats.spearmanr(dth, y)

    # 单变量
    m1 = ols(y, design(df, ["DTHHRDY"]))
    # 全协变量
    Xc = design(df, ["DTHHRDY"] + covs)
    m2 = ols(y, Xc)
    v = vif(df, "IEG_score", covs) if covs else float("nan")

    print(f"\n== {label}　n={len(df)}｜脑区/组数={df['DTHHRDY'].nunique()}")
    print(f"   Kruskal-Wallis p = {kw.pvalue:.3g}｜Spearman rho = {rho:+.3f} (p={rho_p:.3g})")
    print(f"   单变量     ：beta(DTHHRDY) = {m1['beta'][1]:+.4f}, p = {m1['p'][1]:.3g}")
    print(f"   全协变量   ：beta(DTHHRDY) = {m2['beta'][1]:+.4f}, p = {m2['p'][1]:.3g}　"
          f"(R^2={m2['r2']:.3f})")
    if covs:
        print(f"   协变量 VIF：" + "，".join(
            f"{c}={vif(df, c, ['IEG_score'] + [x for x in covs if x != c]):.2f}" for c in covs))
    for i, c in enumerate(covs, start=2):
        print(f"      {c:14s} beta={m2['beta'][i]:+.4f}  p={m2['p'][i]:.3g}")

    rows.append({
        "block": label, "n": len(df),
        "kruskal_p": float(kw.pvalue), "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "beta_unadj": float(m1["beta"][1]), "p_unadj": float(m1["p"][1]),
        "beta_adj": float(m2["beta"][1]), "p_adj": float(m2["p"][1]),
        "r2_adj": float(m2["r2"]), "vif_covariates": float(v),
    })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    cfg = load_cfg()
    df = pd.read_csv(os.path.join(PROC, "ieg_samples.tsv"), sep="\t")
    df["SEX_male"] = (df["SEX"] == "Male").astype(float)
    print(f"载入 {len(df)} 个样本；判据：{cfg['validation']['outcome']}，"
          f"α={cfg['validation']['primary_test']['alpha']}")

    primary = cfg["data"]["gtex"]["tissue"]
    repl = cfg["data"]["gtex"].get("replication_tissue")
    rows = []
    # 先在两个脑区合并（脑区作为协变量），再分别做
    pooled = df.copy()
    pooled["tissue_code"] = (pooled["tissue"] == primary).astype(float)
    rows += analyse_block(pooled.assign(**{}), "合并两脑区（脑区入模）", cfg) if False else []
    for t in (primary, repl):
        if t is None:
            continue
        sub = df[df["tissue"] == t].copy()
        rows += analyse_block(sub, t, cfg)

    # 合并分析：把脑区也作为协变量
    pooled = df.copy()
    pooled["tissue_code"] = (pooled["tissue"] == primary).astype(float)
    covs_all = [c for c in ("RIN", "ischemic_time", "AGE_num", "SEX_male", "tissue_code")
                if pooled[c].notna().all()]
    pooled = pooled.dropna(subset=["IEG_score", "DTHHRDY"] + covs_all)
    y = pooled["IEG_score"].to_numpy(float)
    m = ols(y, design(pooled, ["DTHHRDY"] + covs_all))
    print(f"\n== 合并（n={len(pooled)}）：beta(DTHHRDY) = {m['beta'][1]:+.4f}, p = {m['p'][1]:.3g}")
    rows.append({
        "block": "合并两脑区", "n": len(pooled),
        "kruskal_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan,
        "beta_unadj": np.nan, "p_unadj": np.nan,
        "beta_adj": float(m["beta"][1]), "p_adj": float(m["p"][1]),
        "r2_adj": float(m["r2"]), "vif_covariates": np.nan,
    })

    out = pd.DataFrame(rows)
    dst = os.path.join(ROOT, "tables", "direction_results.tsv")
    out.to_csv(dst, sep="\t", index=False)
    print(f"\n写出 {dst}")

    # 依判据给结论
    primary_row = out[out["block"] == primary]
    if len(primary_row):
        r = primary_row.iloc[0]
        sig_adj = r["p_adj"] < cfg["validation"]["primary_test"]["alpha"]
        same_sign = np.sign(r["beta_adj"]) == np.sign(r["beta_unadj"])
        verdict = "通过" if (sig_adj and same_sign) else "未通过"
        print(f"\n主分析判定：{verdict}（校正后 p={r['p_adj']:.3g}，方向{'一致' if same_sign else '不一致'}）")
        print(f"方向：DTHHRDY 每升高 1 级，IEG 评分变化 {r['beta_adj']:+.4f}"
              f"（{'升高' if r['beta_adj'] > 0 else '降低'}）")
    return 0


def selftest() -> int:
    rng = np.random.default_rng(0)
    n = 200
    dth = rng.integers(0, 5, n).astype(float)
    rin = rng.normal(6.5, 1.0, n)
    y = 5.0 - 0.15 * dth + 0.05 * rin + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"IEG_score": y, "DTHHRDY": dth, "RIN": rin})
    m = ols(y, design(df, ["DTHHRDY", "RIN"]))
    ok1 = abs(m["beta"][1] - (-0.15)) < 0.05
    print(f"{'OK  ' if ok1 else 'FAIL'} beta 估计 {m['beta'][1]:+.3f}（真值 -0.15）")
    # 已知答案的 OLS：y = 1 + 2x
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yy = 1 + 2 * x
    m2 = ols(yy, design(pd.DataFrame({"x": x}), ["x"]))
    ok2 = abs(m2["beta"][0] - 1) < 1e-9 and abs(m2["beta"][1] - 2) < 1e-9
    print(f"{'OK  ' if ok2 else 'FAIL'} 精确解 beta = {np.round(m2['beta'], 6)}（应为 [1, 2]）")
    v = vif(pd.DataFrame({"a": x, "b": 2 * x}), "a", ["b"])
    ok3 = v > 1000
    print(f"{'OK  ' if ok3 else 'FAIL'} 完全共线时 VIF = {v:.0f}（应极大）")
    ok = ok1 and ok2 and ok3
    print("自检：", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
