# -*- coding: utf-8 -*-
"""compute_remaining.py —— 一次算出正文中所有待填的统计量。

覆盖：
  S1 逐级调整后均值、线性/二次/分类检验
  S2 分类 DTHHRDY 模型
  S3 评分与各协变量的 Pearson / Spearman
  S4 Huber 回归、剔除 top1%、DFBETA 影响诊断
  S5 缺失数据分布与缺失指示模型
  S6 十三区合并模型的 donor 聚类稳健标准误
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "tables")
REPORT = os.path.join(ROOT, "tables", "remaining_stats.txt")

COVS = ["RIN", "ischemic_time", "AGE_num", "SEX_male"]
lines = []


def P(s=""):
    lines.append(s)


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    n, k = X.shape
    dof = n - k
    s2 = float(r @ r) / dof
    XtXi = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(s2 * XtXi))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return dict(beta=beta, se=se, t=t, p=p, dof=dof, r2=r2, resid=r, XtXi=XtXi, s2=s2)


def design(df, cols):
    return np.column_stack([np.ones(len(df))] + [df[c].astype(float).to_numpy() for c in cols])


def huber(y, X, c=1.345, iters=100, tol=1e-10):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(iters):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        s = s if s > 0 else 1.0
        u = np.abs(r) / s
        w = np.where(u <= c, 1.0, c / np.maximum(u, 1e-12))
        Xw = X * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    r = y - X @ beta
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    u = np.abs(r) / (s if s > 0 else 1.0)
    w = np.where(u <= c, 1.0, c / np.maximum(u, 1e-12))
    XtWX = X.T @ (X * w[:, None])
    cov = np.linalg.pinv(XtWX) * (np.sum(w * r**2) / (len(y) - X.shape[1]))
    se = np.sqrt(np.diag(cov))
    return dict(beta=beta, se=se, p=2 * stats.t.sf(np.abs(beta / se), len(y) - X.shape[1]))


def cluster_se(X, r, groups, XtXi):
    """Liang-Zeger 聚类稳健协方差（按 donor 聚类）。"""
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        Xg = X[m]
        rg = r[m]
        s = Xg.T @ rg
        meat += np.outer(s, s)
    return XtXi @ meat @ XtXi


df = pd.read_csv(os.path.join(PROC, "ieg_samples.tsv"), sep="\t")
df["SEX_male"] = (df["SEX"] == "Male").astype(float)
BA9 = "Brain - Frontal Cortex (BA9)"

# ---------------------------------------------------------------- S1
P("=" * 70)
P("S1  逐级调整后均值（BA9）与趋势检验")
P("=" * 70)
d = df[df["tissue"] == BA9].dropna(subset=["IEG_score", "DTHHRDY"] + COVS).copy()
m = ols(d["IEG_score"].to_numpy(float), design(d, ["DTHHRDY"] + COVS))
fitted = design(d, ["DTHHRDY"] + COVS) @ m["beta"]
d["adj"] = d["IEG_score"].to_numpy(float) - fitted + d["IEG_score"].mean()
tbl = d.groupby("DTHHRDY").agg(n=("adj", "size"), raw_median=("IEG_score", "median"),
                               adj_mean=("adj", "mean"))
P(tbl.round(3).to_string())

# 分类模型（S2）与二次项
dummies = pd.get_dummies(d["DTHHRDY"].astype(int), prefix="g", drop_first=True).astype(float)
Xc = np.column_stack([np.ones(len(d))] + [dummies[c].to_numpy() for c in dummies.columns]
                     + [d[c].astype(float).to_numpy() for c in COVS])
mc = ols(d["IEG_score"].to_numpy(float), Xc)
Xr = Xc[:, :Xc.shape[1]]
# 分类模型 vs 只有协变量的模型（F 检验任何等级差异）
X0 = np.column_stack([np.ones(len(d))] + [d[c].astype(float).to_numpy() for c in COVS])
m0 = ols(d["IEG_score"].to_numpy(float), X0)
F = ((m0["resid"] @ m0["resid"] - mc["resid"] @ mc["resid"]) / (Xc.shape[1] - X0.shape[1])) \
    / (mc["resid"] @ mc["resid"] / mc["dof"])
pF = stats.f.sf(F, Xc.shape[1] - X0.shape[1], mc["dof"])
P(f"\n  线性模型（序数 DTHHRDY）: beta={m['beta'][1]:+.4f} p={m['p'][1]:.3g}")
P(f"  分类模型整体 vs 仅协变量: F={F:.2f}, p={pF:.3g}（检验任何等级差异）")
# 二次项
d["D2"] = d["DTHHRDY"] ** 2
mq = ols(d["IEG_score"].to_numpy(float), design(d, ["DTHHRDY", "D2"] + COVS))
P(f"  二次项: beta={mq['beta'][2]:+.4f} p={mq['p'][2]:.3g}（显著性说明偏离线性）")
# 仅 1-4 级的线性趋势
sub = d[d["DTHHRDY"] > 0]
m14 = ols(sub["IEG_score"].to_numpy(float), design(sub, ["DTHHRDY"] + COVS))
P(f"  仅 1–4 级线性: beta={m14['beta'][1]:+.4f} p={m14['p'][1]:.3g}, n={len(sub)}")

# ---------------------------------------------------------------- S3
P("\n" + "=" * 70)
P("S3  评分与协变量的相关（BA9）")
P("=" * 70)
for c in COVS:
    pr = stats.pearsonr(d["IEG_score"], d[c])
    sr = stats.spearmanr(d["IEG_score"], d[c])
    P(f"  {c:14s} Pearson r={pr[0]:+.3f} (p={pr[1]:.3g})   "
      f"Spearman rho={sr[0]:+.3f} (p={sr[1]:.3g})")

# ---------------------------------------------------------------- S4
P("\n" + "=" * 70)
P("S4  稳健性：Huber、剔除 top1%、DFBETA")
P("=" * 70)
X = design(d, ["DTHHRDY"] + COVS)
y = d["IEG_score"].to_numpy(float)
mh = huber(y, X)
P(f"  Huber 回归      : beta={mh['beta'][1]:+.4f} p={mh['p'][1]:.3g}")
q99 = np.quantile(y, 0.99)
mask = y <= q99
mo = ols(y[mask], X[mask])
P(f"  剔除 top 1%     : beta={mo['beta'][1]:+.4f} p={mo['p'][1]:.3g}（剔除 {int((~mask).sum())} 例）")
# DFBETA
h = np.einsum("ij,jk,ik->i", X, m["XtXi"], X)
dfb = (m["XtXi"][1, :] @ X.T) * m["resid"] / (1 - h)
thr = 2 / np.sqrt(len(y))
P(f"  DFBETA 阈值 2/sqrt(n) = {thr:.4f}；最大 |DFBETA| = {np.abs(dfb).max():.4f}；"
  f"超过阈值的样本数 = {int((np.abs(dfb) > thr).sum())}")

# ---------------------------------------------------------------- S5
P("\n" + "=" * 70)
P("S5  缺失数据")
P("=" * 70)
P(f"  总样本 {len(df)}；IEG 缺失 {int(df['IEG_score'].isna().sum())}；"
  f"DTHHRDY 缺失 {int(df['DTHHRDY'].isna().sum())}")
need = ["IEG_score", "DTHHRDY"] + COVS
df["complete"] = df[need].notna().all(axis=1)
P(f"  完整 {int(df['complete'].sum())}；不完整 {int((~df['complete']).sum())}")
inc = df[~df["complete"]]
if len(inc):
    P("  不完整样本按脑区：")
    P(inc["tissue"].value_counts().to_string())
    P("  不完整样本按 DTHHRDY：")
    P(inc["DTHHRDY"].value_counts(dropna=False).to_string())
# 缺失指示模型：BA9，把 DTHHRDY 缺失的样本也纳入（加缺失指示）
b_all = df[df["tissue"] == BA9].copy()
b_all["dth_missing"] = b_all["DTHHRDY"].isna().astype(float)
b_all["DTHHRDY_f"] = b_all["DTHHRDY"].fillna(b_all["DTHHRDY"].median())
b_all = b_all.dropna(subset=["IEG_score"] + COVS)
if b_all["dth_missing"].sum() > 0:
    Xm = design(b_all, ["DTHHRDY_f", "dth_missing"] + COVS)
    mm = ols(b_all["IEG_score"].to_numpy(float), Xm)
    P(f"  缺失指示模型（BA9，n={len(b_all)}）: beta={mm['beta'][1]:+.4f} p={mm['p'][1]:.3g}")
else:
    P("  BA9 无 DTHHRDY 缺失样本，缺失指示模型不适用")

# ---------------------------------------------------------------- S6
P("\n" + "=" * 70)
P("S6  十三区合并模型：donor 聚类稳健标准误")
P("=" * 70)
p_all = df[df["complete"]].copy()
p_all["tissue_code"] = (p_all["tissue"] == BA9).astype(float)
cols = ["DTHHRDY", "tissue_code"] + COVS
Xp = design(p_all, cols)
yp = p_all["IEG_score"].to_numpy(float)
mp = ols(yp, Xp)
P(f"  常规 SE : beta={mp['beta'][1]:+.4f}  se={mp['se'][1]:.4f}  "
  f"p={mp['p'][1]:.3g}  n={len(p_all)}")
V = cluster_se(Xp, mp["resid"], p_all["SUBJID"].to_numpy(), mp["XtXi"])
se_c = np.sqrt(np.diag(V))
t_c = mp["beta"][1] / se_c[1]
p_c = 2 * stats.t.sf(abs(t_c), len(p_all) - Xp.shape[1])
n_clu = p_all["SUBJID"].nunique()
P(f"  聚类 SE : beta={mp['beta'][1]:+.4f}  se={se_c[1]:.4f}  p={p_c:.3g}  "
  f"（聚类数 {n_clu} 个 donor）")

io.open(REPORT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("已写出", REPORT)
