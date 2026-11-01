# -*- coding: utf-8 -*-
"""修正并重算 S1（调整均值）与 S5（缺失指示模型）。"""
import io
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "tables", "remaining_stats_fix.txt")

COVS = ["RIN", "ischemic_time", "AGE_num", "SEX_male"]
BA9 = "Brain - Frontal Cortex (BA9)"
lines = []


def P(s=""):
    lines.append(s)


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    n, k = X.shape
    dof = n - k
    s2 = float(r @ r) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    return dict(beta=beta, se=se, p=p, dof=dof, resid=r)


def design(df, cols):
    return np.column_stack([np.ones(len(df))] + [df[c].astype(float).to_numpy() for c in cols])


df = pd.read_csv(os.path.join(PROC, "ieg_samples.tsv"), sep="\t")
df["SEX_male"] = (df["SEX"] == "Male").astype(float)

# ---------- S1 修正：只对协变量做调整，保留组效应 ----------
P("=" * 70)
P("S1（修正）逐级调整后均值 —— 仅对四个协变量做调整")
P("=" * 70)
d = df[df["tissue"] == BA9].dropna(subset=["IEG_score", "DTHHRDY"] + COVS).copy()
Xcov = design(d, COVS)                      # 只含协变量，不含 DTHHRDY
mcov = ols(d["IEG_score"].to_numpy(float), Xcov)
d["adj"] = d["IEG_score"].to_numpy(float) - Xcov @ mcov["beta"] + d["IEG_score"].mean()
tbl = d.groupby("DTHHRDY").agg(n=("adj", "size"), raw_median=("IEG_score", "median"),
                               adj_mean=("adj", "mean"))
P(tbl.round(3).to_string())
seq = [round(tbl.loc[k, "adj_mean"], 3) for k in sorted(tbl.index)]
P(f"\n  调整后均值序列 (0→4): {seq}")
mono14 = all(tbl.loc[a, "adj_mean"] > tbl.loc[b, "adj_mean"]
             for a, b in zip([1, 2, 3], [2, 3, 4]) if a in tbl.index and b in tbl.index)
P(f"  1→4 严格单调下降: {mono14}")
P(f"  2 级 − 1 级 = {tbl.loc[2, 'adj_mean'] - tbl.loc[1, 'adj_mean']:+.3f}")
P(f"  0 级 − 2 级 = {tbl.loc[0, 'adj_mean'] - tbl.loc[2, 'adj_mean']:+.3f}")

# 用调整后均值做加权趋势检验（权重 = n）
x = tbl.index.to_numpy(float)
w = tbl["n"].to_numpy(float)
yv = tbl["adj_mean"].to_numpy(float)
sw = w.sum()
xm = (w * x).sum() / sw
ym = (w * yv).sum() / sw
sxx = (w * (x - xm) ** 2).sum()
sxy = (w * (x - xm) * (yv - ym)).sum()
slope = sxy / sxx
ss_res = (w * (yv - (ym + slope * (x - xm))) ** 2).sum()
ss_tot = (w * (yv - ym) ** 2).sum()
P(f"\n  组均值加权线性斜率 = {slope:+.4f}/级（R2={1-ss_res/ss_tot:.3f}）")

# 组间差异是否显著（1 级 vs 2 级，Mann-Whitney，全协变量调整后的残差）
r1 = d.loc[d["DTHHRDY"] == 1, "adj"]
r2 = d.loc[d["DTHHRDY"] == 2, "adj"]
P(f"  1 级 vs 2 级（调整后，Mann-Whitney）：n={len(r1)} vs {len(r2)}，"
  f"p={stats.mannwhitneyu(r1, r2, alternative='two-sided').pvalue:.3g}")

# ---------- S5 修正：缺失指示模型 ----------
P("\n" + "=" * 70)
P("S5（修正）缺失数据与缺失指示模型")
P("=" * 70)
b = df[df["tissue"] == BA9].copy()
P(f"  BA9 总样本 {len(b)}；DTHHRDY 缺失 {int(b['DTHHRDY'].isna().sum())}")
b["dth_missing"] = b["DTHHRDY"].isna().astype(float)
b["DTHHRDY_f"] = b["DTHHRDY"]
b = b.dropna(subset=["IEG_score"] + COVS)
P(f"  IEG 与协变量齐全者 {len(b)}；其中 DTHHRDY 缺失 {int(b['dth_missing'].sum())}")
if b["dth_missing"].sum() > 0:
    med = b.loc[b["dth_missing"] == 0, "DTHHRDY_f"].median()
    b["DTHHRDY_f"] = b["DTHHRDY_f"].fillna(med)
    Xm = design(b, ["DTHHRDY_f", "dth_missing"] + COVS)
    mm = ols(b["IEG_score"].to_numpy(float), Xm)
    P(f"  缺失指示模型：beta(DTHHRDY)={mm['beta'][1]:+.4f} p={mm['p'][1]:.3g}；"
      f"beta(缺失指示)={mm['beta'][2]:+.4f} p={mm['p'][2]:.3g}")
    P(f"  对照：完整病例模型 beta={mcov['beta'][0]:+.4f}（协变量部分，不可直接比）")
    # 与完整病例主模型对照
    Xfull = design(d, ["DTHHRDY"] + COVS)
    mf = ols(d["IEG_score"].to_numpy(float), Xfull)
    P(f"  完整病例主模型 beta(DTHHRDY)={mf['beta'][1]:+.4f} p={mf['p'][1]:.3g}  "
      f"→ 与缺失指示模型{'一致' if abs(mf['beta'][1] - mm['beta'][1]) < 0.05 else '不一致'}")
else:
    P("  BA9 无 DTHHRDY 缺失，模型不适用")

io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("已写出", OUT)
