# -*- coding: utf-8 -*-
"""fill_inserts.py —— 计算 v6.3 中三个 ⟦INSERT⟧ 标记对应的实际数值。

  Results 1: 评分分布形态
  Results 3 / S7: 脊髓与其他脑区的 RIN、缺血时间分布
  Results 4: 内皮 vs 兴奋性神经元 的面板基因表达（伪批量索引为 样本|细胞类型）
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
CRIT = os.path.join(ROOT, "criteria", "criteria_thresholds.json")
SIB = os.environ.get("MDD_PSEUDOBULK_DIR", os.path.join(ROOT, "external", "pseudobulk"))


def _require_dir(path: str, env: str) -> None:
    """外部输入缺失时给出可执行的提示，而不是原始 FileNotFoundError。"""
    if not os.path.isdir(path):
        raise SystemExit(
            f"External input directory not found: {path}\n"
            f"Set the environment variable {env} to the directory holding the "
            f"pseudobulk matrices (or DE results) derived from GSE144136 and GSE213982."
        )

_require_dir(SIB, "MDD_PSEUDOBULK_DIR")
OUT = os.path.join(ROOT, "tables", "inserts_filled.txt")

BA9 = "Brain - Frontal Cortex (BA9)"
SPINAL = "Brain - Spinal cord (cervical c-1)"
COVS = ["RIN", "ischemic_time", "AGE_num", "SEX_male"]

lines = []


def P(s=""):
    lines.append(s)


cfg = json.load(io.open(CRIT, encoding="utf-8"))
panel18 = cfg["ieg"]["panel"]

df = pd.read_csv(os.path.join(PROC, "ieg_samples.tsv"), sep="\t")
df["SEX_male"] = (df["SEX"] == "Male").astype(float)
comp = df.dropna(subset=["IEG_score", "DTHHRDY"] + COVS).copy()

# ---------------------------------------------------------------- 1
P("=" * 72)
P("1  Results 1：评分分布形态（n=%d）" % len(comp))
P("=" * 72)
for q in (0.5, 1, 5, 25, 50, 75, 95, 99, 99.5):
    P(f"    {q:5.1f}%  {np.percentile(comp['IEG_score'], q):.2f}")
P(f"    均值 {comp['IEG_score'].mean():.2f}   SD {comp['IEG_score'].std():.2f}")

low = comp[comp["IEG_score"] < 3.0]
high = comp[comp["IEG_score"] > 9.0]
P(f"\n  <3.0 的样本 {len(low)} 例（{len(low)/len(comp)*100:.1f}%），分级 "
  f"{low['DTHHRDY'].value_counts().sort_index().to_dict()}")
P(f"  >9.0 的样本 {len(high)} 例（{len(high)/len(comp)*100:.1f}%），分级 "
  f"{high['DTHHRDY'].value_counts().sort_index().to_dict()}")
if len(high):
    i = high["IEG_score"].idxmax()
    P(f"  最高 {high['IEG_score'].max():.2f}（{high.loc[i,'tissue']}，"
      f"分级 {high.loc[i,'DTHHRDY']:.0f}）")
P(f"  BA9 内最低 {comp.loc[comp['tissue']==BA9,'IEG_score'].min():.2f}，"
  f"最高 {comp.loc[comp['tissue']==BA9,'IEG_score'].max():.2f}")

# ---------------------------------------------------------------- 2
P("\n" + "=" * 72)
P("2  Results 3 / S7：脊髓 vs 其他脑区")
P("=" * 72)
sp = comp[comp["tissue"] == SPINAL]
rest = comp[comp["tissue"] != SPINAL]
ba9 = comp[comp["tissue"] == BA9]
for lab, d in (("Spinal cord", sp), ("Other 12 regions", rest), ("BA9", ba9)):
    r, i = d["RIN"], d["ischemic_time"]
    P(f"  {lab:18s} n={len(d):4d}  RIN {r.median():.2f} "
      f"[{r.quantile(.25):.1f}-{r.quantile(.75):.1f}]   "
      f"ischaemic {i.median():.1f} [{i.quantile(.25):.1f}-{i.quantile(.75):.1f}]")
for v in ("RIN", "ischemic_time", "AGE_num"):
    u = stats.mannwhitneyu(sp[v], rest[v], alternative="two-sided")
    P(f"    {v:14s} spinal vs rest  p = {u.pvalue:.3g}")
P("  grade composition spinal: " + str(sp["DTHHRDY"].value_counts().sort_index().to_dict()))
P("  grade composition rest  : " + str(rest["DTHHRDY"].value_counts().sort_index().to_dict()))

# ---------------------------------------------------------------- 3
P("\n" + "=" * 72)
P("3  Results 4：内皮 vs 兴奋性神经元 的面板基因表达（细关系伪批量）")
P("=" * 72)
for label, tag in (("Female", "GSE213982_female"), ("Male", "GSE144136_coarse")):
    counts = pd.read_csv(os.path.join(SIB, f"{tag}.counts.tsv.gz"), sep="\t", index_col=0)
    have = [g for g in panel18 if g in counts.columns]
    lib = counts.sum(axis=1).replace(0, np.nan)
    logcpm = np.log2(counts[have].div(lib, axis=0) * 1e6 + 1)
    parts = logcpm.index.to_series().str.split("|", expand=True)
    logcpm = logcpm.copy()
    logcpm["_sample"] = parts[0].to_numpy()
    logcpm["_ct"] = parts[1].to_numpy()
    P(f"\n  {label} cohort {tag}：面板基因 {len(have)}/18 可用")
    stats_by_ct = {}
    for ct in ("End", "ExN"):
        sub = logcpm[logcpm["_ct"] == ct][have]
        if not len(sub):
            P(f"    {ct}: 无数据")
            continue
        per_donor = sub.mean(axis=1)
        det = (sub > 0).mean(axis=0)
        stats_by_ct[ct] = per_donor
        P(f"    {ct:4s} 供体×类型 n={len(sub):2d}  面板均值 log2CPM 中位 "
          f"{per_donor.median():.2f} IQR [{per_donor.quantile(.25):.2f}-"
          f"{per_donor.quantile(.75):.2f}]  检出率中位 {det.median()*100:.0f}%")
    if "End" in stats_by_ct and "ExN" in stats_by_ct:
        u = stats.mannwhitneyu(stats_by_ct["End"], stats_by_ct["ExN"], alternative="two-sided")
        P(f"    End vs ExN：Mann-Whitney p = {u.pvalue:.3g}")

io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("written", OUT)

