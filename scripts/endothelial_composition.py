# -*- coding: utf-8 -*-
"""endothelial_composition.py —— P1：把组成独立性检验从「神经元中心」补到「内皮中心」。

v7 的自有数据显示，在核内层面内皮才是面板基因的高表达类群，且内皮呈现最大的
病例-对照差异。因此对供者级评分构成最大组成威胁的是内皮比例，而非神经元比例。
本脚本对内皮重复原先针对神经元的三个检验：

  (1) 内皮比例的组间比较
  (2) 供者级评分与内皮比例的秩相关
  (3) 仅用内皮分层重算评分后的组间比较

同时保留神经元口径作为对照。
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
OUT = os.path.join(ROOT, "tables", "endothelial_composition.txt")

COHORTS = [
    ("male", "GSE144136_coarse", {"Suicide"}, "ct_coarse"),
    ("female", "GSE213982_female", {"Case"}, "ct"),
]
NEURON = ["ExN", "InN"]

lines = []


def P(s=""):
    lines.append(s)


cfg = json.load(io.open(CRIT, encoding="utf-8"))
panel18 = cfg["ieg"]["panel"]


def ieg_from_counts(counts: pd.DataFrame, panel: list) -> pd.Series:
    have = [g for g in panel if g in counts.columns]
    lib = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(lib, axis=0) * 1e6
    return np.log2(cpm[have] + 1).mean(axis=1)


def mw(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return float(np.median(b) - np.median(a)), float(
        stats.mannwhitneyu(b, a, alternative="two-sided").pvalue)


for label, tag, cases, ctcol in COHORTS:
    P("=" * 74)
    P(f"cohort {label}  ({tag})")
    P("=" * 74)
    cells = pd.read_csv(os.path.join(SIB, f"{tag}.cells.tsv.gz"), sep="\t")
    counts = pd.read_csv(os.path.join(SIB, f"{tag}.counts.tsv.gz"), sep="\t", index_col=0)
    ct = ctcol if ctcol in cells.columns else "ct"
    cells = cells[["sample", "condition", ct]].rename(columns={ct: "ct"})
    cells["sample"] = cells["sample"].astype(str)
    cells["is_case"] = cells["condition"].isin(cases)

    tab = pd.crosstab(cells["sample"], cells["ct"])
    prop = tab.div(tab.sum(axis=1), axis=0)
    sample_cond = cells.groupby("sample")["is_case"].first()
    prop = prop.loc[sample_cond.index]

    # 供者级评分（全部细胞类型平均，即当前协变量口径）
    ieg_row = ieg_from_counts(counts, panel18)
    parts = pd.Series(ieg_row.index, index=ieg_row.index).str.split("|", expand=True)
    rowdf = pd.DataFrame({"sample": parts[0].values, "ct": parts[1].values,
                          "ieg": ieg_row.values})
    ieg_all = rowdf.groupby("sample")["ieg"].mean()

    # ---- (1) 比例组间比较
    P("\n(1) 细胞类型比例的病例-对照比较")
    P(f"    {'celltype':8s}{'control':>10s}{'case':>10s}{'delta':>9s}{'p':>11s}")
    for c in sorted(prop.columns):
        x = prop.loc[~sample_cond, c].to_numpy(float)
        y = prop.loc[sample_cond, c].to_numpy(float)
        d, p = mw(x, y)
        flag = "  <-- p<0.05" if np.isfinite(p) and p < 0.05 else ""
        P(f"    {c:8s}{np.median(x):>10.3f}{np.median(y):>10.3f}{d:>+9.3f}{p:>11.3g}{flag}")

    # ---- (2) 评分与各比例的相关
    P("\n(2) 供者级评分 与 各细胞类型比例 的秩相关")
    for c in sorted(prop.columns):
        common = ieg_all.index.intersection(prop.index)
        rho, p = stats.spearmanr(prop.loc[common, c], ieg_all.loc[common])
        flag = "  *" if p < 0.05 else ""
        P(f"    {c:8s} Spearman rho = {rho:+.3f}  p = {p:.3g}{flag}")

    # 神经元比例（合并 ExN+InN）与内皮比例，单独列出以便正文引用
    neu = prop[NEURON].sum(axis=1) if set(NEURON).issubset(prop.columns) else None
    end = prop["End"] if "End" in prop.columns else None
    if neu is not None:
        rho, p = stats.spearmanr(neu.loc[ieg_all.index], ieg_all)
        P(f"    neurons (ExN+InN) rho = {rho:+.3f}  p = {p:.3g}")
    if end is not None:
        rho, p = stats.spearmanr(end.loc[ieg_all.index], ieg_all)
        P(f"    endothelium      rho = {rho:+.3f}  p = {p:.3g}")

    # ---- (3) 分层重算
    P("\n(3) 仅用单一细胞类型分层重算评分后的组间比较")
    for single, name in (("End", "endothelium"), (None, "neurons only"), (None, "all classes")):
        if single is not None:
            sub = rowdf[rowdf["ct"] == single]
            per = sub.groupby("sample")["ieg"].mean()
        elif name == "neurons only":
            sub = rowdf[rowdf["ct"].isin(NEURON)]
            per = sub.groupby("sample")["ieg"].mean()
        else:
            per = ieg_all
        cond = sample_cond.reindex(per.index).astype(bool)
        d, p = mw(per[~cond], per[cond])
        P(f"    {name:16s} n_ctrl={int((~cond).sum()):2d} n_case={int(cond.sum()):2d}  "
          f"delta={d:+.3f}  p={p:.3g}")
    P("")

io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("written", OUT)

