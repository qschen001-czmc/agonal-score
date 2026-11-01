# -*- coding: utf-8 -*-
"""results5_counts.py —— P2：Results 5 需要的三个 DEG 计数与评分-QC 主成分相关。

模型层级（均为 limma-voom、供者为重复单位、按细胞类型分别建模）：
  1 de_limma      仅分组
  2 de_qc         分组 + 批次 + 3 个质控主成分
  3 de_qc_ieg     在第 2 层基础上再加供者级 agonal 评分
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
from scipy import stats

WORK = os.environ.get("MDD_DE_DIR", os.path.join(ROOT, "external", "de"))


def _require_dir(path: str, env: str) -> None:
    """外部输入缺失时给出可执行的提示，而不是原始 FileNotFoundError。"""
    if not os.path.isdir(path):
        raise SystemExit(
            f"External input directory not found: {path}\n"
            f"Set the environment variable {env} to the directory holding the "
            f"pseudobulk matrices (or DE results) derived from GSE144136 and GSE213982."
        )

_require_dir(WORK, "MDD_DE_DIR")
COV = os.environ.get("MDD_COV_TSV", os.path.join(SIB, "GSE213982_female.samples_cov.tsv"))
OUT = os.path.join(ROOT, "tables", "results5_counts.txt")

CTS = ["Ast", "End", "ExN", "InN", "Mic", "Mix", "Oli", "OPC"]
MODELS = [
    ("unadjusted", "de_limma", "GSE213982_female.{ct}.de.tsv"),
    ("+batch+QC PCs", "de_qc", "female_qc.{ct}.de.tsv"),
    ("+agonal score", "de_qc_ieg", "female_qc_ieg.{ct}.de.tsv"),
]
THR_FROZEN = (0.5, 0.05)                     # 冻结判据 |log2FC|>=0.5, FDR<=0.05
THR_MAITRA = (np.log2(1.1), 0.05)            # 原文口径 logFC>log2(1.1), FDR<0.05

lines = []


def P(s=""):
    lines.append(s)


P("=" * 74)
P("Results 5：女性队列 GSE213982 的差异表达计数")
P("=" * 74)


def load(folder, pattern, ct):
    p = os.path.join(WORK, folder, pattern.format(ct=ct))
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, sep="\t")


def count_set(thr, models):
    res = {}
    for label, folder, pat in models:
        union, total = set(), 0
        per_ct = {}
        for ct in CTS:
            d = load(folder, pat, ct)
            if d is None or "gene" not in d.columns:
                continue
            sig = d.loc[(d["logFC"].abs() >= thr[0]) & (d["adj.P.Val"] <= thr[1]), "gene"]
            per_ct[ct] = int(len(sig))
            union |= set(sig)
            total += len(sig)
        res[label] = (len(union), total, per_ct)
    return res


for thr_name, thr in (("冻结判据 |log2FC|>=0.5, FDR<=0.05", THR_FROZEN),
                      ("原文口径 logFC>log2(1.1), FDR<0.05", THR_MAITRA)):
    P(f"\n阈值：{thr_name}")
    r = count_set(thr, MODELS)
    P(f"  {'model':16s}{'并集(去重)':>12s}{'合计(含重复)':>14s}")
    for label, _, _ in MODELS:
        u, t, _ = r[label]
        P(f"  {label:16s}{u:>12d}{t:>14d}")
    P("  分细胞类型：")
    for label, _, _ in MODELS:
        _, _, per = r[label]
        P(f"    {label:16s}" + "  ".join(f"{c}={per.get(c,0)}" for c in CTS))

# ---- 评分与 QC 主成分的相关
P("\n" + "=" * 74)
P("供者级 agonal 评分 与 质控主成分 的相关（GSE213982）")
P("=" * 74)
cov = pd.read_csv(COV, sep="\t")
P(f"  协变量表 {cov.shape[0]} 行，列：{', '.join(cov.columns[:12])}")
if "IEG_score" in cov.columns:
    for c in ("QC_PC1", "QC_PC2", "QC_PC3"):
        if c in cov.columns:
            d = cov[["IEG_score", c]].dropna()
            pr = stats.pearsonr(d["IEG_score"], d[c])
            sr = stats.spearmanr(d["IEG_score"], d[c])
            P(f"  {c}: Pearson r={pr[0]:+.3f} (p={pr[1]:.3g})   "
              f"Spearman rho={sr[0]:+.3f} (p={sr[1]:.3g})  n={len(d)}")
    # 多元：评分对三个 QC PC 回归的 R2
    cols = [c for c in ("QC_PC1", "QC_PC2", "QC_PC3") if c in cov.columns]
    d = cov[["IEG_score"] + cols].dropna()
    X = np.column_stack([np.ones(len(d))] + [d[c].astype(float).to_numpy() for c in cols])
    y = d["IEG_score"].to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    P(f"  评分 ~ {len(cols)} 个 QC 主成分：R² = {r2:.3f}（即评分与质控主成分共享方差的份额）")
else:
    P("  协变量表无 IEG_score 列")

io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("written", OUT)

