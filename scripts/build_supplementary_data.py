# -*- coding: utf-8 -*-
"""build_supplementary_data.py —— 生成 S1–S7 所需的全部数值。

输出 JSON + 人类可读报告，供后续排版。
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
A1 = os.path.join(ROOT, "tables")
A2 = os.path.join(ROOT, "tables")
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
WORK = os.environ.get("MDD_DE_DIR", os.path.join(ROOT, "external", "de"))
OUTDIR = os.path.join(ROOT, "tables")
os.makedirs(OUTDIR, exist_ok=True)

COVS = ["RIN", "ischemic_time", "AGE_num", "SEX_male"]
BA9 = "Brain - Frontal Cortex (BA9)"
SPINAL = "Brain - Spinal cord (cervical c-1)"

cfg = json.load(io.open(CRIT, encoding="utf-8"))
panel18 = cfg["ieg"]["panel"]

df = pd.read_csv(os.path.join(PROC, "ieg_samples.tsv"), sep="\t")
df["SEX_male"] = (df["SEX"] == "Male").astype(float)
need = ["IEG_score", "DTHHRDY"] + COVS
comp = df.dropna(subset=need).copy()


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    s2 = float(r @ r) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(b / se), n - k)
    return b, se, p, r


def design(d, cols):
    return np.column_stack([np.ones(len(d))] + [d[c].astype(float).to_numpy() for c in cols])


R = {}

# ============ S1 逐级调整均值 + 缺失分布
d = comp[comp["tissue"] == BA9].copy()
Xc = design(d, COVS)
b, *_ = ols(d["IEG_score"].to_numpy(float), Xc)
d["adj"] = d["IEG_score"].to_numpy(float) - Xc @ b + d["IEG_score"].mean()
s1 = []
for g in sorted(d["DTHHRDY"].unique()):
    sub = d[d["DTHHRDY"] == g]
    s1.append({"grade": int(g), "n": int(len(sub)),
               "raw_median": round(float(sub["IEG_score"].median()), 3),
               "adj_mean": round(float(sub["adj"].mean()), 3),
               "adj_sd": round(float(sub["adj"].std(ddof=1)), 3)})
R["S1_grade_means"] = s1

df["complete"] = df[need].notna().all(axis=1)
inc = df[~df["complete"]]
R["S1_missingness"] = {
    "total": int(len(df)), "complete": int(df["complete"].sum()),
    "excluded": int(len(inc)),
    "by_region": inc["tissue"].value_counts().to_dict(),
    "pattern": "DTHHRDY and age jointly absent; RIN and ischemic time complete",
}

# ============ S2 分类模型 + 二次项
dm = pd.get_dummies(d["DTHHRDY"].astype(int), prefix="g", drop_first=True).astype(float)
Xcat = np.column_stack([np.ones(len(d))] + [dm[c].to_numpy() for c in dm.columns]
                       + [d[c].astype(float).to_numpy() for c in COVS])
mcat = ols(d["IEG_score"].to_numpy(float), Xcat)
m0 = ols(d["IEG_score"].to_numpy(float), design(d, COVS))
X0 = design(d, COVS)
df1 = Xcat.shape[1] - X0.shape[1]          # 零模型参数个数 = 1 + 4 协变量
F = ((m0[3] @ m0[3] - mcat[3] @ mcat[3]) / df1) / (mcat[3] @ mcat[3] / (len(d) - Xcat.shape[1]))
pF = stats.f.sf(F, df1, len(d) - Xcat.shape[1])
d["D2"] = d["DTHHRDY"] ** 2
mq = ols(d["IEG_score"].to_numpy(float), design(d, ["DTHHRDY", "D2"] + COVS))
R["S2_categorical"] = {
    "F": round(float(F), 3), "p": float(pF), "df1": int(df1),
    "df2": len(d) - Xcat.shape[1],
    "contrasts": {f"grade_{g.split(chr(95))[-1]}_vs_0": {"beta": round(float(mcat[0][i + 1]), 3),
                                            "p": float(mcat[2][i + 1])}
                  for i, g in enumerate(dm.columns)},
}
R["S2_quadratic"] = {"beta": round(float(mq[0][2]), 4), "p": float(mq[2][2])}
R["S2_linear"] = {"beta": round(float(ols(d["IEG_score"].to_numpy(float),
                                          design(d, ["DTHHRDY"] + COVS))[0][1]), 4)}

# ============ S3 相关
R["S3_correlations"] = []
for c in COVS + ["DTHHRDY"]:
    pr = stats.pearsonr(d["IEG_score"], d[c])
    sr = stats.spearmanr(d["IEG_score"], d[c])
    R["S3_correlations"].append({"var": c, "pearson_r": round(float(pr[0]), 3),
                                 "pearson_p": float(pr[1]),
                                 "spearman_rho": round(float(sr[0]), 3),
                                 "spearman_p": float(sr[1])})

# ============ S4 LOO
loo = pd.read_csv(os.path.join(A1, "loo_panel.tsv"), sep="\t")
R["S4_loo"] = [{"dropped": str(r["dropped"]), "beta": round(float(r["beta"]), 3),
                "p": float(r["p"])} for _, r in loo.iterrows()]

# ============ S5 细胞类型
R["S5"] = {}
for tag, label, ctcol in (("GSE213982_female", "female", "ct"),
                          ("GSE144136_coarse", "male", "ct_coarse")):
    cells = pd.read_csv(os.path.join(SIB, f"{tag}.cells.tsv.gz"), sep="\t")
    counts = pd.read_csv(os.path.join(SIB, f"{tag}.counts.tsv.gz"), sep="\t", index_col=0)
    # 分层排除计数
    tab = cells.groupby(["sample", ctcol]).size().rename("n_cells").reset_index()
    kept = int((tab["n_cells"] >= 10).sum())
    excl = int((tab["n_cells"] < 10).sum())
    have = [g for g in panel18 if g in counts.columns]
    lib = counts.sum(axis=1).replace(0, np.nan)
    logcpm = np.log2(counts[have].div(lib, axis=0) * 1e6 + 1)
    parts = logcpm.index.to_series().str.split("|", expand=True)
    per_gene_zero = (logcpm == 0).mean(axis=0)
    R["S5"][label] = {
        "strata_kept": kept, "strata_excluded_lt10cells": excl,
        "panel_genes_available": len(have),
        "zero_expr_rate_by_gene": {g: round(float(v), 3) for g, v in per_gene_zero.items()},
    }

# ============ S6 男性队列的协变量示范
R["S6"] = {}
for label, pref, folder_map in (
        ("male", "GSE144136_male", {"none": ("de_limma", "GSE144136_male.{ct}.de.tsv"),
                                    "qc": ("de_qc", "male_qc.{ct}.de.tsv"),
                                    "qc_ieg": ("de_qc_ieg", "male_qc_ieg.{ct}.de.tsv")}),
        ("female", "GSE213982_female", {"none": ("de_limma", "GSE213982_female.{ct}.de.tsv"),
                                        "qc": ("de_qc", "female_qc.{ct}.de.tsv"),
                                        "qc_ieg": ("de_qc_ieg", "female_qc_ieg.{ct}.de.tsv")})):
    for model, (folder, pat) in folder_map.items():
        union = set()
        for ct in ("Ast", "End", "ExN", "InN", "Mic", "Mix", "Oli", "OPC"):
            fp = os.path.join(WORK, folder, pat.format(ct=ct))
            if not os.path.exists(fp):
                continue
            t = pd.read_csv(fp, sep="\t")
            if "gene" in t.columns:
                union |= set(t.loc[(t["logFC"].abs() >= 0.5) & (t["adj.P.Val"] <= 0.05), "gene"])
        R["S6"][f"{label}_{model}"] = len(union)

io.open(os.path.join(OUTDIR, "supplementary_numbers.json"), "w", encoding="utf-8").write(
    json.dumps(R, ensure_ascii=False, indent=1))

rep = []
rep.append("S1 逐级调整均值")
for x in R["S1_grade_means"]:
    rep.append(f"  grade {x['grade']}: n={x['n']}  raw_median={x['raw_median']}  adj={x['adj_mean']}")
rep.append(f"缺失：{R['S1_missingness']['excluded']}/{R['S1_missingness']['total']}")
rep.append("")
rep.append(f"S2 分类模型 F={R['S2_categorical']['F']} p={R['S2_categorical']['p']:.3g}；"
           f"二次项 beta={R['S2_quadratic']['beta']} p={R['S2_quadratic']['p']:.3g}")
rep.append("")
rep.append("S3 相关")
for x in R["S3_correlations"]:
    rep.append(f"  {x['var']:14s} r={x['pearson_r']:+.3f} (p={x['pearson_p']:.3g})  "
               f"rho={x['spearman_rho']:+.3f} (p={x['spearman_p']:.3g})")
rep.append("")
rep.append("S4 LOO 条数: %d" % len(R["S4_loo"]))
rep.append("")
rep.append("S5 分层")
for k, v in R["S5"].items():
    rep.append(f"  {k}: kept={v['strata_kept']} excluded(<10 cells)={v['strata_excluded_lt10cells']} "
               f"genes={v['panel_genes_available']}")
rep.append("")
rep.append("S6 DEG 并集")
for k, v in R["S6"].items():
    rep.append(f"  {k}: {v}")

io.open(os.path.join(OUTDIR, "supplementary_report.txt"), "w", encoding="utf-8", newline="\n").write(
    "\n".join(rep) + "\n")
print("written to", OUTDIR)

