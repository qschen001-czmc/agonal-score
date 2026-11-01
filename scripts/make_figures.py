# -*- coding: utf-8 -*-
"""make_figures.py —— 生成论文插图 Fig 1–5。

所有图使用英文标注（目标期刊为英文），300 dpi，输出到 03_analysis/03_manuscript_figs/。

数据来源：
    region_survey.tsv            脑区调查（Fig 2）
    direction_results.tsv        BA9 主分析（Fig 1）
    ieg_by_celltype.*.tsv        逐细胞类型（Fig 3）
    loo_panel.tsv, panel_variants.tsv   面板稳健性（Fig 4）
    pc1_vs_panel_*.tsv           PC1 对照（Fig 5）
"""
from __future__ import annotations

import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "figures")
A1 = os.path.join(ROOT, "tables")
A2 = os.path.join(ROOT, "tables")

plt.rcParams.update({
    "font.size": 8.5, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})

C_MAIN = "#2c5f8d"
C_ALT = "#c0504d"
C_GREY = "#888888"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print(" 写出", name)


# ------------------------------------------------------------------ Fig 1
def fig1():
    d = pd.read_csv(os.path.join(A1, "region_survey.tsv"), sep="\t")
    ba9 = d[d["tissue"] == "Brain - Frontal Cortex (BA9)"].iloc[0]
    ctx = d[d["tissue"] == "Brain - Cortex"].iloc[0]

    fig = plt.figure(figsize=(9.4, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.42)

    # ---- A 效应量
    ax = fig.add_subplot(gs[0, 0])
    rows = [("BA9", ba9, C_MAIN), ("Cortex", ctx, C_GREY)]
    for i, (lab, r, col) in enumerate(rows):
        for j, (mdl, b, p) in enumerate((("unadjusted", r["beta_unadj"], r["p_unadj"]),
                                         ("full model", r["beta"], r["p"]))):
            dfree = max(3, int(r["n"]) - 6)
            tval = stats.t.isf(p / 2, dfree)
            se = abs(b) / tval if tval > 0 else np.nan
            lo, hi = b - 1.96 * se, b + 1.96 * se
            yy = i + (j - 0.5) * 0.28
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.2, alpha=1.0 if j else 0.45, zorder=2)
            ax.plot(b, yy, "o", color=col, alpha=1.0 if j else 0.45, ms=7, zorder=3)
            ax.text(hi + 0.012, yy, f"p={p:.1e}", va="center", fontsize=6.8)
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["BA9\n(n=162)", "Cortex\n(n=210)"])
    ax.set_xlim(-0.42, 0.06); ax.set_ylim(-0.7, 1.7)
    ax.set_xlabel("beta per grade", fontsize=8)
    ax.set_title("A  Effect size", loc="left", fontsize=9, fontweight="bold")

    # ---- B 评分定义
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.text(0.0, 1.0, "B  Score definition", fontsize=9, fontweight="bold", va="top")
    txt = ("score = mean log2(TPM+1) over panel genes\n\n"
           "core panel (n=6):\n    FOS, FOSB, EGR1, JUN,\n    JUNB, JUND\n\n"
           "extended panel (n=18):\n    core + EGR2, EGR3, EGR4, ATF3,\n"
           "    IER2, DUSP1, ZFP36, HSPA1A,\n    HSPA1B, HSPB1, HSPH1, DNAJB1\n\n"
           "Fixed formula; no reference cohort")
    ax.text(0.0, 0.86, txt, fontsize=6.9, va="top", linespacing=1.45)

    # ---- C 逐级调整均值
    ax = fig.add_subplot(gs[0, 2])
    s1 = _grade_means()
    gx = [x[0] for x in s1]; gy = [x[1] for x in s1]; gn = [x[2] for x in s1]
    ax.plot(gx, gy, "o-", color=C_MAIN, lw=1.6, ms=6, zorder=3)
    for x, y, n in zip(gx, gy, gn):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, -11),
                    ha="center", fontsize=6.4, color=C_GREY)
    ax.axhline(np.mean(gy), color=C_GREY, lw=0.8, ls=":", zorder=1)
    ax.set_xticks(gx)
    ax.set_xticklabels([str(g) for g in gx])
    ax.set_ylim(4.2, 6.5)
    ax.set_xlabel("death-circumstances grade", fontsize=8)
    ax.set_ylabel("adjusted mean score", fontsize=8)
    ax.set_title("C  Grade profile", loc="left", fontsize=9, fontweight="bold")

    save(fig, "Fig1_score_and_validation.png")


def _grade_means():
    """从数据现算逐级协变量调整均值，避免图与正文脱节。"""
    df = pd.read_csv(os.path.join(ROOT, "data", "derived", "ieg_samples.tsv"), sep="\t")
    df["SEX_male"] = (df["SEX"] == "Male").astype(float)
    covs = ["RIN", "ischemic_time", "AGE_num", "SEX_male"]
    d = df[df["tissue"] == "Brain - Frontal Cortex (BA9)"].dropna(
        subset=["IEG_score", "DTHHRDY"] + covs).copy()
    X = np.column_stack([np.ones(len(d))] + [d[c].astype(float).to_numpy() for c in covs])
    b, *_ = np.linalg.lstsq(X, d["IEG_score"].to_numpy(float), rcond=None)
    d["adj"] = d["IEG_score"].to_numpy(float) - X @ b + d["IEG_score"].mean()
    return [(int(g), float(sub["adj"].mean()), int(len(sub)))
            for g, sub in d.groupby("DTHHRDY")]


# ------------------------------------------------------------------ Fig 2
def fig2():
    d = pd.read_csv(os.path.join(A1, "region_survey.tsv"), sep="\t")
    d = d.sort_values("beta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(d))
    for i, r in d.iterrows():
        sig = r["p"] < 0.05
        col = C_MAIN if sig else C_GREY
        lo, hi = r["beta"] - 1.96 * r["se"], r["beta"] + 1.96 * r["se"]
        ax.plot([lo, hi], [i, i], color=col, lw=1.1, alpha=0.85, zorder=2)
        ax.plot(r["beta"], i, "o", color=col, ms=5, zorder=3)
        star = "***" if r["p"] < 1e-3 else ("**" if r["p"] < 1e-2 else ("*" if r["p"] < 0.05 else ""))
        ax.text(0.14, i, f"{star}", va="center", fontsize=8, color=col)

    labels = [t.replace("Brain - ", "").replace(" (basal ganglia)", "")
              .replace(" (cervical c-1)", "").replace(" (BA9)", " (BA9)")
              .replace("Anterior cingulate cortex (BA24)", "Ant. cingulate (BA24)")
              .replace("Nucleus accumbens", "N. accumbens")
              .replace("Cerebellar Hemisphere", "Cerebellar hem.")
              for t in d["tissue"]]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, d["n"])], fontsize=7.6)
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("β for DTHHRDY, adjusted for RIN, ischemic time, age, sex")
    ax.set_xlim(-0.45, 0.18)
    ax.set_title("Agonal score across 13 GTEx brain regions", loc="left",
                 fontsize=9.5, fontweight="bold")
    ax.text(0.02, -0.16, "*** p<0.001   ** p<0.01   * p<0.05   grey = not significant",
            transform=ax.transAxes, fontsize=7, color=C_GREY)
    save(fig, "Fig2_region_map.png")


# ------------------------------------------------------------------ Fig 3
def fig3():
    m = pd.read_csv(os.path.join(A2, "ieg_by_celltype.GSE144136_coarse.tsv"), sep="\t")
    f = pd.read_csv(os.path.join(A2, "ieg_by_celltype.GSE213982_female.tsv"), sep="\t")
    order = ["ExN", "InN", "End", "Ast", "OPC", "Oli", "Mic", "Mix"]

    def pick(df):
        rows = []
        for ct in order:
            r = df[df["definition"] == f"仅{ct}"]
            if len(r):
                rows.append((ct, float(r.iloc[0]["diff"]), float(r.iloc[0]["p"])))
        return rows

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True,
                             gridspec_kw={"wspace": 0.30})
    for ax, rows, title, col in ((axes[0], pick(f), "Female cohort", C_MAIN),
                                 (axes[1], pick(m), "Male cohort", C_ALT)):
        names = [x[0] for x in rows]
        diffs = np.array([x[1] for x in rows])
        ps = np.array([x[2] for x in rows])
        yy = np.arange(len(names))
        colors = [col if p < 0.05 else C_GREY for p in ps]
        ax.barh(yy, diffs, color=colors, alpha=0.85, height=0.62)
        # 显著性标记统一排在右侧一列，避免压住条形与坐标轴
        for i, p in enumerate(ps):
            s_ = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))
            ax.text(0.985, i, s_, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7.6,
                    color=col if p < 0.05 else C_GREY)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(yy)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Δ agonal score (case − control)")
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        lo = min(diffs.min() * 1.25, -0.1)
        # 右侧留白给显著性标记：0 在左、条形向左延伸，故 0→0.30 必须是空白
        ax.set_xlim(lo, 0.30)
    fig.suptitle("Cell-type-resolved agonal score difference", x=0.02, ha="left",
                 fontsize=9.5, fontweight="bold", y=1.03)
    save(fig, "Fig3_celltype.png")


# ------------------------------------------------------------------ Fig 4
def fig4():
    loo = pd.read_csv(os.path.join(A1, "loo_panel.tsv"), sep="\t")
    pv = pd.read_csv(os.path.join(A1, "panel_variants.tsv"), sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    loo = loo.sort_values("beta")
    x = np.arange(len(loo))
    full_beta = float(loo.loc[loo["dropped"] == "(none)", "beta"].iloc[0])
    ax.bar(x, loo["beta"], color=[C_GREY if d == "(none)" else C_MAIN for d in loo["dropped"]],
           alpha=0.85, width=0.72)
    ax.axhline(full_beta, color="k", lw=0.8, ls="--")
    # 放大 y 轴，否则 19 根柱子几乎等高、看不出稳健性
    ax.set_ylim(-0.40, -0.24)
    ax.set_xticks(x)
    ax.set_xticklabels([d if d != "(none)" else "full" for d in loo["dropped"]],
                       rotation=90, fontsize=6.8)
    ax.set_ylabel("β (DTHHRDY)")
    ax.set_title("A  Leave-one-out across the 18-gene panel", loc="left",
                 fontsize=9, fontweight="bold")

    ax = axes[1]
    pv["lab"] = ["core 6", "full 18"]
    ax.bar([0, 1], pv["beta"], color=[C_MAIN, C_GREY], alpha=0.85, width=0.55)
    for i, (b, p) in enumerate(zip(pv["beta"], pv["p"])):
        ax.text(i, b + 0.015, f"p={p:.1e}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(-0.45, 0.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["core 6", "full 18"])
    ax.set_ylabel("β (DTHHRDY)")
    ax.set_title("B  Panel size", loc="left", fontsize=9, fontweight="bold")
    save(fig, "Fig4_panel_robustness.png")


# ------------------------------------------------------------------ Fig 5
def fig5():
    full = pd.read_csv(os.path.join(A1, "pc1_vs_panel_full.tsv"), sep="\t")
    g = pd.read_csv(os.path.join(A1, "pc1_vs_panel_gradient.tsv"), sep="\t")

    labels = ["PCA-PC1", "panel 6", "panel 18"]
    cols = [C_ALT, C_MAIN, C_GREY]
    r2 = full["r2"].to_numpy(float)
    cv = (full["se"] / full["beta"].abs()).to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9),
                             gridspec_kw={"width_ratios": [1, 1, 1.35], "wspace": 0.42})

    ax = axes[0]
    ax.bar(range(3), r2, color=cols, alpha=0.85, width=0.6)
    for i, v in enumerate(r2):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("R²"); ax.set_ylim(0, 0.40)
    ax.set_title("A  Explained variance", loc="left", fontsize=9, fontweight="bold")

    ax = axes[1]
    ax.bar(range(3), cv, color=cols, alpha=0.85, width=0.6)
    for i, v in enumerate(cv):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("SE / |β|"); ax.set_ylim(0, 0.40)
    ax.set_title("B  Relative uncertainty", loc="left", fontsize=9, fontweight="bold")

    ax = axes[2]
    for score, col, lab in (("PCA-PC1", C_ALT, "PCA-PC1"), ("panel6", C_MAIN, "panel 6")):
        sub = g[g["score"] == score].sort_values("n")
        ax.plot(sub["n"].to_numpy(), sub["pct_sig"].to_numpy(), "o-", color=col,
                lw=1.5, ms=5, label=lab)
    ax.set_xlabel("samples per subsample")
    ax.set_ylabel("% subsamples with p < 0.05")
    ax.set_ylim(0, 105)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")
    ax.set_title("C  Stability vs sample size", loc="left", fontsize=9, fontweight="bold")
    ax.text(0.98, 0.90, "100 subsamples per size", transform=ax.transAxes,
            fontsize=6.6, ha="right", color=C_GREY)

    save(fig, "Fig5_pc1_vs_panel.png")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    print("生成插图：")
    for fn in (fig1, fig2, fig3, fig4, fig5):
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__} 失败：{type(e).__name__} {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
