# -*- coding: utf-8 -*-
"""composition_vs_intrinsic.py —— 检验「IEG 下降」是细胞组成变化还是细胞内在变化。

这是审阅意见中最致命的一条：18 个 IEG 都是神经元富集的活动依赖基因。
如果病例组神经元比例下降，样本级 IEG 评分会**机械下降**，与濒死状态无关。

做四件事：
  1. 各细胞类型比例的病例-对照比较；
  2. 样本级 IEG 评分（当前协变量）与神经元比例的相关；
  3. 只用神经元（ExN/InN）重算 IEG，看组间差异是否还在；
  4. 逐细胞类型分别比较，看差异是否被某一类驱动。

数据来自兄弟项目 mdd-tcm-screen（伪批量与逐细胞元数据）。
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
OUT = os.path.join(ROOT, "tables")

COHORTS = [
    ("男", "GSE144136_coarse", {"Suicide"}, "ct_coarse"),
    ("女", "GSE213982_female", {"Case"}, "ct"),
]
NEURON = ["ExN", "InN"]


def load_cfg() -> dict:
    with io.open(CRIT, encoding="utf-8") as fh:
        return json.load(fh)


def celltype_col(df: pd.DataFrame, prefer: str) -> str:
    for c in (prefer, "ct_coarse", "ct"):
        if c in df.columns:
            return c
    raise SystemExit(f"找不到细胞类型列：{list(df.columns)}")


def ieg_from_counts(counts: pd.DataFrame, panel: list[str]) -> pd.Series:
    """每个 (样本|细胞类型) 行算 IEG 评分：组内 CPM 后取 log2，再对面板基因求均值。"""
    have = [g for g in panel if g in counts.columns]
    if not have:
        raise SystemExit("伪批量矩阵中没有面板基因")
    lib = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(lib, axis=0) * 1e6
    logcpm = np.log2(cpm + 1)
    return logcpm[have].mean(axis=1)


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """返回 (中位差 b-a, Mann-Whitney p, Cohen's d)。"""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan
    u = stats.mannwhitneyu(b, a, alternative="two-sided")
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (np.mean(b) - np.mean(a)) / sd if sd > 0 else np.nan
    return float(np.median(b) - np.median(a)), float(u.pvalue), float(d)


def run_cohort(label: str, tag: str, cases: set, ctcol: str, panel: list[str]) -> dict:
    print(f"\n===== {label}性队列：{tag} =====")
    cells = pd.read_csv(os.path.join(SIB, f"{tag}.cells.tsv.gz"), sep="\t")
    counts = pd.read_csv(os.path.join(SIB, f"{tag}.counts.tsv.gz"), sep="\t", index_col=0)

    # 只保留一列细胞类型：优先用指定的那列，避免同时读进 ct 与 ct_coarse 造成重名
    ct = ctcol if ctcol in cells.columns else celltype_col(cells, ctcol)
    cells = cells[["sample", "condition", ct]].rename(columns={ct: "ct"})
    # 男性样本号是整数、女性是 F# 字符串，计数行索引是字符串，统一成字符串否则对不上
    cells["sample"] = cells["sample"].astype(str)
    cells["is_case"] = cells["condition"].isin(cases)

    # ---- 1. 各细胞类型比例 ----
    tab = pd.crosstab(cells["sample"], cells["ct"])
    prop = tab.div(tab.sum(axis=1), axis=0)
    sample_cond = cells.groupby("sample")["is_case"].first()
    prop = prop.loc[sample_cond.index]

    rows = []
    for c in sorted(prop.columns):
        x = prop.loc[~sample_cond, c].to_numpy(float)
        y = prop.loc[sample_cond, c].to_numpy(float)
        diff, p, d = compare(x, y)
        rows.append({"celltype": c, "median_control": float(np.median(x)),
                     "median_case": float(np.median(y)),
                     "diff": diff, "p": p, "cohen_d": d})
    comp = pd.DataFrame(rows)
    print("\n-- 细胞类型比例（病例 vs 对照）--")
    for r in comp.itertuples():
        flag = "  <-- 显著" if r.p < 0.05 else ""
        print(f"   {r.celltype:6s} 对照 {r.median_control:.3f} → 病例 {r.median_case:.3f}"
              f"  Δ={r.diff:+.3f}  d={r.cohen_d:+.2f}  p={r.p:.3g}{flag}")

    # ---- 2. 样本级 IEG 与神经元比例的关系 ----
    ieg_row = ieg_from_counts(counts, panel)
    parts = pd.Series(ieg_row.index, index=ieg_row.index).str.split("|", expand=True)
    ieg_row = pd.DataFrame({"sample": parts[0].values, "ct": parts[1].values,
                            "ieg": ieg_row.values})
    ieg_all = ieg_row.groupby("sample")["ieg"].mean()        # 当前协变量的口径
    neuron_frac = prop[NEURON].sum(axis=1) if set(NEURON).issubset(prop.columns) else None
    if neuron_frac is not None:
        common = ieg_all.index.intersection(neuron_frac.index)
        rho, p_rho = stats.spearmanr(neuron_frac.loc[common], ieg_all.loc[common])
        print(f"\n-- 样本级 IEG 与神经元比例：Spearman rho={rho:+.3f} (p={p_rho:.3g}) --")

    # ---- 3. 三种口径下的组间差异 ----
    res = {}

    def summ(name, series):
        x = series[~sample_cond.reindex(series.index).astype(bool)]
        y = series[sample_cond.reindex(series.index).astype(bool)]
        diff, p, d = compare(x.to_numpy(float), y.to_numpy(float))
        res[name] = {"n_control": int(x.notna().sum()), "n_case": int(y.notna().sum()),
                     "median_control": float(np.median(x)), "median_case": float(np.median(y)),
                     "diff": diff, "p": p, "cohen_d": d}
        flag = "  <-- 显著" if p < 0.05 else ""
        print(f"   {name:16s} 对照 {np.median(x):.3f} → 病例 {np.median(y):.3f}"
              f"  Δ={diff:+.3f}  d={d:+.2f}  p={p:.3g}{flag}")

    print("\n-- 三种口径的 IEG 组间差异 --")
    summ("全细胞类型均值", ieg_all)

    if set(NEURON).issubset(set(ieg_row["ct"])):
        neu = ieg_row[ieg_row["ct"].isin(NEURON)].groupby("sample")["ieg"].mean()
        summ("仅神经元", neu)

    for c in sorted(ieg_row["ct"].unique()):
        s = ieg_row[ieg_row["ct"] == c].set_index("sample")["ieg"]
        summ(f"仅{c}", s)

    comp.to_csv(os.path.join(OUT, f"composition.{tag}.tsv"), sep="\t", index=False)
    # 逐细胞类型的 IEG 结果也落盘，供出图使用
    ieg_rows = [{"cohort": label, "tag": tag, "definition": k, **v} for k, v in res.items()]
    pd.DataFrame(ieg_rows).to_csv(os.path.join(OUT, f"ieg_by_celltype.{tag}.tsv"),
                                  sep="\t", index=False)
    return {"cohort": label, "tag": tag, "composition": comp, "ieg": res,
            "neuron_ieq_rho": (rho, p_rho) if neuron_frac is not None else None,
            "ieg_all": ieg_all, "sample_cond": sample_cond}



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    cfg = load_cfg()
    panel = cfg["ieg"]["panel"]
    os.makedirs(OUT, exist_ok=True)

    out = {}
    for label, tag, cases, ctcol in COHORTS:
        out[tag] = run_cohort(label, tag, cases, ctcol, panel)

    # 结论判定
    print("\n" + "=" * 66)
    print("判定：神经元专用 IEG 是否仍显示组间差异？")
    for tag, r in out.items():
        if "仅神经元" in r["ieg"]:
            v = r["ieg"]["仅神经元"]
            verdict = "仍显著" if v["p"] < 0.05 else "不再显著"
            print(f"  {tag}: Δ={v['diff']:+.3f}, p={v['p']:.3g} → {verdict}")
        if r["neuron_ieq_rho"]:
            rho, p = r["neuron_ieq_rho"]
            print(f"    样本级 IEG 与神经元比例 rho={rho:+.3f} (p={p:.3g})")
    return 0


def selftest() -> int:
    rng = np.random.default_rng(0)
    n = 40
    frames = []
    for i in range(n):
        sample = f"S{i}"
        is_case = i % 2 == 1
        # 每个样本由不同数量的细胞构成：病例组 ExN 少一半 → 比例下降
        for ct, k in (("ExN", 15 if is_case else 30), ("Ast", 10), ("Mic", 5)):
            for _ in range(k):
                frames.append({"sample": sample, "ct": ct,
                               "condition": "Case" if is_case else "Control"})
    cells = pd.DataFrame(frames)
    tab = pd.crosstab(cells["sample"], cells["ct"])
    prop = tab.div(tab.sum(axis=1), axis=0)
    sc = cells.groupby("sample")["condition"].first()
    exn_c = prop.loc[sc == "Control", "ExN"]
    exn_k = prop.loc[sc == "Case", "ExN"]
    diff, p, d = compare(exn_c.to_numpy(), exn_k.to_numpy())
    ok1 = diff < 0 and p < 0.01
    print(f"{'OK  ' if ok1 else 'FAIL'} 合成数据里病例组 ExN 比例更低：Δ={diff:+.3f} p={p:.3g}")

    panel = ["FOS", "EGR1", "JUN"]
    counts = pd.DataFrame(rng.poisson(100, size=(6, len(panel) + 2)),
                          index=[f"S{i}|ExN" for i in range(3)] + [f"S{i}|Ast" for i in range(3)],
                          columns=panel + ["X", "Y"])
    s = ieg_from_counts(counts, panel)
    ok2 = len(s) == 6 and np.isfinite(s).all()
    print(f"{'OK  ' if ok2 else 'FAIL'} IEG 逐组计算返回 {len(s)} 个值")
    ok = ok1 and ok2
    print("自检：", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
