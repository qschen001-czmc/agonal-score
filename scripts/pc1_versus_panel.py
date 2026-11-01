# -*- coding: utf-8 -*-
"""pc1_versus_panel.py —— PCA 式评分（Li 2007 ASR 思路）与固定面板评分的对照。

核心问题：既有做法用队列内 PCA 第一主成分作濒死评分。它在**原队列**里可能很有效，
但换一批样本就必须重做、符号与尺度会变、小样本下不稳定。

本脚本在同一份 GTEx BA9 数据上同时构建两种评分，并做**样本量梯度**比较：

    PCA 式：每个子样本内重新做 PCA 取 PC1（模拟真实使用场景）
    固定面板：6 或 18 个基因的均值，公式不随样本变化

判据：谁在样本量下降时效应估计更稳、显著性保持得更好、方向更一致。

用法：
    python pc1_versus_panel.py --build-cache     # 首次：从 TPM 矩阵抽 BA9 表达并缓存
    python pc1_versus_panel.py                    # 正式分析
    python pc1_versus_panel.py --selftest
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CRIT = os.path.join(ROOT, "criteria", "criteria_thresholds.json")
RAW = os.environ.get("GTEX_RAW", os.path.join(ROOT, "data", "raw", "gtex"))
PROC = os.path.join(ROOT, "data", "derived")
CACHE = os.path.join(PROC, "ba9_expression.npz")
OUT = os.path.join(ROOT, "tables")

TOP_N_GENES = 5000
COVS = ["RIN", "ischemic_time", "AGE_num", "SEX_male"]


def load_cfg() -> dict:
    with io.open(CRIT, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- 数据缓存
def build_cache(tissue: str) -> None:
    """从全组织 TPM 矩阵抽取指定脑区，缓存为 npz（基因 × 样本）。"""
    tpm = os.path.join(RAW, "GTEx_v8_gene_tpm.gct.gz")
    if not os.path.exists(tpm):
        raise SystemExit("找不到 GTEx TPM 矩阵")

    sa = pd.read_csv(os.path.join(RAW, "GTEx_v8_SampleAttributesDS.txt"),
                     sep="\t", low_memory=False, usecols=["SAMPID", "SMTSD"])
    keep = set(sa.loc[sa["SMTSD"] == tissue, "SAMPID"])
    print(f"目标脑区 {tissue}：{len(keep)} 个样本")

    t0 = time.time()
    with gzip.open(tpm, "rt", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
        samples = header[2:]
        idx = [i for i, s in enumerate(samples) if s in keep]
        keep_samples = [samples[i] for i in idx]
        idx_arr = np.array(idx)

        syms, mat = [], []
        n = 0
        for line in fh:
            n += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            vals = np.array(parts[2:], dtype=np.float32)[idx_arr]
            syms.append(parts[1])
            mat.append(vals)
            if n % 10000 == 0:
                print(f"  已扫 {n:,} 行 [{time.time()-t0:.0f}s]", end="\r")
    print(f"\n  扫描完成：{n:,} 行，{len(keep_samples)} 个样本，用时 {time.time()-t0:.0f}s")

    M = np.vstack(mat)
    print(f"  矩阵 {M.shape}，占用 {M.nbytes/1e6:.0f} MB")
    np.savez_compressed(CACHE, expr=M, genes=np.array(syms), samples=np.array(keep_samples))
    print(f"  已缓存 -> {CACHE}")


def load_cache() -> tuple:
    z = np.load(CACHE, allow_pickle=True)
    return z["expr"], list(z["genes"]), list(z["samples"])


# ---------------------------------------------------------------- 评分
def panel_score(expr: np.ndarray, genes: list, samples: list, panel: list[str]) -> np.ndarray:
    """固定面板评分：log2(TPM+1) 后在面板基因上取均值。"""
    gi = {g: i for i, g in enumerate(genes)}
    rows = [gi[g] for g in panel if g in gi]
    if not rows:
        raise SystemExit("面板基因不在缓存中")
    logtpm = np.log2(expr[rows, :].astype(np.float64) + 1)
    return logtpm.mean(axis=0)


def pc1_score(expr: np.ndarray, genes: list, top_n: int = TOP_N_GENES,
              sample_idx: np.ndarray | None = None,
              exclude: set | None = None) -> np.ndarray:
    """PCA 式评分：高变基因标准化后取第一主成分。

    exclude：选高变基因时排除的基因符号集合。剔除面板基因是为了避免两种评分
    在构造上机械重叠——否则 PC1 会直接包含被测面板，比较失去意义。

    注意：PC1 的符号是任意的，使用者必须外部定向——这本身就是 PCA 式评分的局限之一。
    """
    M = expr if sample_idx is None else expr[:, sample_idx]
    logtpm = np.log2(M.astype(np.float64) + 1)
    # 取变异最大的基因（先剔除面板基因）
    keep = np.ones(len(genes), dtype=bool)
    if exclude:
        keep = np.array([g not in exclude for g in genes])
    mad = np.median(np.abs(logtpm - np.median(logtpm, axis=1, keepdims=True)), axis=1)
    mad = np.where(keep, mad, -np.inf)
    order = np.argsort(mad)[::-1][:top_n]
    X = logtpm[order, :]
    X = X[np.isfinite(X).all(axis=1)]
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, ddof=0, keepdims=True) + 1e-9)
    # 样本 × 基因 做 SVD
    U, S, Vt = np.linalg.svd(X.T, full_matrices=False)
    return U[:, 0] * S[0]


def orient(score: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把 PC1 的符号对齐到参考评分（只翻转符号，不改变数值结构）。"""
    r = np.corrcoef(score, ref)[0, 1]
    return -score if r < 0 else score


# ---------------------------------------------------------------- 模型
def fit(y: np.ndarray, df: pd.DataFrame) -> dict:
    X = np.column_stack([np.ones(len(df))] +
                        [df["DTHHRDY"].to_numpy(float)] +
                        [df[c].to_numpy(float) for c in COVS])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {"beta": float(beta[1]), "se": float(se[1]), "p": float(p[1]), "r2": float(r2), "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=100)
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    cfg = load_cfg()
    tissue = cfg["data"]["gtex"]["tissue"]
    panel18 = cfg["ieg"]["panel"]
    panel6 = cfg["ieg"]["core_panel"]

    if a.build_cache or not os.path.exists(CACHE):
        build_cache(tissue)

    expr, genes, samples = load_cache()
    print(f"缓存：{expr.shape[0]:,} 基因 × {expr.shape[1]} 样本（{tissue}）")

    meta = pd.read_csv(os.path.join(PROC, "ieg_samples.tsv"), sep="\t")
    meta = meta[meta["tissue"] == tissue].set_index("SAMPID").reindex(samples)
    meta["SEX_male"] = (meta["SEX"] == "Male").astype(float)
    meta = meta.dropna(subset=["IEG_score", "DTHHRDY", "RIN", "ischemic_time", "AGE_num", "SEX_male"])
    ok = meta.index.tolist()
    pos = {s: i for i, s in enumerate(samples)}
    keep_idx = np.array([pos[s] for s in ok])
    E = expr[:, keep_idx]
    meta = meta.reset_index(drop=True)
    print(f"纳入模型：{len(meta)} 例\n")

    s_panel6 = panel_score(E, genes, ok, panel6)
    s_panel18 = panel_score(E, genes, ok, panel18)
    EXCL = set(panel18) | set(panel6)
    s_pc1 = orient(pc1_score(E, genes, exclude=EXCL), s_panel18)

    rows = []
    for name, s in (("PCA-PC1（Li 2007 思路）", s_pc1),
                    ("固定面板 6 基因", s_panel6),
                    ("固定面板 18 基因", s_panel18)):
        m = fit(s.astype(float), meta)
        rho, rp = stats.spearmanr(meta["DTHHRDY"], s)
        rows.append({"score": name, **m, "rho": float(rho), "rho_p": float(rp)})
        print(f"  {name:24s} β={m['beta']:+.4f} p={m['p']:.3g} R^2={m['r2']:.3f} "
              f"| ρ={rho:+.3f}")

    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(OUT, "pc1_vs_panel_full.tsv"), sep="\t", index=False)

    # ---------- 样本量梯度 ----------
    print(f"\n样本量梯度（每档 {a.bootstrap} 次重复抽样）")
    rng = np.random.default_rng(20261101)
    sizes = [160, 80, 40, 20]
    grad = []
    ref_beta_pc1 = rows[0]["beta"]
    ref_beta_p6 = rows[1]["beta"]
    for n in sizes:
        if n >= len(meta):
            continue
        for kind in ("pc1", "panel6"):
            betas, ps = [], []
            for b in range(a.bootstrap):
                sel = rng.choice(len(meta), size=n, replace=False)
                sub_meta = meta.iloc[sel].reset_index(drop=True)
                sub_expr = E[:, sel]
                if kind == "pc1":
                    s = orient(pc1_score(sub_expr, genes, exclude=EXCL), s_panel18[sel])
                else:
                    s = panel_score(sub_expr, genes, None, panel6)
                try:
                    m = fit(s.astype(float), sub_meta)
                except Exception:
                    continue
                betas.append(m["beta"])
                ps.append(m["p"])
            betas = np.array(betas)
            ps = np.array(ps)
            ref = ref_beta_pc1 if kind == "pc1" else ref_beta_p6
            grad.append({
                "n": n, "score": "PCA-PC1" if kind == "pc1" else "panel6",
                "n_boot": len(betas),
                "beta_mean": float(betas.mean()), "beta_sd": float(betas.std(ddof=1)),
                "beta_cv": float(abs(betas.std(ddof=1) / betas.mean())) if betas.mean() != 0 else np.nan,
                "pct_sign_ok": float(np.mean(np.sign(betas) == np.sign(ref)) * 100),
                "pct_sig": float(np.mean(ps < 0.05) * 100),
            })
            g = grad[-1]
            print(f"  n={n:3d} {g['score']:9s} β={g['beta_mean']:+.3f} "
                  f"SD={g['beta_sd']:.3f} 方向一致={g['pct_sign_ok']:5.1f}% "
                  f"显著率={g['pct_sig']:5.1f}%")
    pd.DataFrame(grad).to_csv(os.path.join(OUT, "pc1_vs_panel_gradient.tsv"), sep="\t", index=False)
    print(f"\n写出 {OUT}\\pc1_vs_panel_full.tsv、pc1_vs_panel_gradient.tsv")
    return 0


def selftest() -> int:
    rng = np.random.default_rng(0)
    n_genes, n_samp = 200, 40
    E = rng.gamma(2, 2, size=(n_genes, n_samp)).astype(np.float32)
    genes = [f"G{i}" for i in range(n_genes)] + ["FOS", "EGR1"]
    E = np.vstack([E, rng.gamma(2, 5, size=(2, n_samp))]).astype(np.float32)
    s1 = panel_score(E, genes, None, ["FOS", "EGR1"])
    ok1 = s1.shape == (n_samp,) and np.isfinite(s1).all()
    print(f"{'OK  ' if ok1 else 'FAIL'} 面板评分形状 {s1.shape}")
    s2 = pc1_score(E, genes, top_n=100)
    ok2 = s2.shape == (n_samp,) and np.isfinite(s2).all()
    print(f"{'OK  ' if ok2 else 'FAIL'} PC1 评分形状 {s2.shape}")
    o = orient(s2, s1)
    ok3 = np.corrcoef(o, s1)[0, 1] > 0
    print(f"{'OK  ' if ok3 else 'FAIL'} 定向后与面板正相关：{np.corrcoef(o, s1)[0,1]:+.3f}")
    ok = ok1 and ok2 and ok3
    print("自检：", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
