# -*- coding: utf-8 -*-
"""extract_ieg_gtex.py —— 从 GTEx 全组织 TPM 矩阵中抽取 IEG 面板并算评分。

GTEx 的 gene TPM 是全组织一个文件（1.6 GB gz，约 5.6 万基因 × 1.7 万样本）。
我们只需要 IEG 面板那 18 行，按行流式读取、其余直接丢弃，内存占用可忽略。

输出：
    02_data/processed/ieg_expression.tsv   基因 × 样本（仅面板基因）
    02_data/processed/ieg_samples.tsv      样本级表：IEG 评分 + DTHHRDY + RIN + 缺血时间 + 年龄 + 性别 + 脑区

用法：
    python extract_ieg_gtex.py
    python extract_ieg_gtex.py --selftest
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CRIT = os.path.join(ROOT, "criteria", "criteria_thresholds.json")
RAW = os.environ.get("GTEX_RAW", os.path.join(ROOT, "data", "raw", "gtex"))
OUT = os.path.join(ROOT, "data", "derived")

def target_tissues() -> list:
    """脑区清单从判据读取。

    注意：不能写成模块级常量——那时 load_cfg 尚未定义，异常会被静默吞掉并回落到两个脑区，
    表面正常但实际只跑了主分析（踩过这个坑）。
    """
    try:
        with io.open(CRIT, encoding="utf-8") as fh:
            g = json.load(fh)["data"]["gtex"]
        t = g.get("region_survey") or []
        if t:
            return list(t)
        return [g["tissue"]] + ([g["replication_tissue"]] if g.get("replication_tissue") else [])
    except Exception as e:
        print(f"警告：读取脑区清单失败（{e}），回落到两个脑区", file=sys.stderr)
        return ["Brain - Frontal Cortex (BA9)", "Brain - Cortex"]


TARGET_TISSUES = None   # 运行时由 main() 填入


def load_cfg() -> dict:
    with io.open(CRIT, encoding="utf-8") as fh:
        return json.load(fh)


def read_gct_header(path: str) -> list[str]:
    """GCT 前三行是 #1.2 / 维度 / 表头。返回样本 ID 列表（去掉 Name 与 Description 两列）。"""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        fh.readline()                      # #1.2
        fh.readline()                      # n_genes \t n_samples
        header = fh.readline().rstrip("\n").split("\t")
    return header[2:]


def read_annotations() -> tuple:
    import pandas as pd
    sa = pd.read_csv(os.path.join(RAW, "GTEx_v8_SampleAttributesDS.txt"),
                     sep="\t", low_memory=False, usecols=["SAMPID", "SMRIN", "SMTSISCH", "SMTSD"])
    sp = pd.read_csv(os.path.join(RAW, "GTEx_v8_SubjectPhenotypesDS.txt"), sep="\t")
    return sa, sp


def extract(panel: list[str], tissues: list[str]) -> dict:
    tpm_path = os.path.join(RAW, "GTEx_v8_gene_tpm.gct.gz")
    if not os.path.exists(tpm_path):
        raise SystemExit(f"找不到 TPM 矩阵：{tpm_path}")

    sa, sp = read_annotations()
    sa = sa[sa["SMTSD"].isin(tissues)].copy()
    keep_samples = set(sa["SAMPID"])
    print(f"目标脑区 {tissues} → {len(sa)} 个样本")

    print("读取 GCT 表头…")
    header = read_gct_header(tpm_path)
    print(f"  矩阵共 {len(header):,} 个样本")

    want = set(panel)
    found: dict[str, dict] = {}
    t0 = time.time()
    n_line = 0
    with gzip.open(tpm_path, "rt", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        fh.readline()
        fh.readline()                      # 跳过前三行
        for line in fh:
            n_line += 1
            # GTEx GCT 的第一列是 Ensembl ID，第二列 Description 才是基因符号
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            symbol = parts[1]
            if symbol not in want or symbol in found:
                continue
            vals = {}
            for sid, v in zip(header, parts[2:]):
                if sid in keep_samples:
                    vals[sid] = float(v)
            found[symbol] = vals
            if len(found) == len(want):
                break
            if n_line % 5000 == 0:
                print(f"  已扫 {n_line:,} 行 [{time.time()-t0:.0f}s]", end="\r")
    print(f"\n  命中面板基因 {len(found)}/{len(want)}，扫描 {n_line:,} 行，用时 {time.time()-t0:.0f}s")

    missing = [g for g in panel if g not in found]
    if missing:
        print(f"  警告：矩阵中找不到 {missing}", file=sys.stderr)
    return {"expr": found, "sample_attr": sa, "subject": sp, "samples_in_matrix": len(header)}


def build_tables(res: dict, panel: list[str]) -> tuple:
    import numpy as np
    import pandas as pd

    genes = [g for g in panel if g in res["expr"]]
    if not genes:
        raise SystemExit("没有任何面板基因命中，无法继续")
    samples = sorted(res["expr"][genes[0]].keys())
    mat = pd.DataFrame(index=genes, columns=samples, dtype=float)
    for g in genes:
        mat.loc[g] = [res["expr"][g].get(s, np.nan) for s in samples]

    logtpm = np.log2(mat.astype(float) + 1)
    score = logtpm.mean(axis=0, skipna=True)
    n_present = logtpm.notna().sum(axis=0)

    meta = res["sample_attr"].set_index("SAMPID").reindex(samples)
    df = pd.DataFrame({
        "SAMPID": samples,
        "SUBJID": [s[:10] for s in samples],
        "tissue": meta["SMTSD"].values,
        "RIN": pd.to_numeric(meta["SMRIN"], errors="coerce").values,
        "ischemic_time": pd.to_numeric(meta["SMTSISCH"], errors="coerce").values,
        "IEG_score": score.values,
        "n_genes": n_present.values,
    })
    df = df.merge(res["subject"], on="SUBJID", how="left")
    df["AGE_num"] = df["AGE"].astype(str).str.split("-").str[0].astype(float) + 5
    df["SEX"] = df["SEX"].map({1: "Male", 2: "Female"})
    return mat, df


def analyze(df: pd.DataFrame) -> None:
    print("\n== 按脑区与 DTHHRDY 的样本分布 ==")
    tab = df.pivot_table(index="tissue", columns="DTHHRDY", values="IEG_score", aggfunc="count")
    print(tab.fillna(0).astype(int).to_string())
    print("\n== IEG 评分中位数 ==")
    med = df.pivot_table(index="tissue", columns="DTHHRDY", values="IEG_score", aggfunc="median")
    print(med.round(3).to_string())


def selftest() -> int:
    """用合成 GCT 验证解析与评分。"""
    import tempfile
    tmp = os.path.join(ROOT, "_selftest_gtex")
    os.makedirs(tmp, exist_ok=True)
    gct = os.path.join(tmp, "mini.gct.gz")
    panel = ["FOS", "EGR1", "JUN"]
    with gzip.open(gct, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write("#1.2\n")
        fh.write("3\t2\n")
        fh.write("Name\tDescription\tS1\tS2\n")
        fh.write("ENSG00000170345.9\tFOS\t8\t32\n")
        fh.write("ENSG00000120738.7\tEGR1\t8\t32\n")
        fh.write("ENSG00000177606.6\tJUN\t8\t32\n")
    header = read_gct_header(gct)
    assert header == ["S1", "S2"], header
    with gzip.open(gct, "rt", encoding="utf-8") as fh:
        fh.readline(); fh.readline(); fh.readline()
        rows = {}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            rows[p[1]] = p[2:]
    import math
    s1 = sum(math.log2(float(v) + 1) for v in (rows[g][0] for g in panel)) / 3
    s2 = sum(math.log2(float(v) + 1) for v in (rows[g][1] for g in panel)) / 3
    ok = abs(s1 - math.log2(9)) < 1e-9 and abs(s2 - math.log2(33)) < 1e-9
    print(f"{'OK  ' if ok else 'FAIL'} GCT 表头解析与 IEG 评分：S1={s1:.3f} S2={s2:.3f}")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("自检：", "通过" if ok else "未通过")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    global TARGET_TISSUES
    TARGET_TISSUES = target_tissues()
    cfg = load_cfg()
    panel = cfg["ieg"]["panel"]
    print(f"IEG 面板 {len(panel)} 个基因（判据：{cfg['ieg']['min_genes_present']} 个为下限）")

    res = extract(panel, TARGET_TISSUES)
    mat, df = build_tables(res, panel)

    os.makedirs(OUT, exist_ok=True)
    mat.T.to_csv(os.path.join(OUT, "ieg_expression.tsv"), sep="\t")
    df.to_csv(os.path.join(OUT, "ieg_samples.tsv"), sep="\t", index=False)
    print(f"\n写出 {os.path.join(OUT, 'ieg_expression.tsv')}（{mat.shape[0]} 基因 × {mat.shape[1]} 样本）")
    print(f"写出 {os.path.join(OUT, 'ieg_samples.tsv')}（{len(df)} 行）")

    low = df[df["n_genes"] < cfg["ieg"]["min_genes_present"]]
    if len(low):
        print(f"警告：{len(low)} 个样本命中基因数低于下限 {cfg['ieg']['min_genes_present']}", file=sys.stderr)
    analyze(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
