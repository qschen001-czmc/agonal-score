# -*- coding: utf-8 -*-
"""给脑区调查补 BH 校正，并整理成正文可用的表。"""
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = os.path.join(ROOT, "tables", "region_survey.tsv")


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


d = pd.read_csv(P, sep="\t")
d = d.sort_values("beta")
d["q_bh"] = bh(d["p"].values)
d["ci_lo"] = d["beta"] - 1.96 * d["se"]
d["ci_hi"] = d["beta"] + 1.96 * d["se"]
d["sig_raw"] = d["p"] < 0.05
d["sig_bh"] = d["q_bh"] < 0.05
d.to_csv(os.path.join(ROOT, "tables", "region_survey_bh.tsv"), sep="\t", index=False)

print(f"{'region':44s}{'n':>5}{'beta':>9}{'p':>11}{'q_BH':>11}  raw/BH")
for r in d.itertuples():
    mark = ("*" if r.sig_raw else " ") + ("*" if r.sig_bh else " ")
    print(f"{r.tissue[:44]:44s}{r.n:>5}{r.beta:>9.3f}{r.p:>11.2g}{r.q_bh:>11.2g}   {mark}")
print()
print(f"原始 p<0.05: {int(d['sig_raw'].sum())}/{len(d)}"
      f"｜BH 后 q<0.05: {int(d['sig_bh'].sum())}/{len(d)}")
