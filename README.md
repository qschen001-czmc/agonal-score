# A formula-portable agonal score for post-mortem brain transcriptomics

Code and fixed resources for the manuscript *"A formula-portable, externally validated
immediate-early-gene score for agonal state"* (submitted to GigaScience).

The archive contains the frozen gene panels, the version-controlled criteria file that fixes
every threshold, all analysis scripts, the derived tables and the figures. It does not
redistribute the primary data, which are public.

## What the score is

```
score = mean over panel genes of log2(TPM + 1)
```

Two panels are provided and were fixed before any association analysis:

| File | Genes |
|---|---|
| `panels/core_6genes.txt` | FOS, FOSB, EGR1, JUN, JUNB, JUND |
| `panels/extended_18genes.txt` | core plus EGR2, EGR3, EGR4, ATF3, IER2, DUSP1, ZFP36, HSPA1A, HSPA1B, HSPB1, HSPH1, DNAJB1 |

A **lower** score corresponds to a slower, more prolonged death process. The ordinal
coefficient summarises a grade profile that is **not monotonic at the fast end**; always
report the grade-wise means alongside the coefficient (Supplementary Table S1).

## Frozen criteria

`criteria/criteria_freeze.yaml` is the single human-editable source of truth: panels, model
specification, region list and every threshold. `criteria/freeze_criteria.R` compiles it to
`criteria/criteria_thresholds.json`, a machine-readable mirror that also records the MD5 of
the source. **No script carries its own constants**; each reads the mirror and fails loudly
if a required key is missing.

Verify the checksum before and after any change:

```
python -c "import json,hashlib;d=json.load(open('criteria/criteria_thresholds.json'));\
print(hashlib.md5(open('criteria/criteria_freeze.yaml','rb').read()).hexdigest()==d['source_md5'])"
```

The YAML carries comments in Chinese, the working language of the project; the field names
and values are English.

## Environment

Python 3.8.5 with numpy 1.24.4, pandas 2.0.3, scipy 1.5.4, matplotlib 3.3.3.
Pseudo-bulk differential expression used limma 3.62.2 under R 4.4.1 with voom weights.

```
pip install -r requirements.txt
```

### Inputs you must supply

Three environment variables point at data that is either large or not redistributed here.
Nothing else needs configuring.

| Variable | Points to | Needed by |
|---|---|---|
| `GTEX_RAW` | directory holding `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz`, `GTEx_v8_SubjectPhenotypesDS.txt`, `GTEx_v8_SampleAttributesDS.txt` | `extract_ieg_gtex.py`, `pc1_versus_panel.py` |
| `MDD_PSEUDOBULK_DIR` | directory holding `GSE144136_coarse` and `GSE213982_female` pseudobulk matrices (`.counts.tsv.gz`, `.cells.tsv.gz`) | the cell-type and composition scripts |
| `MDD_DE_DIR` | directory holding the limma-voom model outputs for the two cohorts | `results5_counts.py` |

Because `data/derived/` already contains the GTEx sample table and the panel expression
matrix, **most analyses run without downloading the 1.6 GB GTEx matrix**.

## Data sources

| Data | Accession | Notes |
|---|---|---|
| GTEx v8 gene TPM | GCS bucket `adult-gtex` | open; controlled route is dbGaP phs000424 |
| GTEx subject phenotypes | same bucket | holds DTHHRDY |
| GTEx sample attributes | same bucket | holds SMRIN, SMTSISCH |
| GSE144136 | GEO | 34 male donors, dlPFC snRNA-seq |
| GSE213982 | GEO | 38 female donors, dlPFC snRNA-seq |

## Layout

```
panels/        fixed gene panels
criteria/      frozen criteria, compiled mirror, MD5
data/derived/  GTEx sample table and panel expression matrix (pipeline runs without GTEx)
scripts/       analysis code
tables/        derived tables backing the manuscript
figures/       Figures 1-5, 300 dpi
```

## Reproducing the analysis

Run in this order from the repository root. Steps 2-4 need only `data/derived/`; step 1
additionally needs `GTEX_RAW`; step 6 needs the pseudobulk inputs; step 7 needs the DE outputs.

1. `scripts/extract_ieg_gtex.py` — pulls panel genes from the GTEx matrix (needs `GTEX_RAW`).
2. `scripts/estimate_direction.py` — primary model in BA9 and cortex.
3. `scripts/region_survey.py` — thirteen-region survey, leave-one-out, panel variants.
4. `scripts/add_bh.py` — Benjamini-Hochberg correction across regions.
5. `scripts/pc1_versus_panel.py` — cohort-derived principal component versus the fixed panel,
   with the sample-size gradient (seed 20261101).
6. `scripts/composition_vs_intrinsic.py`, `scripts/endothelial_composition.py` — cell-type and
   composition analyses.
7. `scripts/results5_counts.py` — nested differential-expression model counts.
8. `scripts/compute_remaining.py`, `scripts/fix_s1_s5.py`, `scripts/fill_inserts.py` —
   remaining statistics reported in the supplement.
9. `scripts/build_supplementary_data.py` — every value in Supplementary Tables S1-S8.
10. `scripts/make_figures.py` — Figures 1-5 at 300 dpi.

## Outputs committed

`tables/` holds the thirteen-region survey with and without correction, the leave-one-out
fits, the panel-variant comparison, the principal-component comparison and its sample-size
gradient, and the cell-type results. They are the exact files used to build the manuscript,
so a reader can check any number without re-running anything.

## Citation

See `CITATION.cff`. Please cite the manuscript and this archive.

## Licence

Code: MIT (see `LICENSE`). Fixed panels and derived tables: CC BY 4.0.
GTEx and GEO data are not redistributed here and remain under their original terms.
