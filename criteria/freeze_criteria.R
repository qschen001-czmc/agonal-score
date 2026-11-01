# freeze_criteria.R —— 校验 criteria_freeze.yaml 并生成机器可读镜像
#
# 设计沿用兄弟项目（mdd-tcm-screen）的经验：
#   * 阈值只有一处人类可读来源（YAML），脚本读机器镜像（JSON）；
#   * 缺键即报错，绝不回落到代码里的默认值；
#   * 镜像里记录源文件 MD5，便于核对是否被改过。
#
# 用法：Rscript freeze_criteria.R

suppressMessages({
  library(yaml)
  library(jsonlite)
})

here <- dirname(normalizePath(sub("^--file=", "",
          grep("^--file=", commandArgs(FALSE), value = TRUE)[1]), mustWork = FALSE))
yml <- file.path(here, "criteria_freeze.yaml")
out <- file.path(here, "criteria_thresholds.json")

if (!file.exists(yml)) stop("找不到 ", yml)
cfg <- yaml::read_yaml(yml)

# ---- 必需键 ----
need <- list(
  c("data", "gtex", "version"),
  c("data", "gtex", "tissue"),
  c("data", "gtex", "base_url"),
  c("data", "gtex", "files", "gene_tpm"),
  c("ieg", "panel"),
  c("ieg", "core_panel"),
  c("ieg", "min_genes_present"),
  c("validation", "outcome"),
  c("validation", "primary_test", "alpha"),
  c("validation", "covariates"),
  c("validation", "pass_rule")
)
for (k in need) {
  cur <- cfg
  for (p in k) {
    if (is.null(cur[[p]])) stop("缺少必需键：", paste(k, collapse = "."))
    cur <- cur[[p]]
  }
}

# ---- 一致性检查 ----
if (length(cfg$ieg$panel) < cfg$ieg$min_genes_present) {
  stop("IEG 面板基因数(", length(cfg$ieg$panel), ") 少于 min_genes_present(",
       cfg$ieg$min_genes_present, ")")
}
if (!all(cfg$ieg$core_panel %in% cfg$ieg$panel)) {
  stop("core_panel 中有基因不在 panel 内：",
       paste(setdiff(cfg$ieg$core_panel, cfg$ieg$panel), collapse = ", "))
}
if (!"RIN" %in% cfg$validation$covariates) {
  warning("协变量中没有 RIN——这是死后脑研究最关键的变量，请确认")
}

# ---- 输出镜像：整份 cfg 原样带出，不挑段 ----
# 早先版本的教训：只导出部分段落，其余在镜像里静默丢失，脚本读不到就回落默认值。
result <- c(
  list(
    frozen_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    source_yaml = basename(yml),
    source_md5 = as.character(tools::md5sum(yml))
  ),
  cfg
)
writeLines(jsonlite::toJSON(result, auto_unbox = TRUE, pretty = TRUE, digits = NA), out)

cat("已生成：", out, "\n")
cat("  IEG 面板   :", length(cfg$ieg$panel), "个基因（核心", length(cfg$ieg$core_panel), "个）\n")
cat("  结局变量   :", cfg$validation$outcome, "\n")
cat("  协变量     :", paste(cfg$validation$covariates, collapse = ", "), "\n")
cat("  组织       :", cfg$data$gtex$tissue, "（", cfg$data$gtex$version, "）\n", sep = "")
