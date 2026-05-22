#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- substring(arg, 3L)
    bits <- strsplit(kv, "=", fixed = TRUE)[[1L]]
    key <- gsub("-", "_", bits[[1L]], fixed = TRUE)
    out[[key]] <- if (length(bits) > 1L) paste(bits[-1L], collapse = "=") else "TRUE"
  }
  out
}

args <- parse_args()
arg <- function(key, default = NULL) {
  val <- args[[key]]
  if (is.null(val) || !nzchar(val)) default else val
}

input <- arg("input", "")
if (!nzchar(input)) stop("--input is required")
input <- normalizePath(input, winslash = "/", mustWork = TRUE)

out_dir <- arg("out-dir", file.path(dirname(input), "rearranged_tables"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

raw <- utils::read.csv(input, stringsAsFactors = FALSE, check.names = FALSE)
required <- c(
  "dataset", "task_type", "method_id", "package", "function_name", "algorithm",
  "total_runtime_ms", "metric_name", "metric_value", "status", "error_message"
)
missing <- setdiff(required, names(raw))
if (length(missing)) stop("Missing columns in input: ", paste(missing, collapse = ", "))

dataset_order <- c(
  "metref", "ccle", "tcga_brca", "tcga_hnsc_methylation",
  "gtex_v8", "tcga_pan_cancer", "singlecell", "cifar100",
  "cbmc_citeseq", "prism", "nmr", "imagenet"
)
dataset_order <- c(intersect(dataset_order, unique(raw$dataset)), setdiff(unique(raw$dataset), dataset_order))

method_family <- function(method_id, algorithm, function_name) {
  method_id <- tolower(method_id)
  algorithm <- tolower(algorithm)
  function_name <- tolower(function_name)
  if (grepl("plssvd", method_id, fixed = TRUE)) return("plssvd")
  if (grepl("kernelpls|kernel_pls|kernelpls", method_id) ||
      grepl("kernel", algorithm) || grepl("kernelpls", function_name)) return("kernelpls")
  if (grepl("opls", method_id, fixed = TRUE) ||
      grepl("\\bopls\\b", algorithm) || grepl("ropls::opls", function_name, fixed = TRUE)) return("opls")
  "simpls"
}

raw$family <- mapply(method_family, raw$method_id, raw$algorithm, raw$function_name, USE.NAMES = FALSE)

short_fastpls <- function(method_id) {
  x <- sub("^fastPLS_", "", method_id)
  x <- gsub("_cpu_", " CPU ", x, fixed = TRUE)
  x <- gsub("_cuda_", " CUDA ", x, fixed = TRUE)
  x <- gsub("_irlba", " IRLBA", x, fixed = TRUE)
  x <- gsub("_rsvd", " rSVD", x, fixed = TRUE)
  x <- gsub("_cknn$", " cKNN", x)
  x <- gsub("_lda$", " LDA", x)
  x <- gsub("_", " ", x, fixed = TRUE)
  paste("fastPLS", x)
}

method_label <- function(d) {
  if (identical(d$package[[1]], "fastPLS")) return(short_fastpls(d$method_id[[1]]))
  fn <- d$function_name[[1]]
  alg <- d$algorithm[[1]]
  pkg <- d$package[[1]]
  if (is.na(fn) || !nzchar(fn)) fn <- d$method_id[[1]]
  if (is.na(alg) || !nzchar(alg)) {
    sprintf("%s: %s", pkg, fn)
  } else {
    sprintf("%s: %s (%s)", pkg, fn, alg)
  }
}

format_number <- function(x, digits = 4) {
  if (!is.finite(x) || is.na(x)) return("")
  formatC(x, digits = digits, format = "fg", flag = "#")
}

format_time <- function(ms) {
  if (!is.finite(ms) || is.na(ms)) return("")
  sec <- ms / 1000
  if (sec < 10) return(sprintf("%.3f s", sec))
  if (sec < 100) return(sprintf("%.2f s", sec))
  sprintf("%.1f s", sec)
}

format_error <- function(d) {
  status <- d$status[[1]]
  msg <- d$error_message[[1]]
  if (is.na(msg) || !nzchar(msg)) msg <- d$warning_message[[1]]
  if (is.na(msg) || !nzchar(msg)) msg <- status
  paste0(status, ": ", msg)
}

format_metric_cell <- function(d) {
  if (!identical(d$status[[1]], "ok")) return(format_error(d))
  metric <- d$metric_value[[1]]
  if (!is.finite(metric) || is.na(metric)) return("ok: metric unavailable")
  name <- d$metric_name[[1]]
  if (identical(name, "accuracy") || identical(name, "q2")) format_number(metric, 4) else format_number(metric, 6)
}

format_time_cell <- function(d) {
  if (!identical(d$status[[1]], "ok")) {
    t <- format_time(d$total_runtime_ms[[1]])
    if (nzchar(t)) return(paste0(format_error(d), " (", t, ")"))
    return(format_error(d))
  }
  format_time(d$total_runtime_ms[[1]])
}

metric_label_for_dataset <- function(d) {
  ok_metric <- d$metric_name[d$status == "ok" & nzchar(d$metric_name)]
  if (length(ok_metric)) return(ok_metric[[1]])
  if (identical(d$task_type[[1]], "classification")) return("accuracy")
  if (any(grepl("nmr|prism|cbmc", d$dataset[[1]], ignore.case = TRUE))) return("RMSD")
  "Q2/RMSD"
}

build_family_table <- function(raw, family) {
  d <- raw[raw$family == family, , drop = FALSE]
  if (!nrow(d)) return(data.frame())
  keys <- unique(d$method_id)
  rows <- list()
  idx <- 1L
  for (key in keys) {
    dk <- d[d$method_id == key, , drop = FALSE]
    label <- method_label(dk)
    metric_row <- data.frame(function_package = label, measure = "metric", stringsAsFactors = FALSE)
    time_row <- data.frame(function_package = label, measure = "time", stringsAsFactors = FALSE)
    for (ds in dataset_order) {
      cell <- dk[dk$dataset == ds, , drop = FALSE]
      if (!nrow(cell)) {
        metric_row[[ds]] <- ""
        time_row[[ds]] <- ""
        next
      }
      cell <- cell[order(cell$replicate), , drop = FALSE][1L, , drop = FALSE]
      metric_row[[ds]] <- format_metric_cell(cell)
      time_row[[ds]] <- format_time_cell(cell)
    }
    rows[[idx]] <- metric_row
    rows[[idx + 1L]] <- time_row
    idx <- idx + 2L
  }
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out
}

metric_map <- do.call(rbind, lapply(dataset_order, function(ds) {
  d <- raw[raw$dataset == ds, , drop = FALSE]
  data.frame(
    dataset = ds,
    task_type = d$task_type[[1]],
    displayed_metric = metric_label_for_dataset(d),
    n_train = d$n_train[[1]],
    n_test = d$n_test[[1]],
    p = d$p[[1]],
    n_response = d$n_response[[1]],
    ncomp_requested = d$ncomp_requested[[1]],
    stringsAsFactors = FALSE
  )
}))
utils::write.csv(metric_map, file.path(out_dir, "pipeline2_package_dataset_metric_map.csv"), row.names = FALSE, na = "")

manifest <- data.frame()
for (family in c("plssvd", "simpls", "opls", "kernelpls")) {
  tab <- build_family_table(raw, family)
  csv <- file.path(out_dir, paste0("pipeline2_", family, "_package_wide_table.csv"))
  tsv <- file.path(out_dir, paste0("pipeline2_", family, "_package_wide_table.tsv"))
  utils::write.csv(tab, csv, row.names = FALSE, quote = TRUE, na = "")
  utils::write.table(tab, tsv, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
  manifest <- rbind(
    manifest,
    data.frame(family = family, rows = nrow(tab), csv = csv, tsv = tsv, stringsAsFactors = FALSE)
  )
}
utils::write.csv(manifest, file.path(out_dir, "pipeline2_package_wide_tables_manifest.csv"), row.names = FALSE)
message("Wrote pipeline 2 package-comparison wide tables to: ", out_dir)
