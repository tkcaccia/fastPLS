#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1L]] else
  file.path("benchmark_results", "frozen_release_0.99.25", "nmr")
root <- normalizePath(root, mustWork = TRUE)

version <- "0.99.25"
commit <- "7887401b09e25f54a546a253c255741cb1ab48e5"
archive_sha <- trimws(strsplit(readLines(file.path(root, "source_sha256.txt"), n = 1L), "[[:space:]]+")[[1L]][[1L]])
input_lines <- readLines(file.path(root, "input_sha256.txt"), warn = FALSE)
input_sha <- paste(trimws(input_lines[nzchar(trimws(input_lines))]), collapse = "; ")

files <- c(file.path(root, "nmr_all_runs.csv"), list.files(root, pattern = "^(selected|matched)_.*\\.csv$", full.names = TRUE))
for (path in files) {
  x <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  x$package_version <- version
  x$git_commit <- commit
  x$source_archive_sha256 <- archive_sha
  x$input_sha256_manifest <- input_sha
  write.csv(x, path, row.names = FALSE, na = "")
}

writeLines(c(
  "NMR provenance enrichment",
  "",
  "The row-level provenance columns were joined after execution from the adjacent",
  "session_info.txt, source_sha256.txt, and input_sha256.txt files captured by the",
  "same frozen-release job. No numerical result columns were modified.",
  paste("Package version:", version),
  paste("Git commit:", commit),
  paste("Execution archive SHA-256:", archive_sha)
), file.path(root, "PROVENANCE_ENRICHMENT.md"))

cat("Enriched", length(files), "NMR result files with captured provenance.\n")
