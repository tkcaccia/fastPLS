#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(
  if (length(args)) args[[1L]] else
    file.path("benchmark_results", "frozen_release_0.99.25"),
  mustWork = TRUE
)
out <- if (length(args) >= 2L) args[[2L]] else
  file.path(root, "provenance")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

release_version <- "0.99.25"
release_commit <- "7887401b09e25f54a546a253c255741cb1ab48e5"
archive_sha <- "604f74d72f5e9540f7378efd37f2ca6ed87ccb9e73b912211331a8cd97233481"

sha256 <- function(path) {
  command <- if (nzchar(Sys.which("shasum"))) "shasum" else "sha256sum"
  command_args <- if (identical(command, "shasum")) c("-a", "256", path) else path
  value <- system2(command, command_args, stdout = TRUE, stderr = TRUE)
  if (!length(value)) return(NA_character_)
  strsplit(trimws(value[[1L]]), "[[:space:]]+")[[1L]][[1L]]
}

relative_path <- function(path) {
  prefix <- paste0(root, .Platform$file.sep)
  if (startsWith(path, prefix)) substring(path, nchar(prefix) + 1L) else path
}

files <- list.files(root, recursive = TRUE, full.names = TRUE, all.files = TRUE)
files <- files[file.info(files)$isdir %in% FALSE]
files <- files[!grepl("(^|/)(provenance)(/|$)", files)]

ledger <- lapply(files, function(path) {
  extension <- tolower(tools::file_ext(path))
  rows <- NA_integer_
  statuses <- NA_character_
  versions <- NA_character_
  source_hashes <- NA_character_
  if (extension %in% c("csv", "tsv")) {
    separator <- if (extension == "tsv") "\t" else ","
    data <- tryCatch(
      utils::read.table(
        path, header = TRUE, sep = separator, quote = '"', comment.char = "",
        stringsAsFactors = FALSE, check.names = FALSE
      ),
      error = function(e) NULL
    )
    if (!is.null(data)) {
      rows <- nrow(data)
      if ("status" %in% names(data)) {
        statuses <- paste(sort(unique(na.omit(data$status))), collapse = ";")
      }
      version_columns <- intersect(
        c("package_version", "fastPLS_version", "version"), names(data)
      )
      if (length(version_columns)) {
        values <- unique(unlist(data[version_columns], use.names = FALSE))
        versions <- paste(sort(unique(na.omit(values[nzchar(values)]))), collapse = ";")
      }
      hash_columns <- intersect(
        c("source_archive_sha256", "fastpls_source_archive_sha256"), names(data)
      )
      if (length(hash_columns)) {
        values <- unique(unlist(data[hash_columns], use.names = FALSE))
        source_hashes <- paste(
          sort(unique(na.omit(values[nzchar(values)]))), collapse = ";"
        )
      }
    }
  }
  data.frame(
    path = relative_path(path), bytes = file.info(path)$size,
    sha256 = sha256(path), rows = rows, statuses = statuses,
    recorded_versions = versions, recorded_source_hashes = source_hashes,
    stringsAsFactors = FALSE
  )
})
ledger <- do.call(rbind, ledger)
ledger <- ledger[order(ledger$path), , drop = FALSE]
utils::write.csv(
  ledger, file.path(out, "frozen_evidence_file_ledger.csv"), row.names = FALSE,
  na = ""
)

manifest <- data.frame(
  field = c(
    "package", "version", "git_commit", "execution_archive_sha256",
    "evidence_root", "file_count", "generated_utc"
  ),
  value = c(
    "fastPLS", release_version, release_commit, archive_sha, root,
    nrow(ledger), format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
)
utils::write.table(
  manifest, file.path(out, "frozen_evidence_manifest.tsv"), sep = "\t",
  row.names = FALSE, quote = FALSE
)

cat("Audited", nrow(ledger), "files under", root, "\n")
