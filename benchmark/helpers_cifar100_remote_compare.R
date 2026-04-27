parse_kv_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  if (!length(args)) return(out)
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    keyval <- substring(arg, 3L)
    if (grepl("=", keyval, fixed = TRUE)) {
      bits <- strsplit(keyval, "=", fixed = TRUE)[[1L]]
      key <- gsub("-", "_", bits[[1L]], fixed = TRUE)
      val <- paste(bits[-1L], collapse = "=")
    } else {
      key <- gsub("-", "_", keyval, fixed = TRUE)
      val <- "true"
    }
    out[[key]] <- val
  }
  out
}

arg_value <- function(args, key, default = NULL, required = FALSE) {
  val <- args[[key]]
  if (is.null(val) || identical(val, "")) {
    if (isTRUE(required)) stop("Missing required argument --", gsub("_", "-", key))
    return(default)
  }
  val
}

arg_flag <- function(args, key, default = FALSE) {
  val <- arg_value(args, key, default = NULL, required = FALSE)
  if (is.null(val)) return(isTRUE(default))
  tolower(trimws(as.character(val))) %in% c("1", "true", "yes", "y", "on")
}

normalize_path_if_exists <- function(path) {
  path <- trimws(path)
  if (!nzchar(path)) return(path)
  path <- path.expand(path)
  if (file.exists(path)) normalizePath(path, winslash = "/", mustWork = TRUE) else path
}

find_cifar100_rdata <- function() {
  home_dir <- path.expand("~")
  candidates <- c(
    Sys.getenv("FASTPLS_CIFAR100_RDATA", ""),
    file.path(home_dir, "Documents", "Rdatasets", "CIFAR100.RData"),
    file.path(home_dir, "GPUPLS", "Data", "CIFAR100.RData")
  )
  candidates <- unique(Filter(nzchar, vapply(candidates, normalize_path_if_exists, character(1))))
  for (cand in candidates) {
    if (file.exists(cand)) return(cand)
  }
  found <- list.files(
    home_dir,
    pattern = "^CIFAR100.*\\.RData$",
    full.names = TRUE,
    recursive = TRUE,
    ignore.case = TRUE
  )
  if (length(found)) {
    found <- normalizePath(found, winslash = "/", mustWork = TRUE)
    return(found[[1L]])
  }
  stop(
    "CIFAR100 RData file not found. Checked FASTPLS_CIFAR100_RDATA, ~/Documents/Rdatasets/CIFAR100.RData, ",
    "~/GPUPLS/Data/CIFAR100.RData, and a recursive search under the remote home directory."
  )
}

load_cifar100_task <- function(path, split_seed = 123L) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  if (!exists("r", envir = e, inherits = FALSE)) {
    stop("Object `r` not found in ", path)
  }
  r_obj <- get("r", envir = e, inherits = FALSE)
  if (!is.data.frame(r_obj)) {
    stop("Object `r` in ", path, " is not a data.frame")
  }
  if (!("label_idx" %in% colnames(r_obj))) {
    stop("Column `label_idx` not found in object `r` from ", path)
  }
  if (ncol(r_obj) < 4L) {
    stop("Expected at least 4 columns in object `r`; found ", ncol(r_obj))
  }

  y_all <- factor(r_obj[, "label_idx"])
  x_raw <- r_obj[, -(1:3), drop = FALSE]
  x_num <- data.frame(lapply(x_raw, function(col) suppressWarnings(as.numeric(col))), check.names = FALSE)
  keep <- vapply(x_num, function(col) any(is.finite(col)), logical(1))
  x_num <- x_num[, keep, drop = FALSE]
  if (!ncol(x_num)) stop("No usable numeric predictor columns remained after coercion for ", path)
  X_all <- as.matrix(x_num)
  storage.mode(X_all) <- "double"
  X_all[!is.finite(X_all)] <- 0

  n_all <- nrow(X_all)
  set.seed(as.integer(split_seed))
  perm <- sample.int(n_all)
  n_train <- floor(n_all / 2)
  train_idx <- sort(perm[seq_len(n_train)])
  test_idx <- sort(perm[-seq_len(n_train)])

  y_levels <- levels(y_all)
  Ytrain <- factor(y_all[train_idx], levels = y_levels)
  Ytest <- factor(y_all[test_idx], levels = y_levels)

  list(
    dataset = "cifar100",
    dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
    split_seed = as.integer(split_seed),
    train_idx = train_idx,
    test_idx = test_idx,
    Xtrain = X_all[train_idx, , drop = FALSE],
    Ytrain = Ytrain,
    Xtest = X_all[test_idx, , drop = FALSE],
    Ytest = Ytest,
    n_train = length(train_idx),
    n_test = length(test_idx),
    p = ncol(X_all),
    n_classes = nlevels(y_all)
  )
}

variant_specs <- function(include_optional_cpu = TRUE, include_test_cpu = TRUE) {
  rows <- list(
    data.frame(
      variant_name = c("baseline_gpu_plssvd", "baseline_gpu_simpls_fast"),
      code_tree = "baseline",
      method_family = c("plssvd", "simpls_fast"),
      engine = "GPU",
      backend = "gpu_native",
      precision_mode = "default",
      label_mode = "default",
      stringsAsFactors = FALSE
    ),
    data.frame(
      variant_name = c("baseline_cpu_plssvd_cpu_rsvd", "baseline_cpu_simpls_fast_cpu_rsvd"),
      code_tree = "baseline",
      method_family = c("plssvd", "simpls_fast"),
      engine = "CPU",
      backend = "cpu_rsvd",
      precision_mode = "default",
      label_mode = "default",
      stringsAsFactors = FALSE
    ),
    data.frame(
      variant_name = c("test_gpu_plssvd", "test_gpu_simpls_fast"),
      code_tree = "test",
      method_family = c("plssvd", "simpls_fast"),
      engine = "GPU",
      backend = "gpu_native",
      precision_mode = "default",
      label_mode = "default",
      stringsAsFactors = FALSE
    ),
    data.frame(
      variant_name = c(
        "test_gpu_plssvd_host_qr",
        "test_gpu_simpls_fast_host_qr",
        "test_gpu_plssvd_qless",
        "test_gpu_simpls_fast_qless"
      ),
      code_tree = "test",
      method_family = c("plssvd", "simpls_fast", "plssvd", "simpls_fast"),
      engine = "GPU",
      backend = "gpu_native",
      precision_mode = c("host_qr_eig", "host_qr_eig", "qless_host", "qless_host"),
      label_mode = "default",
      stringsAsFactors = FALSE
    )
  )

  if (isTRUE(include_optional_cpu)) {
    rows[[length(rows) + 1L]] <- data.frame(
      variant_name = c("baseline_cpu_plssvd_irlba", "baseline_cpu_simpls_fast_irlba"),
      code_tree = "baseline",
      method_family = c("plssvd", "simpls_fast"),
      engine = "CPU",
      backend = "irlba",
      precision_mode = "default",
      label_mode = "default",
      stringsAsFactors = FALSE
    )
  }

  if (isTRUE(include_test_cpu)) {
    rows[[length(rows) + 1L]] <- data.frame(
      variant_name = c("test_cpu_plssvd_cpu_rsvd", "test_cpu_simpls_fast_cpu_rsvd"),
      code_tree = "test",
      method_family = c("plssvd", "simpls_fast"),
      engine = "CPU",
      backend = "cpu_rsvd",
      precision_mode = "default",
      label_mode = "default",
      stringsAsFactors = FALSE
    )
    if (isTRUE(include_optional_cpu)) {
      rows[[length(rows) + 1L]] <- data.frame(
        variant_name = c("test_cpu_plssvd_irlba", "test_cpu_simpls_fast_irlba"),
        code_tree = "test",
        method_family = c("plssvd", "simpls_fast"),
        engine = "CPU",
        backend = "irlba",
        precision_mode = "default",
        label_mode = "default",
        stringsAsFactors = FALSE
      )
    }
  }

  do.call(rbind, rows)
}

variant_spec <- function(variant_name, include_optional_cpu = TRUE, include_test_cpu = TRUE) {
  specs <- variant_specs(include_optional_cpu = include_optional_cpu, include_test_cpu = include_test_cpu)
  hit <- specs[specs$variant_name == variant_name, , drop = FALSE]
  if (!nrow(hit)) stop("Unknown variant_name: ", variant_name)
  hit[1L, , drop = FALSE]
}

reference_variant_name <- function(method_family, engine, backend) {
  if (identical(engine, "GPU")) {
    return(if (identical(method_family, "plssvd")) "baseline_gpu_plssvd" else "baseline_gpu_simpls_fast")
  }
  if (identical(backend, "irlba")) {
    return(if (identical(method_family, "plssvd")) "baseline_cpu_plssvd_irlba" else "baseline_cpu_simpls_fast_irlba")
  }
  if (identical(engine, "CPU")) {
    return(if (identical(method_family, "plssvd")) "baseline_cpu_plssvd_cpu_rsvd" else "baseline_cpu_simpls_fast_cpu_rsvd")
  }
  NA_character_
}

extract_pred_labels <- function(pred_res) {
  if (is.null(pred_res$Ypred)) stop("Prediction result is missing `Ypred`")
  yp <- pred_res$Ypred
  if (is.data.frame(yp)) return(as.character(yp[[1L]]))
  if (is.factor(yp)) return(as.character(yp))
  if (is.matrix(yp) && ncol(yp) >= 1L) return(as.character(yp[, 1L]))
  if (is.list(yp) && length(yp) >= 1L) return(as.character(yp[[1L]]))
  stop("Unsupported prediction structure in `Ypred`")
}

safe_accuracy <- function(truth, pred) {
  mean(as.character(truth) == as.character(pred), na.rm = TRUE)
}

safe_effective_ncomp <- function(model, requested_ncomp, fallback_cap = NA_integer_) {
  if (!is.null(model$ncomp)) {
    vals <- suppressWarnings(as.integer(model$ncomp))
    vals <- vals[is.finite(vals) & !is.na(vals)]
    if (length(vals)) return(max(vals))
  }
  if (is.finite(fallback_cap) && !is.na(fallback_cap)) return(as.integer(fallback_cap))
  as.integer(requested_ncomp)
}

write_one_row_csv <- function(row, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(row, file = path, row.names = FALSE, quote = TRUE)
}

read_prediction_file <- function(path) {
  if (!file.exists(path)) return(NULL)
  readRDS(path)
}

safe_system_output <- function(cmd) {
  out <- tryCatch(system(cmd, intern = TRUE, ignore.stderr = TRUE), error = function(e) character())
  paste(out, collapse = "\n")
}
