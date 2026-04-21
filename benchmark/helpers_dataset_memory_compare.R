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

normalize_path_if_exists <- function(path) {
  path <- trimws(path)
  if (!nzchar(path)) return(path)
  path <- path.expand(path)
  if (file.exists(path)) normalizePath(path, winslash = "/", mustWork = TRUE) else path
}

dataset_filename <- function(dataset_id) {
  switch(
    tolower(dataset_id),
    cifar100 = "CIFAR100.RData",
    ccle = "ccle.RData",
    stop("Unsupported dataset_id: ", dataset_id)
  )
}

find_dataset_rdata <- function(dataset_id) {
  dataset_id <- tolower(dataset_id)
  home_dir <- path.expand("~")
  env_name <- sprintf("FASTPLS_%s_RDATA", toupper(dataset_id))
  fname <- dataset_filename(dataset_id)
  candidates <- c(
    Sys.getenv(env_name, ""),
    file.path(home_dir, "Documents", "Rdatasets", fname),
    file.path(home_dir, "Documents", "fastpls", "data", fname),
    file.path(home_dir, "GPUPLS", "Data", fname)
  )
  candidates <- unique(Filter(nzchar, vapply(candidates, normalize_path_if_exists, character(1))))
  for (cand in candidates) {
    if (file.exists(cand)) return(cand)
  }
  found <- list.files(
    home_dir,
    pattern = sprintf("^%s$", gsub(".", "\\\\.", fname, fixed = TRUE)),
    full.names = TRUE,
    recursive = TRUE,
    ignore.case = TRUE
  )
  if (length(found)) {
    found <- normalizePath(found, winslash = "/", mustWork = TRUE)
    return(found[[1L]])
  }
  stop("Dataset RData not found for ", dataset_id, " (checked ", env_name, " and common remote paths).")
}

make_stratified_split <- function(y, train_frac = 0.9) {
  y <- droplevels(as.factor(y))
  idx <- seq_along(y)
  by_class <- split(idx, y)
  train_idx <- unlist(lapply(by_class, function(ii) {
    n_train <- max(1L, floor(length(ii) * train_frac))
    sample(ii, n_train)
  }), use.names = FALSE)
  test_idx <- setdiff(idx, train_idx)
  list(train = sort(train_idx), test = sort(test_idx))
}

as_task <- function(path, dataset_id, split_seed = 123L) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  objs <- ls(e)

  set.seed(as.integer(split_seed))

  if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
    y_train <- get("Ytrain", envir = e)
    y_test <- get("Ytest", envir = e)
    if (is.factor(y_train)) {
      y_train <- droplevels(y_train)
      y_test <- factor(y_test, levels = levels(y_train))
      n_classes <- nlevels(y_train)
    } else {
      n_classes <- ncol(as.matrix(y_train))
    }
    return(list(
      dataset = dataset_id,
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      split_seed = as.integer(split_seed),
      Xtrain = as.matrix(get("Xtrain", envir = e)),
      Ytrain = y_train,
      Xtest = as.matrix(get("Xtest", envir = e)),
      Ytest = y_test,
      n_train = nrow(get("Xtrain", envir = e)),
      n_test = nrow(get("Xtest", envir = e)),
      p = ncol(get("Xtrain", envir = e)),
      n_classes = n_classes
    ))
  }

  if ("out" %in% objs && is.list(get("out", envir = e)) &&
      all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% names(get("out", envir = e)))) {
    obj <- get("out", envir = e)
    y_train <- obj$Ytrain
    y_test <- obj$Ytest
    if (is.factor(y_train)) {
      y_train <- droplevels(y_train)
      y_test <- factor(y_test, levels = levels(y_train))
      n_classes <- nlevels(y_train)
    } else {
      n_classes <- ncol(as.matrix(y_train))
    }
    return(list(
      dataset = dataset_id,
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      split_seed = as.integer(split_seed),
      Xtrain = as.matrix(obj$Xtrain),
      Ytrain = y_train,
      Xtest = as.matrix(obj$Xtest),
      Ytest = y_test,
      n_train = nrow(obj$Xtrain),
      n_test = nrow(obj$Xtest),
      p = ncol(obj$Xtrain),
      n_classes = n_classes
    ))
  }

  if ("r" %in% objs && is.data.frame(e$r) && "label_idx" %in% colnames(e$r)) {
    dt <- data.table::as.data.table(get("r", envir = e))
    feat_cols <- grep("^feat_", names(dt), value = TRUE)
    if (!length(feat_cols)) {
      feat_cols <- setdiff(names(dt), c("image_path", "split", "label_idx", "label_name"))
    }
    if ("split" %in% names(dt)) {
      train_idx <- which(dt$split == "train")
      test_idx <- which(dt$split == "test")
    } else {
      sp <- make_stratified_split(dt$label_idx, train_frac = 0.9)
      train_idx <- sp$train
      test_idx <- sp$test
    }
    y_all <- factor(dt$label_idx)
    return(list(
      dataset = dataset_id,
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      split_seed = as.integer(split_seed),
      Xtrain = as.matrix(dt[train_idx, ..feat_cols]),
      Ytrain = droplevels(y_all[train_idx]),
      Xtest = as.matrix(dt[test_idx, ..feat_cols]),
      Ytest = factor(y_all[test_idx], levels = levels(y_all[train_idx])),
      n_train = length(train_idx),
      n_test = length(test_idx),
      p = length(feat_cols),
      n_classes = nlevels(y_all[train_idx])
    ))
  }

  if (all(c("data", "labels") %in% objs)) {
    X <- as.matrix(get("data", envir = e))
    y <- droplevels(as.factor(get("labels", envir = e)))
    sp <- make_stratified_split(y, train_frac = 0.9)
    return(list(
      dataset = dataset_id,
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      split_seed = as.integer(split_seed),
      Xtrain = X[sp$train, , drop = FALSE],
      Ytrain = droplevels(y[sp$train]),
      Xtest = X[sp$test, , drop = FALSE],
      Ytest = factor(y[sp$test], levels = levels(y[sp$train])),
      n_train = length(sp$train),
      n_test = length(sp$test),
      p = ncol(X),
      n_classes = nlevels(y[sp$train])
    ))
  }

  stop("Unsupported dataset format for ", dataset_id)
}

variant_specs <- function() {
  data.frame(
    variant_name = c(
      "cpp_plssvd_cpu_rsvd",
      "r_plssvd_cpu_rsvd",
      "gpu_plssvd_fp64",
      "gpu_plssvd_fp32",
      "cpp_simpls_cpu_rsvd",
      "r_simpls_cpu_rsvd",
      "pls_pkg_simpls",
      "cpp_simpls_fast_cpu_rsvd",
      "r_simpls_fast_cpu_rsvd",
      "gpu_simpls_fast_fp64",
      "gpu_simpls_fast_fp32"
    ),
    method_family = c(
      "plssvd", "plssvd", "plssvd", "plssvd",
      "simpls", "simpls", "simpls",
      "simpls_fast", "simpls_fast", "simpls_fast", "simpls_fast"
    ),
    engine = c(
      "CPU", "CPU", "GPU", "GPU",
      "CPU", "CPU", "CPU",
      "CPU", "CPU", "GPU", "GPU"
    ),
    backend = c(
      "cpu_rsvd", "cpu_rsvd", "gpu_native", "gpu_native",
      "cpu_rsvd", "cpu_rsvd", "pls_pkg",
      "cpu_rsvd", "cpu_rsvd", "gpu_native", "gpu_native"
    ),
    implementation_label = c(
      "Cpp", "R", "CUDA 64-bit", "CUDA 32-bit",
      "Cpp", "R", "pls_pkg",
      "Cpp", "R", "CUDA 64-bit", "CUDA 32-bit"
    ),
    stringsAsFactors = FALSE
  )
}

variant_spec <- function(variant_name) {
  specs <- variant_specs()
  hit <- specs[specs$variant_name == variant_name, , drop = FALSE]
  if (!nrow(hit)) stop("Unknown variant_name: ", variant_name)
  hit[1L, , drop = FALSE]
}

method_panel_label <- function(method_family) {
  switch(
    method_family,
    plssvd = "plssvd",
    simpls = "simpls",
    simpls_fast = "simpls-fast",
    method_family
  )
}

safe_effective_ncomp <- function(task, requested_ncomp) {
  cap <- min(
    as.integer(requested_ncomp),
    as.integer(task$n_train) - 1L,
    as.integer(task$p),
    as.integer(task$n_classes)
  )
  max(1L, cap)
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

write_one_row_csv <- function(row, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(row, file = path, row.names = FALSE, quote = TRUE)
}
