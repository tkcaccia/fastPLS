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
    metref = "metref.RData",
    cbmc_citeseq = "cbmc_citeseq.RData",
    cifar100 = "CIFAR100.RData",
    ccle = "ccle.RData",
    gtex_v8 = "gtex_v8.RData",
    imagenet = "imagenet.RData",
    nmr = "nmr.RData",
    prism = "prism.RData",
    singlecell = "singlecell.RData",
    tcga_brca = "tcga_brca.RData",
    tcga_hnsc_methylation = "tcga_hnsc_methylation.RData",
    tcga_pan_cancer = "tcga_pan_cancer.RData",
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
    file.path(home_dir, "GPUPLS", "Data", fname),
    if (dataset_id == "metref") file.path(home_dir, "Documents", "GPUPLS", "Data", "metref_remote_task.RData") else "",
    if (dataset_id == "metref") "/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData" else ""
  )
  candidates <- unique(Filter(nzchar, vapply(candidates, normalize_path_if_exists, character(1))))
  for (cand in candidates) {
    if (file.exists(cand)) return(cand)
  }
  if (dataset_id == "metref") {
    return("KODAMA::MetRef")
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

half_split_idx <- function(n) {
  idx <- seq_len(n)
  train_idx <- sample(idx, size = floor(n / 2))
  test_idx <- setdiff(idx, train_idx)
  list(train = sort(train_idx), test = sort(test_idx))
}

fixed_train_split <- function(n, train_n) {
  if (n < 2L) stop("Need at least 2 rows to split train/test")
  train_n_eff <- min(max(1L, as.integer(train_n)), n - 1L)
  train_idx <- sample.int(n, size = train_n_eff)
  test_idx <- setdiff(seq_len(n), train_idx)
  list(train = sort(train_idx), test = sort(test_idx))
}

env_positive_int <- function(name, default) {
  val <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (!is.finite(val) || is.na(val) || val < 1L) default else val
}

sample_stratified_n <- function(y, n_target) {
  y <- safe_factor(y)
  n <- length(y)
  n_target <- min(max(1L, as.integer(n_target)), n)
  idx <- seq_len(n)
  by_class <- split(idx, y)
  non_empty <- by_class[vapply(by_class, length, integer(1)) > 0L]
  if (n_target <= length(non_empty)) {
    return(sort(sample(vapply(non_empty, function(ii) sample(ii, 1L), integer(1)), n_target)))
  }
  base <- vapply(non_empty, function(ii) sample(ii, 1L), integer(1))
  remaining <- setdiff(idx, base)
  extra_n <- n_target - length(base)
  extra <- if (extra_n > 0L) sample(remaining, extra_n) else integer(0)
  sort(c(base, extra))
}

sample_rows_n <- function(n, n_target) {
  n_target <- min(max(1L, as.integer(n_target)), n)
  sort(sample.int(n, size = n_target))
}

numeric_frame_to_matrix <- function(x) {
  is_plain_numeric <- vapply(
    x,
    function(v) is.numeric(v) || is.integer(v) || is.logical(v),
    logical(1)
  )
  if (all(is_plain_numeric)) {
    return(as.matrix(x))
  }
  x <- as.data.frame(lapply(x, function(v) {
    if (is.numeric(v) || is.integer(v) || is.logical(v)) {
      as.numeric(v)
    } else {
      suppressWarnings(as.numeric(as.character(v)))
    }
  }))
  as.matrix(x)
}

safe_factor <- function(y) {
  if (is.factor(y)) return(droplevels(y))
  droplevels(factor(y))
}

load_standard_task <- function(path, dataset_id, split_seed) {
  e <- new.env(parent = emptyenv())
  objs <- load(path, envir = e)
  set.seed(as.integer(split_seed))

  if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
    y_train <- get("Ytrain", envir = e)
    y_test <- get("Ytest", envir = e)
    if (is.factor(y_train)) {
      y_train <- safe_factor(y_train)
      y_test <- factor(y_test, levels = levels(y_train))
      n_classes <- nlevels(y_train)
      task_type <- "classification"
    } else {
      n_classes <- ncol(as.matrix(y_train))
      task_type <- "regression"
    }
    return(list(
      dataset = dataset_id,
      task_type = task_type,
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
      y_train <- safe_factor(y_train)
      y_test <- factor(y_test, levels = levels(y_train))
      n_classes <- nlevels(y_train)
      task_type <- "classification"
    } else {
      n_classes <- ncol(as.matrix(y_train))
      task_type <- "regression"
    }
    return(list(
      dataset = dataset_id,
      task_type = task_type,
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
    split_col <- if ("split" %in% names(dt)) trimws(tolower(as.character(dt$split))) else rep("train", nrow(dt))
    train_idx <- which(split_col == "train")
    test_idx <- which(split_col == "test")
    if (!length(train_idx) || !length(test_idx)) {
      sp <- half_split_idx(nrow(dt))
      train_idx <- sp$train
      test_idx <- sp$test
    }
    y_all <- safe_factor(dt$label_idx)
    return(list(
      dataset = dataset_id,
      task_type = "classification",
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
    y <- safe_factor(get("labels", envir = e))
    sp <- make_stratified_split(y, train_frac = 0.5)
    return(list(
      dataset = dataset_id,
      task_type = "classification",
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

  stop("Unsupported standard task format: ", path)
}

as_task <- function(path, dataset_id, split_seed = 123L) {
  dataset_id <- tolower(dataset_id)
  if (dataset_id %in% c("cifar100", "ccle", "gtex_v8", "prism", "cbmc_citeseq", "tcga_brca", "tcga_hnsc_methylation", "tcga_pan_cancer")) {
    return(load_standard_task(path, dataset_id = dataset_id, split_seed = split_seed))
  }

  if (dataset_id == "imagenet") {
    e <- new.env(parent = emptyenv())
    objs <- load(path, envir = e)
    train_n <- env_positive_int("FASTPLS_IMAGENET_TRAIN_N", 50000L)
    test_n <- env_positive_int("FASTPLS_IMAGENET_TEST_N", 10000L)
    set.seed(as.integer(split_seed))

    if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
      y_train_all <- safe_factor(e$Ytrain)
      train_idx <- sample_stratified_n(y_train_all, min(train_n, nrow(e$Xtrain)))
      test_idx <- sample_rows_n(nrow(e$Xtest), min(test_n, nrow(e$Xtest)))
      y_train <- droplevels(y_train_all[train_idx])
      y_test <- factor(e$Ytest[test_idx], levels = levels(y_train))
      task <- list(
        dataset = dataset_id,
        task_type = "classification",
        dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
        split_seed = as.integer(split_seed),
        Xtrain = as.matrix(e$Xtrain[train_idx, , drop = FALSE]),
        Ytrain = y_train,
        Xtest = as.matrix(e$Xtest[test_idx, , drop = FALSE]),
        Ytest = y_test,
        n_train = length(train_idx),
        n_test = length(test_idx),
        p = ncol(e$Xtrain),
        n_classes = nlevels(y_train)
      )
      rm(e)
      gc()
      return(task)
    }

    if ("r" %in% objs && is.data.frame(e$r) && "label_idx" %in% colnames(e$r)) {
      y <- safe_factor(e$r[, "label_idx"])
      sp <- fixed_train_split(nrow(e$r), min(train_n, nrow(e$r) - 1L))
      if (length(sp$test) > test_n) {
        sp$test <- sort(sample(sp$test, test_n))
      }
      rows <- c(sp$train, sp$test)
      feat_cols <- grep("^feat_", names(e$r), value = TRUE)
      if (!length(feat_cols)) {
        feat_cols <- setdiff(names(e$r), names(e$r)[seq_len(min(3L, ncol(e$r)))])
      }
      Xsub <- e$r[rows, feat_cols, drop = FALSE]
      X <- numeric_frame_to_matrix(Xsub)
      keep <- colSums(is.finite(X)) > 0
      X <- as.matrix(X[, keep, drop = FALSE])
      train_rows <- seq_along(sp$train)
      test_rows <- length(sp$train) + seq_along(sp$test)
      y_train <- droplevels(y[sp$train])
      task <- list(
        dataset = dataset_id,
        task_type = "classification",
        dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
        split_seed = as.integer(split_seed),
        Xtrain = X[train_rows, , drop = FALSE],
        Ytrain = y_train,
        Xtest = X[test_rows, , drop = FALSE],
        Ytest = factor(y[sp$test], levels = levels(y_train)),
        n_train = length(sp$train),
        n_test = length(sp$test),
        p = ncol(X),
        n_classes = nlevels(y_train)
      )
      rm(e, Xsub, X)
      gc()
      return(task)
    }

    if (all(c("data", "labels") %in% objs)) {
      y <- safe_factor(e$labels)
      sp <- fixed_train_split(nrow(e$data), min(train_n, nrow(e$data) - 1L))
      if (length(sp$test) > test_n) {
        sp$test <- sort(sample(sp$test, test_n))
      }
      y_train <- droplevels(y[sp$train])
      task <- list(
        dataset = dataset_id,
        task_type = "classification",
        dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
        split_seed = as.integer(split_seed),
        Xtrain = as.matrix(e$data[sp$train, , drop = FALSE]),
        Ytrain = y_train,
        Xtest = as.matrix(e$data[sp$test, , drop = FALSE]),
        Ytest = factor(y[sp$test], levels = levels(y_train)),
        n_train = length(sp$train),
        n_test = length(sp$test),
        p = ncol(e$data),
        n_classes = nlevels(y_train)
      )
      rm(e)
      gc()
      return(task)
    }

    stop("Unsupported imagenet.RData format: ", path)
  }

  if (dataset_id == "nmr") {
    e <- new.env(parent = emptyenv())
    load(path, envir = e)
    return(list(
      dataset = dataset_id,
      task_type = "regression",
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      split_seed = as.integer(split_seed),
      Xtrain = as.matrix(e$Xtrain),
      Ytrain = as.matrix(e$Ytrain),
      Xtest = as.matrix(e$Xtest),
      Ytest = as.matrix(e$Ytest),
      n_train = nrow(e$Xtrain),
      n_test = nrow(e$Xtest),
      p = ncol(e$Xtrain),
      n_classes = ncol(e$Ytrain)
    ))
  }

  if (dataset_id == "singlecell") {
    e <- new.env(parent = emptyenv())
    load(path, envir = e)
    X <- as.matrix(e$data)
    y <- safe_factor(e$labels)
    set.seed(as.integer(split_seed))
    sp <- make_stratified_split(y, train_frac = 0.5)
    return(list(
      dataset = dataset_id,
      task_type = "classification",
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

  if (dataset_id == "metref") {
    if (file.exists(path)) {
      try({
        task <- load_standard_task(path, dataset_id = dataset_id, split_seed = split_seed)
        return(task)
      }, silent = TRUE)
    }
    if (!requireNamespace("KODAMA", quietly = TRUE)) {
      stop("KODAMA package is required to load metref")
    }
    suppressPackageStartupMessages(library(KODAMA))
    data("MetRef", package = "KODAMA")
    X <- MetRef$data
    X <- X[, colSums(X) != 0, drop = FALSE]
    X <- normalization(X)$newXtrain
    y <- safe_factor(MetRef$donor)
    set.seed(as.integer(split_seed))
    ss <- sample(seq_len(nrow(X)), min(100L, floor(nrow(X) / 5L)))
    tr <- setdiff(seq_len(nrow(X)), ss)
    return(list(
      dataset = dataset_id,
      task_type = "classification",
      dataset_path = "KODAMA::MetRef",
      split_seed = as.integer(split_seed),
      Xtrain = as.matrix(X[tr, , drop = FALSE]),
      Ytrain = y[tr],
      Xtest = as.matrix(X[ss, , drop = FALSE]),
      Ytest = y[ss],
      n_train = length(tr),
      n_test = length(ss),
      p = ncol(X),
      n_classes = nlevels(y)
    ))
  }

  stop("Unsupported dataset format for ", dataset_id)
}

variant_specs <- function() {
  rows <- list(
    c("cpp_plssvd_cpu_rsvd", "plssvd", "CPU", "cpu_rsvd", "Cpp", "argmax"),
    c("cpp_plssvd_irlba", "plssvd", "CPU", "irlba", "Cpp", "argmax"),
    c("gpu_plssvd_fp64", "plssvd", "GPU", "gpu_native", "CUDA 64-bit", "argmax"),
    c("cpp_simpls_cpu_rsvd", "simpls", "CPU", "cpu_rsvd", "Cpp", "argmax"),
    c("cpp_simpls_irlba", "simpls", "CPU", "irlba", "Cpp", "argmax"),
    c("gpu_simpls_fp64", "simpls", "GPU", "gpu_native", "CUDA 64-bit", "argmax"),
    c("pls_pkg_simpls", "simpls", "CPU", "pls_pkg", "pls_pkg", "argmax"),
    c("cpp_kernelpls_cpu_rsvd", "kernelpls", "CPU", "cpu_rsvd", "Cpp", "argmax"),
    c("cpp_kernelpls_irlba", "kernelpls", "CPU", "irlba", "Cpp", "argmax"),
    c("gpu_kernelpls_fp64", "kernelpls", "GPU", "gpu_native", "CUDA 64-bit", "argmax"),
    c("pls_pkg_kernelpls", "kernelpls", "CPU", "pls_pkg", "pls_pkg", "argmax"),
    c("cpp_opls_cpu_rsvd", "opls", "CPU", "cpu_rsvd", "Cpp", "argmax"),
    c("cpp_opls_irlba", "opls", "CPU", "irlba", "Cpp", "argmax"),
    c("gpu_opls_fp64", "opls", "GPU", "gpu_native", "CUDA 64-bit", "argmax"),
    c("pls_pkg_opls", "opls", "CPU", "pls_pkg", "pls_pkg", "argmax")
  )
  out <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
  names(out) <- c("variant_name", "method_family", "engine", "backend", "implementation_label", "classifier")
  lda_rows <- out[out$implementation_label %in% c("Cpp", "CUDA 64-bit"), , drop = FALSE]
  lda_rows$variant_name <- paste0(lda_rows$variant_name, "_lda")
  lda_rows$classifier <- ifelse(lda_rows$engine == "GPU", "lda_cuda", "lda_cpp")
  out <- rbind(out, lda_rows)
  out
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
    opls = "opls",
    kernelpls = "kernelpls",
    method_family
  )
}

safe_effective_ncomp <- function(task, requested_ncomp, method_family = NULL) {
  base_cap <- min(
    as.integer(requested_ncomp),
    as.integer(task$n_train) - 1L,
    as.integer(task$p)
  )

  if (identical(method_family, "plssvd")) {
    base_cap <- min(base_cap, as.integer(task$n_classes))
  }

  max(1L, base_cap)
}

extract_pred_labels <- function(pred_res, levels_y = NULL) {
  if (is.null(pred_res$Ypred)) stop("Prediction result is missing `Ypred`")
  yp <- pred_res$Ypred
  if (is.data.frame(yp)) return(as.character(yp[[1L]]))
  if (is.factor(yp)) return(as.character(yp))
  if (is.array(yp) && length(dim(yp)) == 3L) {
    yp <- yp[, , 1L, drop = FALSE]
    yp <- matrix(yp, nrow = dim(pred_res$Ypred)[1L], ncol = dim(pred_res$Ypred)[2L])
    pred_names <- colnames(pred_res$Ypred)
    if (is.null(pred_names)) pred_names <- colnames(yp)
    if (is.null(pred_names) && !is.null(levels_y) && ncol(yp) == length(levels_y)) pred_names <- levels_y
    if (is.null(pred_names)) pred_names <- as.character(seq_len(ncol(yp)))
    return(as.character(pred_names[max.col(yp, ties.method = "first")]))
  }
  if (is.matrix(yp) && ncol(yp) >= 1L) {
    if (!is.null(levels_y) && ncol(yp) > 1L) {
      pred_names <- colnames(yp)
      if (is.null(pred_names) && ncol(yp) == length(levels_y)) pred_names <- levels_y
      if (!is.null(pred_names)) {
        return(as.character(pred_names[max.col(yp, ties.method = "first")]))
      }
    }
    return(as.character(yp[, 1L]))
  }
  if (is.list(yp) && length(yp) >= 1L) return(as.character(yp[[1L]]))
  stop("Unsupported prediction structure in `Ypred`")
}

metric_from_pred <- function(y_true, pred_obj, y_train = NULL) {
  yp <- pred_obj$Ypred
  if (is.factor(y_true)) {
    pred <- NULL
    if (is.data.frame(yp)) pred <- as.factor(yp[[1L]])
    if (is.null(pred) && is.factor(yp)) pred <- as.factor(yp)
    if (is.null(pred) && is.matrix(yp) && ncol(yp) == 1L) pred <- as.factor(yp[, 1L])
    if (is.null(pred) && is.vector(yp)) pred <- as.factor(yp)
    if (is.null(pred) && length(dim(yp)) == 3L) {
      mat <- yp[, , 1L, drop = FALSE]
      lev <- pred_obj$lev
      if (is.null(lev)) lev <- levels(y_true)
      cls <- apply(mat, 1L, which.max)
      pred <- factor(lev[cls], levels = lev)
    }
    if (is.null(pred) && is.matrix(yp) && ncol(yp) > 1L) {
      lev <- colnames(yp)
      if (is.null(lev)) lev <- levels(y_true)
      pred <- factor(lev[max.col(yp, ties.method = "first")], levels = lev)
    }
    if (is.null(pred)) stop("Cannot decode classification predictions")
    val <- mean(as.character(pred) == as.character(y_true), na.rm = TRUE)
    return(list(metric_name = "accuracy", metric_value = as.numeric(val), pred = pred))
  }

  y_num <- as.matrix(y_true)
  pred_num <- NULL
  if (length(dim(yp)) == 3L) {
    pred_num <- as.matrix(yp[, , 1L, drop = TRUE])
  } else if (is.matrix(yp)) {
    pred_num <- yp
  } else {
    pred_num <- matrix(as.numeric(yp), ncol = 1L)
  }
  if (!all(dim(pred_num) == dim(y_num))) {
    pred_num <- matrix(as.numeric(pred_num), nrow = nrow(y_num), ncol = ncol(y_num))
  }
  if (ncol(y_num) == 1L) {
    train_mean <- suppressWarnings(mean(as.numeric(as.matrix(y_train)), na.rm = TRUE))
    if (!is.finite(train_mean)) {
      train_mean <- mean(y_num[, 1L], na.rm = TRUE)
    }
    press <- sum((pred_num[, 1L] - y_num[, 1L])^2, na.rm = TRUE)
    tss <- sum((y_num[, 1L] - train_mean)^2, na.rm = TRUE)
    q2 <- if (is.finite(tss) && tss > 0) 1 - (press / tss) else NA_real_
    return(list(metric_name = "q2", metric_value = as.numeric(q2), pred = pred_num))
  }

  rmsd <- sqrt(mean((pred_num - y_num)^2, na.rm = TRUE))
  list(metric_name = "rmsd", metric_value = as.numeric(rmsd), pred = pred_num)
}

safe_accuracy <- function(truth, pred) {
  mean(as.character(truth) == as.character(pred), na.rm = TRUE)
}

write_one_row_csv <- function(row, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(row, file = path, row.names = FALSE, quote = TRUE)
}
