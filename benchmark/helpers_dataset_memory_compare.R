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
    train_n <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_IMAGENET_TRAIN_N", "1000000")))
    if (!is.finite(train_n) || is.na(train_n) || train_n < 1L) train_n <- 1000000L
    set.seed(as.integer(split_seed))

    if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
      Xall <- rbind(as.matrix(e$Xtrain), as.matrix(e$Xtest))
      yall <- safe_factor(c(as.character(e$Ytrain), as.character(e$Ytest)))
      sp <- fixed_train_split(nrow(Xall), train_n)
      return(list(
        dataset = dataset_id,
        task_type = "classification",
        dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
        split_seed = as.integer(split_seed),
        Xtrain = Xall[sp$train, , drop = FALSE],
        Ytrain = droplevels(yall[sp$train]),
        Xtest = Xall[sp$test, , drop = FALSE],
        Ytest = factor(yall[sp$test], levels = levels(yall[sp$train])),
        n_train = length(sp$train),
        n_test = length(sp$test),
        p = ncol(Xall),
        n_classes = nlevels(yall[sp$train])
      ))
    }

    if ("r" %in% objs && is.data.frame(e$r) && "label_idx" %in% colnames(e$r)) {
      X <- e$r[, -c(1:3), drop = FALSE]
      X <- as.data.frame(lapply(X, function(x) suppressWarnings(as.numeric(as.character(x)))))
      keep <- vapply(X, function(v) any(is.finite(v)), logical(1))
      X <- as.matrix(X[, keep, drop = FALSE])
      y <- safe_factor(e$r[, "label_idx"])
      sp <- fixed_train_split(nrow(X), train_n)
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

    if (all(c("data", "labels") %in% objs)) {
      X <- as.matrix(e$data)
      y <- safe_factor(e$labels)
      sp <- fixed_train_split(nrow(X), train_n)
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
  data.frame(
    variant_name = c(
      "cpp_plssvd_cpu_rsvd",
      "cpp_plssvd_irlba",
      "cpp_plssvd_arpack",
      "r_plssvd_cpu_rsvd",
      "r_plssvd_irlba",
      "r_plssvd_arpack",
      "gpu_plssvd_fp64",
      "gpu_plssvd_fp32",
      "cpp_simpls_cpu_rsvd",
      "cpp_simpls_irlba",
      "cpp_simpls_arpack",
      "r_simpls_cpu_rsvd",
      "r_simpls_irlba",
      "r_simpls_arpack",
      "pls_pkg_simpls",
      "cpp_simpls_fast_cpu_rsvd",
      "cpp_simpls_fast_irlba",
      "cpp_simpls_fast_arpack",
      "r_simpls_fast_cpu_rsvd",
      "r_simpls_fast_irlba",
      "r_simpls_fast_arpack",
      "gpu_simpls_fast_fp64",
      "gpu_simpls_fast_fp32"
    ),
    method_family = c(
      "plssvd", "plssvd", "plssvd", "plssvd", "plssvd", "plssvd", "plssvd", "plssvd",
      "simpls", "simpls", "simpls", "simpls", "simpls", "simpls", "simpls",
      "simpls_fast", "simpls_fast", "simpls_fast", "simpls_fast", "simpls_fast", "simpls_fast", "simpls_fast", "simpls_fast"
    ),
    engine = c(
      "CPU", "CPU", "CPU", "CPU", "CPU", "CPU", "GPU", "GPU",
      "CPU", "CPU", "CPU", "CPU", "CPU", "CPU", "CPU",
      "CPU", "CPU", "CPU", "CPU", "CPU", "CPU", "GPU", "GPU"
    ),
    backend = c(
      "cpu_rsvd", "irlba", "arpack", "cpu_rsvd", "irlba", "arpack", "gpu_native", "gpu_native",
      "cpu_rsvd", "irlba", "arpack", "cpu_rsvd", "irlba", "arpack", "pls_pkg",
      "cpu_rsvd", "irlba", "arpack", "cpu_rsvd", "irlba", "arpack", "gpu_native", "gpu_native"
    ),
    implementation_label = c(
      "Cpp", "Cpp", "Cpp", "R", "R", "R", "CUDA 64-bit", "CUDA 32-bit",
      "Cpp", "Cpp", "Cpp", "R", "R", "R", "pls_pkg",
      "Cpp", "Cpp", "Cpp", "R", "R", "R", "CUDA 64-bit", "CUDA 32-bit"
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
