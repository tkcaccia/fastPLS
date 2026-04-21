`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

simfast_parse_csv <- function(x) {
  vals <- trimws(unlist(strsplit(x %||% "", ",", fixed = TRUE), use.names = FALSE))
  vals[nzchar(vals)]
}

simfast_bool_env <- function(name, default = FALSE) {
  val <- tolower(Sys.getenv(name, if (default) "true" else "false"))
  val %in% c("1", "true", "yes", "y")
}

simfast_num_env <- function(name, default) {
  val <- suppressWarnings(as.numeric(Sys.getenv(name, as.character(default))))
  if (!is.finite(val) || is.na(val)) default else val
}

simfast_int_env <- function(name, default) {
  val <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (!is.finite(val) || is.na(val)) default else val
}

simfast_default_grids <- function() {
  list(
    ncomp = c(2L, 3L, 4L, 5L, 10L, 20L, 50L, 100L, 200L),
    sample_fraction = c(0.33, 0.66, 1.0),
    xvar_fraction = c(0.10, 0.20, 0.50, 1.0),
    yvar_fraction = c(0.10, 0.20, 0.50, 1.0),
    spectrum_regime = c("fast_decay", "sharp_decay", "slow_decay", "clustered_top"),
    noise_regime = c("low_noise", "medium_noise", "high_noise")
  )
}

simfast_noise_targets <- c(
  low_noise = 20,
  medium_noise = 5,
  high_noise = 1
)

simfast_family_catalog <- function() {
  list(
    sim_reg_base = list(
      sim_family = "sim_reg_base",
      panel = "main",
      task_type = "regression",
      n = 1500L,
      p = 5000L,
      q = 50L,
      n_classes = NA_integer_,
      r_true = 20L,
      spectrum_regime = "sharp_decay",
      noise_regime = "medium_noise",
      dropout_rate = 0,
      count_like = FALSE,
      small_pls_pkg = TRUE,
      description = "Baseline multivariate regression synthetic dataset."
    ),
    sim_reg_large_q = list(
      sim_family = "sim_reg_large_q",
      panel = "main",
      task_type = "regression",
      n = 1500L,
      p = 5000L,
      q = 500L,
      n_classes = NA_integer_,
      r_true = 20L,
      spectrum_regime = "slow_decay",
      noise_regime = "medium_noise",
      dropout_rate = 0,
      count_like = FALSE,
      small_pls_pkg = FALSE,
      description = "Large-q regression to stress multivariate response scaling."
    ),
    sim_reg_pggn = list(
      sim_family = "sim_reg_pggn",
      panel = "supplement",
      task_type = "regression",
      n = 300L,
      p = 20000L,
      q = 20L,
      n_classes = NA_integer_,
      r_true = 10L,
      spectrum_regime = "sharp_decay",
      noise_regime = "medium_noise",
      dropout_rate = 0,
      count_like = FALSE,
      small_pls_pkg = FALSE,
      description = "Extreme p >> n regression synthetic dataset."
    ),
    sim_cls_base = list(
      sim_family = "sim_cls_base",
      panel = "main",
      task_type = "classification",
      n = 1500L,
      p = 5000L,
      q = 10L,
      n_classes = 10L,
      r_true = 20L,
      spectrum_regime = "sharp_decay",
      noise_regime = "medium_noise",
      dropout_rate = 0,
      count_like = FALSE,
      class_bias = rep(0, 10L),
      small_pls_pkg = TRUE,
      description = "Baseline multiclass classification synthetic dataset."
    ),
    sim_cls_many_classes = list(
      sim_family = "sim_cls_many_classes",
      panel = "supplement",
      task_type = "classification",
      n = 3000L,
      p = 5000L,
      q = 50L,
      n_classes = 50L,
      r_true = 30L,
      spectrum_regime = "slow_decay",
      noise_regime = "medium_noise",
      dropout_rate = 0,
      count_like = FALSE,
      class_bias = rep(0, 50L),
      small_pls_pkg = FALSE,
      description = "Many-class classification synthetic dataset."
    ),
    sim_sparse_singlecell_like = list(
      sim_family = "sim_sparse_singlecell_like",
      panel = "main",
      task_type = "classification",
      n = 5000L,
      p = 10000L,
      q = 20L,
      n_classes = 20L,
      r_true = 20L,
      spectrum_regime = "slow_decay",
      noise_regime = "medium_noise",
      dropout_rate = 0.8,
      count_like = TRUE,
      class_bias = rep(0, 20L),
      small_pls_pkg = FALSE,
      description = "Sparse single-cell-like classification dataset with zero inflation."
    )
  )
}

simfast_manifest_dt <- function() {
  fams <- simfast_family_catalog()
  data.table::rbindlist(lapply(fams, function(cfg) {
    data.table::data.table(
      sim_family = cfg$sim_family,
      panel = cfg$panel,
      task_type = cfg$task_type,
      n = cfg$n,
      p = cfg$p,
      q = cfg$q,
      n_classes = cfg$n_classes %||% NA_integer_,
      r_true = cfg$r_true,
      spectrum_regime_default = cfg$spectrum_regime,
      noise_regime_default = cfg$noise_regime,
      dropout_rate_default = cfg$dropout_rate %||% 0,
      count_like_default = isTRUE(cfg$count_like),
      pls_pkg_allowed = isTRUE(cfg$small_pls_pkg),
      description = cfg$description %||% ""
    )
  }), fill = TRUE)
}

simfast_select_families <- function(panel = c("main", "all", "supplement"), families = NULL) {
  panel <- match.arg(panel)
  catalog <- simfast_family_catalog()
  if (!is.null(families) && length(families)) {
    miss <- setdiff(families, names(catalog))
    if (length(miss)) stop("Unknown synthetic families: ", paste(miss, collapse = ", "))
    return(families)
  }
  fam_dt <- simfast_manifest_dt()
  if (panel == "all") return(fam_dt$sim_family)
  panel_name <- panel
  fam_dt[panel == panel_name, sim_family]
}

simfast_default_analyses <- function() {
  c("ncomp", "sample_fraction", "xvar_fraction", "yvar_fraction", "spectrum_and_noise")
}

simfast_select_analyses <- function(analyses = NULL) {
  vals <- analyses %||% simfast_default_analyses()
  miss <- setdiff(vals, simfast_default_analyses())
  if (length(miss)) stop("Unknown simulation analyses: ", paste(miss, collapse = ", "))
  vals
}

simfast_mode_reps <- function(mode = c("paper", "stability"), explicit_reps = NA_integer_) {
  mode <- match.arg(mode)
  if (is.finite(explicit_reps) && !is.na(explicit_reps) && explicit_reps >= 1L) {
    return(as.integer(explicit_reps))
  }
  if (identical(mode, "stability")) 10L else 3L
}

simfast_spectrum <- function(rank, regime = c("fast_decay", "sharp_decay", "slow_decay", "clustered_top"), beta = NULL) {
  regime <- match.arg(regime)
  idx <- seq_len(rank)
  if (regime == "fast_decay") {
    return(1 / (idx^2))
  }
  if (regime == "slow_decay") {
    return(1 / (idx^0.1))
  }
  if (regime == "sharp_decay") {
    beta <- beta %||% max(4, floor(rank / 3))
    return(1e-4 + 1 / (1 + exp(idx + 1 - beta)))
  }
  top <- min(10L, rank)
  vals <- c(rep(1, top), seq(from = 0.8, to = 0.2, length.out = max(rank - top, 1L)))
  vals[seq_len(rank)]
}

simfast_random_orthonormal <- function(nrow, ncol) {
  z <- matrix(rnorm(nrow * ncol), nrow = nrow, ncol = ncol)
  q <- qr.Q(qr(z))
  q[, seq_len(ncol), drop = FALSE]
}

simfast_center_matrix <- function(x) {
  sweep(x, 2, colMeans(x), "-", check.margin = FALSE)
}

simfast_total_variance <- function(x) {
  mean((x - mean(x))^2)
}

simfast_scale_noise <- function(signal, noise, target_snr) {
  signal_var <- simfast_total_variance(signal)
  noise_var_unit <- simfast_total_variance(noise)
  sigma <- if (signal_var <= 0 || noise_var_unit <= 0) 0 else sqrt(signal_var / (target_snr * noise_var_unit))
  list(
    noise = sigma * noise,
    sigma = sigma,
    signal_var = signal_var,
    noise_var = simfast_total_variance(sigma * noise),
    realized_snr = if (signal_var <= 0 || sigma == 0) NA_real_ else signal_var / simfast_total_variance(sigma * noise)
  )
}

simfast_apply_dropout <- function(x, dropout_rate = 0, count_like = FALSE, count_scale = 3) {
  if (!is.finite(dropout_rate) || is.na(dropout_rate) || dropout_rate <= 0) {
    if (isTRUE(count_like)) {
      xpos <- x - min(x)
      xpos <- xpos / (mean(xpos) + 1e-8)
      lam <- pmax(xpos * count_scale, 0)
      x <- matrix(rpois(length(lam), lambda = as.vector(lam)), nrow = nrow(x), ncol = ncol(x))
      x <- log1p(x)
    }
    return(x)
  }

  if (isTRUE(count_like)) {
    xpos <- x - min(x)
    xpos <- xpos / (mean(xpos) + 1e-8)
    lam <- pmax(xpos * count_scale, 0)
    x <- matrix(rpois(length(lam), lambda = as.vector(lam)), nrow = nrow(x), ncol = ncol(x))
  }

  drop_mask <- matrix(runif(length(x)) < dropout_rate, nrow = nrow(x), ncol = ncol(x))
  x[drop_mask] <- 0
  if (isTRUE(count_like)) {
    x <- log1p(x)
  }
  x
}

simfast_dummy_matrix <- function(y) {
  lev <- levels(y)
  out <- matrix(0, nrow = length(y), ncol = length(lev))
  out[cbind(seq_along(y), match(y, lev))] <- 1
  colnames(out) <- lev
  out
}

simfast_stratified_split <- function(y, train_fraction = 0.8) {
  y <- droplevels(as.factor(y))
  idx_by <- split(seq_along(y), y)
  train_idx <- unlist(lapply(idx_by, function(ix) {
    n_take <- max(1L, min(length(ix) - 1L, floor(length(ix) * train_fraction)))
    if (length(ix) <= 2L) {
      ix[seq_len(max(1L, length(ix) - 1L))]
    } else {
      sample(ix, n_take)
    }
  }), use.names = FALSE)
  train_idx <- sort(unique(train_idx))
  test_idx <- setdiff(seq_along(y), train_idx)
  if (!length(test_idx)) stop("Stratified split produced empty test set")
  list(train = train_idx, test = test_idx)
}

simfast_random_split <- function(n, train_fraction = 0.8) {
  n_train <- max(1L, min(n - 1L, floor(n * train_fraction)))
  train_idx <- sort(sample.int(n, size = n_train))
  list(train = train_idx, test = setdiff(seq_len(n), train_idx))
}

simfast_sample_rows <- function(y, fraction) {
  n_total <- if (is.factor(y) || is.vector(y)) {
    length(y)
  } else {
    nrow(as.matrix(y))
  }
  if (!is.finite(fraction) || is.na(fraction) || fraction >= 1) return(seq_len(n_total))
  if (is.factor(y)) {
    idx_by <- split(seq_along(y), y)
    keep <- unlist(lapply(idx_by, function(ix) {
      n_take <- max(1L, floor(length(ix) * fraction))
      if (n_take >= length(ix)) ix else sample(ix, n_take)
    }), use.names = FALSE)
    return(sort(unique(keep)))
  }
  n_keep <- max(1L, min(n_total, floor(n_total * fraction)))
  sort(sample.int(n_total, n_keep))
}

simfast_apply_sample_fraction <- function(ds, fraction, seed) {
  set.seed(seed)
  train_idx <- simfast_sample_rows(ds$Ytrain, fraction)
  set.seed(seed + 1L)
  test_idx <- simfast_sample_rows(ds$Ytest, fraction)
  ds$Xtrain <- ds$Xtrain[train_idx, , drop = FALSE]
  ds$Xtest <- ds$Xtest[test_idx, , drop = FALSE]
  if (is.factor(ds$Ytrain)) {
    ds$Ytrain <- droplevels(ds$Ytrain[train_idx])
    ds$Ytest <- droplevels(factor(ds$Ytest[test_idx], levels = levels(ds$Ytrain)))
    if (!is.null(ds$Ytrain_dummy)) {
      ds$Ytrain_dummy <- simfast_dummy_matrix(ds$Ytrain)
    }
    if (!is.null(ds$Ytest_dummy)) {
      ds$Ytest_dummy <- simfast_dummy_matrix(ds$Ytest)
    }
  } else {
    ds$Ytrain <- ds$Ytrain[train_idx, , drop = FALSE]
    ds$Ytest <- ds$Ytest[test_idx, , drop = FALSE]
  }
  ds
}

simfast_apply_xvar_fraction <- function(ds, fraction, seed) {
  set.seed(seed)
  p <- ncol(ds$Xtrain)
  keep <- max(1L, min(p, floor(p * fraction)))
  cols <- sort(sample.int(p, size = keep))
  ds$Xtrain <- ds$Xtrain[, cols, drop = FALSE]
  ds$Xtest <- ds$Xtest[, cols, drop = FALSE]
  ds
}

simfast_apply_yvar_fraction <- function(ds, fraction, seed) {
  if (is.factor(ds$Ytrain)) return(ds)
  set.seed(seed)
  q <- ncol(ds$Ytrain)
  keep <- max(1L, min(q, floor(q * fraction)))
  cols <- sort(sample.int(q, size = keep))
  ds$Ytrain <- ds$Ytrain[, cols, drop = FALSE]
  ds$Ytest <- ds$Ytest[, cols, drop = FALSE]
  ds
}

simfast_class_margin <- function(logits) {
  mean(apply(logits, 1, function(v) {
    s <- sort(v, decreasing = TRUE)
    if (length(s) < 2L) NA_real_ else s[1] - s[2]
  }), na.rm = TRUE)
}

simfast_generate_regression <- function(cfg, seed_data, seed_split, train_fraction = 0.8) {
  set.seed(seed_data)
  n <- cfg$n
  p <- cfg$p
  q <- cfg$q
  r_true <- cfg$r_true

  Tmat <- matrix(rnorm(n * r_true), nrow = n, ncol = r_true)
  P <- simfast_random_orthonormal(p, r_true)
  C <- simfast_random_orthonormal(q, r_true)
  sx <- simfast_spectrum(r_true, cfg$spectrum_regime)
  sy <- simfast_spectrum(r_true, cfg$spectrum_regime)

  X_signal <- sweep(Tmat, 2, sx, "*") %*% t(P)
  Y_signal <- sweep(Tmat, 2, sy, "*") %*% t(C)
  X_noise_raw <- matrix(rnorm(n * p), nrow = n, ncol = p)
  Y_noise_raw <- matrix(rnorm(n * q), nrow = n, ncol = q)
  target_snr <- simfast_noise_targets[[cfg$noise_regime]]
  X_noise <- simfast_scale_noise(X_signal, X_noise_raw, target_snr)
  Y_noise <- simfast_scale_noise(Y_signal, Y_noise_raw, target_snr)

  X_full <- X_signal + X_noise$noise
  X_full <- simfast_apply_dropout(X_full, cfg$dropout_rate %||% 0, count_like = isTRUE(cfg$count_like))
  observed_zero_rate_X <- mean(X_full == 0)
  X_full <- simfast_center_matrix(X_full)
  Y_full <- simfast_center_matrix(Y_signal + Y_noise$noise)

  set.seed(seed_split)
  split <- simfast_random_split(n, train_fraction = train_fraction)

  list(
    task_type = "regression",
    Xtrain = X_full[split$train, , drop = FALSE],
    Xtest = X_full[split$test, , drop = FALSE],
    Ytrain = Y_full[split$train, , drop = FALSE],
    Ytest = Y_full[split$test, , drop = FALSE],
    meta = list(
      n = n,
      p = p,
      q = q,
      n_classes = NA_integer_,
      r_true = r_true,
      spectrum_regime = cfg$spectrum_regime,
      noise_regime = cfg$noise_regime,
      dropout_regime = if ((cfg$dropout_rate %||% 0) > 0) sprintf("dropout_%0.2f", cfg$dropout_rate) else "none",
      seed_data = seed_data,
      seed_split = seed_split,
      signal_var_X = X_noise$signal_var,
      noise_var_X = X_noise$noise_var,
      signal_var_Y = Y_noise$signal_var,
      noise_var_Y = Y_noise$noise_var,
      realized_snr_X = X_noise$realized_snr,
      realized_snr_Y = Y_noise$realized_snr,
      observed_zero_rate_X = observed_zero_rate_X,
      class_margin = NA_real_
    )
  )
}

simfast_generate_classification <- function(cfg, seed_data, seed_split, train_fraction = 0.8, max_tries = 20L) {
  n <- cfg$n
  p <- cfg$p
  k <- cfg$n_classes
  r_true <- cfg$r_true
  class_bias <- cfg$class_bias %||% rep(0, k)
  if (length(class_bias) != k) {
    class_bias <- rep(class_bias, length.out = k)
  }

  for (attempt in seq_len(max_tries)) {
    set.seed(seed_data + attempt - 1L)
    Tmat <- matrix(rnorm(n * r_true), nrow = n, ncol = r_true)
    P <- simfast_random_orthonormal(p, r_true)
    sx <- simfast_spectrum(r_true, cfg$spectrum_regime)
    X_signal <- sweep(Tmat, 2, sx, "*") %*% t(P)
    X_noise_raw <- matrix(rnorm(n * p), nrow = n, ncol = p)
    target_snr <- simfast_noise_targets[[cfg$noise_regime]]
    X_noise <- simfast_scale_noise(X_signal, X_noise_raw, target_snr)
    X_full <- X_signal + X_noise$noise
    X_full <- simfast_apply_dropout(X_full, cfg$dropout_rate %||% 0, count_like = isTRUE(cfg$count_like))
    observed_zero_rate_X <- mean(X_full == 0)
    X_full <- simfast_center_matrix(X_full)

    W <- matrix(rnorm(r_true * k), nrow = r_true, ncol = k)
    W <- sweep(W, 2, sqrt(colSums(W^2)) + 1e-8, "/")
    G_signal <- Tmat %*% W + matrix(rep(class_bias, each = n), nrow = n, ncol = k)
    G_noise_raw <- matrix(rnorm(n * k), nrow = n, ncol = k)
    G_noise <- simfast_scale_noise(G_signal, G_noise_raw, target_snr)
    logits <- G_signal + G_noise$noise
    y_idx <- max.col(logits, ties.method = "first")
    y <- factor(paste0("cls_", y_idx), levels = paste0("cls_", seq_len(k)))
    if (length(unique(y)) < k) next

    set.seed(seed_split + attempt - 1L)
    split <- simfast_stratified_split(y, train_fraction = train_fraction)
    if (length(unique(y[split$train])) < k || length(unique(y[split$test])) < k) next

    return(list(
      task_type = "classification",
      Xtrain = X_full[split$train, , drop = FALSE],
      Xtest = X_full[split$test, , drop = FALSE],
      Ytrain = droplevels(y[split$train]),
      Ytest = droplevels(factor(y[split$test], levels = levels(y))),
      Ytrain_dummy = simfast_dummy_matrix(droplevels(y[split$train])),
      Ytest_dummy = simfast_dummy_matrix(droplevels(factor(y[split$test], levels = levels(y)))),
      meta = list(
        n = n,
        p = p,
        q = k,
        n_classes = k,
        r_true = r_true,
        spectrum_regime = cfg$spectrum_regime,
        noise_regime = cfg$noise_regime,
        dropout_regime = if ((cfg$dropout_rate %||% 0) > 0) sprintf("dropout_%0.2f", cfg$dropout_rate) else "none",
        seed_data = seed_data + attempt - 1L,
        seed_split = seed_split + attempt - 1L,
        signal_var_X = X_noise$signal_var,
        noise_var_X = X_noise$noise_var,
        signal_var_Y = G_noise$signal_var,
        noise_var_Y = G_noise$noise_var,
        realized_snr_X = X_noise$realized_snr,
        realized_snr_Y = G_noise$realized_snr,
        observed_zero_rate_X = observed_zero_rate_X,
        class_margin = simfast_class_margin(logits)
      )
    ))
  }
  stop("Could not generate non-degenerate classification data for family ", cfg$sim_family)
}

simfast_generate_dataset <- function(cfg, seed_data, seed_split, train_fraction = 0.8) {
  if (identical(cfg$task_type, "regression")) {
    out <- simfast_generate_regression(cfg, seed_data = seed_data, seed_split = seed_split, train_fraction = train_fraction)
  } else {
    out <- simfast_generate_classification(cfg, seed_data = seed_data, seed_split = seed_split, train_fraction = train_fraction)
  }
  out$sim_family <- cfg$sim_family
  out
}

simfast_legal_ncomp <- function(ds) {
  y_cap <- if (is.factor(ds$Ytrain)) nlevels(ds$Ytrain) else ncol(as.matrix(ds$Ytrain))
  max(1L, min(nrow(ds$Xtrain) - 1L, ncol(ds$Xtrain), y_cap))
}

simfast_extract_regression_slice <- function(x, ncomp_index) {
  if (is.null(x)) return(NULL)
  if (length(dim(x)) == 3L) {
    return(as.matrix(x[, , ncomp_index, drop = TRUE]))
  }
  if (is.matrix(x)) return(x)
  if (is.data.frame(x)) return(as.matrix(x))
  matrix(as.numeric(x), ncol = 1)
}

simfast_extract_class_column <- function(x, ncomp_index) {
  if (is.null(x)) return(NULL)
  if (is.data.frame(x)) {
    idx <- min(ncomp_index, ncol(x))
    return(as.factor(x[[idx]]))
  }
  if (is.factor(x)) return(x)
  as.factor(x)
}

simfast_regression_scores <- function(y_true, y_pred) {
  y_true <- as.matrix(y_true)
  y_pred <- as.matrix(y_pred)
  sse <- sum((y_true - y_pred)^2)
  sst <- sum((y_true - matrix(colMeans(y_true), nrow = nrow(y_true), ncol = ncol(y_true), byrow = TRUE))^2)
  if (!is.finite(sst) || sst <= 0) return(NA_real_)
  1 - (sse / sst)
}

simfast_measure_model <- function(model, ds, task_type, effective_ncomp) {
  if (identical(task_type, "classification")) {
    pred <- simfast_extract_class_column(model$Ypred, effective_ncomp)
    acc <- mean(as.character(pred) == as.character(ds$Ytest), na.rm = TRUE)
    return(list(
      accuracy = as.numeric(acc),
      Q2 = NA_real_,
      train_R2 = NA_real_
    ))
  }

  y_pred_test <- simfast_extract_regression_slice(model$Ypred, effective_ncomp)
  y_fit_train <- simfast_extract_regression_slice(model$Yfit, effective_ncomp)
  list(
    accuracy = NA_real_,
    Q2 = simfast_regression_scores(ds$Ytest, y_pred_test),
    train_R2 = simfast_regression_scores(ds$Ytrain, y_fit_train)
  )
}

simfast_pls_pkg_fit <- function(ds, task_type, effective_ncomp) {
  if (!requireNamespace("pls", quietly = TRUE)) {
    stop("pls package not available")
  }

  if (identical(task_type, "classification")) {
    y_factor <- droplevels(ds$Ytrain)
    Ymm <- simfast_dummy_matrix(y_factor)
    x_names <- paste0("x_", seq_len(ncol(ds$Xtrain)))
    Xtr <- ds$Xtrain
    colnames(Xtr) <- x_names
    df_train <- data.frame(Ymm, Xtr, check.names = FALSE)
    form <- stats::as.formula(paste0("cbind(", paste(colnames(Ymm), collapse = ","), ") ~ ."))

    t0 <- proc.time()[3]
    mdl <- pls::plsr(form, data = df_train, ncomp = as.integer(effective_ncomp), method = "simpls", scale = FALSE, validation = "none")
    elapsed_ms <- (proc.time()[3] - t0) * 1000

    Xte <- as.data.frame(ds$Xtest)
    colnames(Xte) <- x_names
    pred_arr <- predict(mdl, newdata = Xte, ncomp = as.integer(effective_ncomp))
    pred_mat <- pred_arr[, , 1, drop = TRUE]
    pred_idx <- max.col(pred_mat, ties.method = "first")
    lev <- colnames(Ymm)
    pred <- factor(lev[pred_idx], levels = lev)
    acc <- mean(as.character(pred) == as.character(ds$Ytest), na.rm = TRUE)

    list(
      elapsed_ms = as.numeric(elapsed_ms),
      accuracy = as.numeric(acc),
      Q2 = NA_real_,
      train_R2 = NA_real_,
      model_size_mb = as.numeric(object.size(mdl)) / (1024^2)
    )
  } else {
    ymat <- as.matrix(ds$Ytrain)
    x_names <- paste0("x_", seq_len(ncol(ds$Xtrain)))
    Xtr <- ds$Xtrain
    colnames(Xtr) <- x_names
    df_train <- data.frame(ymat, Xtr, check.names = FALSE)
    y_cols <- colnames(df_train)[seq_len(ncol(ymat))]
    form <- stats::as.formula(paste0("cbind(", paste(y_cols, collapse = ","), ") ~ ."))

    t0 <- proc.time()[3]
    mdl <- pls::plsr(form, data = df_train, ncomp = as.integer(effective_ncomp), method = "simpls", scale = FALSE, validation = "none")
    elapsed_ms <- (proc.time()[3] - t0) * 1000

    Xte <- as.data.frame(ds$Xtest)
    colnames(Xte) <- x_names
    pred_arr <- predict(mdl, newdata = Xte, ncomp = as.integer(effective_ncomp))
    pred_test <- as.matrix(pred_arr[, , 1, drop = TRUE])
    fit_arr <- predict(mdl, newdata = as.data.frame(Xtr), ncomp = as.integer(effective_ncomp))
    fit_train <- as.matrix(fit_arr[, , 1, drop = TRUE])

    list(
      elapsed_ms = as.numeric(elapsed_ms),
      accuracy = NA_real_,
      Q2 = simfast_regression_scores(ds$Ytest, pred_test),
      train_R2 = simfast_regression_scores(ds$Ytrain, fit_train),
      model_size_mb = as.numeric(object.size(mdl)) / (1024^2)
    )
  }
}

simfast_fastpls_fit <- function(ds, cfg, task_type, effective_ncomp, seed_fit) {
  ncomp_seq <- seq_len(effective_ncomp)
  common_args <- list(
    Xtrain = ds$Xtrain,
    Ytrain = ds$Ytrain,
    Xtest = ds$Xtest,
    Ytest = ds$Ytest,
    ncomp = ncomp_seq,
    scaling = "centering",
    fit = TRUE,
    proj = FALSE,
    seed = as.integer(seed_fit)
  )

  t0 <- proc.time()[3]
  if (identical(cfg$engine, "GPU")) {
    if (identical(cfg$method, "plssvd")) {
      model <- do.call(fastPLS::plssvd_gpu, c(common_args, list(
        rsvd_oversample = 10L,
        rsvd_power = 1L,
        svds_tol = 0
      )))
    } else if (identical(cfg$method, "simpls_fast")) {
      model <- do.call(fastPLS::simpls_gpu, c(common_args, list(
        rsvd_oversample = 10L,
        rsvd_power = 1L,
        svds_tol = 0
      )))
    } else {
      stop("Unsupported GPU synthetic method: ", cfg$method)
    }
  } else {
    fn <- if (identical(cfg$engine, "R")) fastPLS::pls_r else fastPLS::pls
    extra_args <- list(
      method = cfg$method,
      svd.method = cfg$svd_method,
      rsvd_oversample = 10L,
      rsvd_power = 1L,
      svds_tol = 0,
      irlba_svtol = 1e-6,
      rsvd_tol = 0
    )
    if (identical(cfg$method, "simpls_fast")) {
      extra_args <- c(extra_args, list(
        fast_incremental = TRUE,
        fast_inc_iters = 2L,
        fast_defl_cache = TRUE,
        fast_center_t = FALSE,
        fast_reorth_v = FALSE,
        fast_block = 8L
      ))
    }
    model <- do.call(fn, c(common_args, extra_args))
  }
  elapsed_ms <- (proc.time()[3] - t0) * 1000
  metrics <- simfast_measure_model(model, ds, task_type = task_type, effective_ncomp = effective_ncomp)

  list(
    elapsed_ms = as.numeric(elapsed_ms),
    accuracy = metrics$accuracy,
    Q2 = metrics$Q2,
    train_R2 = metrics$train_R2,
    model_size_mb = as.numeric(object.size(model)) / (1024^2)
  )
}

simfast_method_grid <- function(include_cuda = FALSE, include_r_impl = FALSE) {
  dt <- data.table::CJ(
    engine = "Rcpp",
    method = c("plssvd", "simpls", "simpls_fast"),
    svd_method = c("irlba", "cpu_rsvd", "arpack"),
    fast_profile = c("default", "default", "incdefl"),
    unique = TRUE
  )
  dt <- dt[
    (method %in% c("plssvd", "simpls") & fast_profile == "default") |
      (method == "simpls_fast" & fast_profile == "incdefl")
  ]

  if (isTRUE(include_r_impl)) {
    dtr <- data.table::copy(dt)
    dtr[, engine := "R"]
    dt <- data.table::rbindlist(list(dt, dtr), fill = TRUE)
  }

  if (isTRUE(include_cuda) && isTRUE(fastPLS::has_cuda())) {
    dt <- data.table::rbindlist(list(
      dt,
      data.table::data.table(
        engine = "GPU",
        method = c("plssvd", "simpls_fast"),
        svd_method = "gpu_native",
        fast_profile = "gpu_native"
      )
    ), fill = TRUE)
  }

  dt[, method_id := paste(engine, method, svd_method, fast_profile, sep = "_")]
  dt[]
}

simfast_methods_for_family <- function(sim_family,
                                       methods_all,
                                       include_pls_pkg = FALSE,
                                       include_pls_pkg_pggn = FALSE) {
  out <- data.table::copy(methods_all)
  if (isTRUE(include_pls_pkg) || (isTRUE(include_pls_pkg_pggn) && identical(sim_family, "sim_reg_pggn"))) {
    out <- data.table::rbindlist(list(
      out,
      data.table::data.table(
        engine = "pls_pkg",
        method = "simpls",
        svd_method = "pls_pkg",
        fast_profile = "pls_pkg",
        method_id = "pls_pkg_simpls_pls_pkg_pls_pkg"
      )
    ), fill = TRUE)
  }
  out[]
}

simfast_method_label <- function(engine, method, svd_method) {
  if (engine == "GPU") {
    return(sprintf("GPU / %s", method))
  }
  if (engine == "pls_pkg") {
    return("pls_pkg / simpls")
  }
  sprintf("%s / %s / %s", engine, method, svd_method)
}

simfast_prepare_analysis_dataset <- function(base_ds, analysis_type, analysis_value, seed_analysis) {
  ds <- base_ds
  sample_fraction <- 1.0
  xvar_fraction <- 1.0
  yvar_fraction <- if (is.factor(ds$Ytrain)) NA_real_ else 1.0

  if (identical(analysis_type, "sample_fraction")) {
    sample_fraction <- as.numeric(analysis_value)
    ds <- simfast_apply_sample_fraction(ds, sample_fraction, seed_analysis)
  } else if (identical(analysis_type, "xvar_fraction")) {
    xvar_fraction <- as.numeric(analysis_value)
    ds <- simfast_apply_xvar_fraction(ds, xvar_fraction, seed_analysis)
  } else if (identical(analysis_type, "yvar_fraction")) {
    yvar_fraction <- as.numeric(analysis_value)
    ds <- simfast_apply_yvar_fraction(ds, yvar_fraction, seed_analysis)
  }

  list(
    dataset = ds,
    sample_fraction = sample_fraction,
    xvar_fraction = xvar_fraction,
    yvar_fraction = yvar_fraction
  )
}

simfast_validate_dataset <- function(ds) {
  stopifnot(all(is.finite(ds$Xtrain)), all(is.finite(ds$Xtest)))
  if (is.factor(ds$Ytrain)) {
    if (length(unique(ds$Ytrain)) < 2L) stop("Degenerate classification train labels")
    if (length(unique(ds$Ytest)) < 2L) stop("Degenerate classification test labels")
  } else {
    stopifnot(all(is.finite(ds$Ytrain)), all(is.finite(ds$Ytest)))
  }
  invisible(TRUE)
}
