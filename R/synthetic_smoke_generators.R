`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

if (getRversion() >= "2.15.1") {
  utils::globalVariables(c(
    ".", ":=", ".N", "algorithm", "axis", "effective_ncomp", "engine",
    "fast_profile", "method_id", "msg", "n_train", "n_unique",
    "noise_rank", "p", "scenario_family", "status", "svd_method",
    "value"
  ))
}

smoke_noise_targets <- c(
  very_low_noise = 40,
  low_noise = 20,
  medium_noise = 10,
  high_noise = 5,
  very_high_noise = 1
)

smoke_noise_levels <- function() {
  names(smoke_noise_targets)
}

smoke_default_noise_level <- function() {
  "medium_noise"
}

smoke_noise_target_snr <- function(noise_regime) {
  unname(smoke_noise_targets[[as.character(noise_regime)]])
}

smoke_noise_rank <- function(noise_regime) {
  match(as.character(noise_regime), smoke_noise_levels())
}

smoke_requested_ncomp_default <- function() {
  50L
}

smoke_grid_n_train <- function() c(100L, 200L, 300L, 400L, 500L, 1000L, 2000L, 5000L, 10000L)
smoke_grid_p <- function() c(2L, 5L, 10L, 20L, 50L, 100L, 200L, 500L, 1000L)
smoke_grid_q <- function() c(2L, 5L, 10L, 20L, 50L, 100L, 200L, 500L, 1000L)

smoke_family_specs <- function() {
  list(
    sim_reg_n_p50 = list(
      task_type = "regression",
      analysis = "n_train",
      p = 50L,
      q = 50L,
      n_test = 1000L,
      fixed_noise = smoke_default_noise_level(),
      values = smoke_grid_n_train()
    ),
    sim_reg_n_p500 = list(
      task_type = "regression",
      analysis = "n_train",
      p = 500L,
      q = 50L,
      n_test = 1000L,
      fixed_noise = smoke_default_noise_level(),
      values = smoke_grid_n_train()
    ),
    sim_reg_n_p1000_q1000_ncomp500 = list(
      task_type = "regression",
      analysis = "n_train",
      p = 1000L,
      q = 1000L,
      n_test = 1000L,
      requested_ncomp = 500L,
      fixed_noise = smoke_default_noise_level(),
      values = smoke_grid_n_train()
    ),
    sim_reg_p_sweep = list(
      task_type = "regression",
      analysis = "p",
      n_train = 1000L,
      n_test = 1000L,
      q = 50L,
      fixed_noise = smoke_default_noise_level(),
      values = smoke_grid_p()
    ),
    sim_reg_q_sweep = list(
      task_type = "regression",
      analysis = "q",
      n_train = 1000L,
      n_test = 1000L,
      p = 500L,
      fixed_noise = smoke_default_noise_level(),
      values = smoke_grid_q()
    ),
    sim_reg_noise_sweep = list(
      task_type = "regression",
      analysis = "noise_regime",
      n_train = 1000L,
      n_test = 1000L,
      p = 500L,
      q = 50L,
      values = smoke_noise_levels()
    )
  )
}

smoke_filter_call_args <- function(fn, args) {
  keep <- intersect(names(args), names(formals(fn)))
  args[keep]
}

smoke_moderate_profile <- function(rank) {
  if (rank <= 1L) return(1)
  seq(1, 0.2, length.out = rank)
}

smoke_total_variance <- function(x) {
  x <- as.numeric(x)
  stats::var(x)
}

smoke_scale_noise_to_snr <- function(signal, noise_unit, target_snr) {
  signal_var <- smoke_total_variance(signal)
  noise_var_unit <- smoke_total_variance(noise_unit)
  sigma <- if (!is.finite(signal_var) || signal_var <= 0 || !is.finite(noise_var_unit) || noise_var_unit <= 0) {
    0
  } else {
    sqrt(signal_var / (target_snr * noise_var_unit))
  }
  noise <- sigma * noise_unit
  noise_var <- smoke_total_variance(noise)
  list(
    noise = noise,
    sigma = sigma,
    signal_var = signal_var,
    noise_var = noise_var,
    realized_snr = if (is.finite(signal_var) && is.finite(noise_var) && noise_var > 0) signal_var / noise_var else NA_real_
  )
}

smoke_random_orthonormal <- function(nrow, ncol) {
  z <- matrix(stats::rnorm(nrow * ncol), nrow = nrow, ncol = ncol)
  q <- qr.Q(qr(z))
  q[, seq_len(ncol), drop = FALSE]
}

smoke_center_train_test <- function(train, test) {
  mu <- colMeans(train)
  list(
    train = sweep(train, 2, mu, "-", check.margin = FALSE),
    test = sweep(test, 2, mu, "-", check.margin = FALSE),
    center = mu
  )
}

smoke_dummy_matrix <- function(y) {
  y <- droplevels(as.factor(y))
  out <- matrix(0, nrow = length(y), ncol = nlevels(y))
  out[cbind(seq_along(y), as.integer(y))] <- 1
  colnames(out) <- levels(y)
  out
}

smoke_r2_score <- function(y_true, y_pred) {
  y_true <- as.matrix(y_true)
  y_pred <- as.matrix(y_pred)
  y_centered <- sweep(y_true, 2, colMeans(y_true), "-", check.margin = FALSE)
  sst <- sum(y_centered^2)
  if (!is.finite(sst) || sst <= 0) return(NA_real_)
  sse <- sum((y_true - y_pred)^2)
  1 - (sse / sst)
}

smoke_capacity_limit <- function(n_train, p, y_dim, requested_ncomp) {
  max(1L, min(as.integer(requested_ncomp), as.integer(n_train) - 1L, as.integer(p), as.integer(y_dim)))
}

smoke_capacity_limited <- function(n_train, p, y_dim, requested_ncomp) {
  smoke_capacity_limit(n_train, p, y_dim, requested_ncomp) < as.integer(requested_ncomp)
}

smoke_balanced_indices <- function(n, K) {
  rep(seq_len(K), length.out = n)
}

smoke_logit_margin <- function(logits) {
  mean(apply(logits, 1, function(v) {
    s <- sort(v, decreasing = TRUE)
    if (length(s) < 2L) return(NA_real_)
    s[[1L]] - s[[2L]]
  }), na.rm = TRUE)
}

smoke_distance_logits <- function(Tmat, prototypes) {
  t_norm <- rowSums(Tmat^2)
  p_norm <- rowSums(prototypes^2)
  -(outer(t_norm, p_norm, "+") - 2 * (Tmat %*% t(prototypes)))
}

synthetic_smoke_generate_regression <- function(n_train,
                                                n_test,
                                                p,
                                                q,
                                                noise_regime = "medium_noise",
                                                requested_ncomp = smoke_requested_ncomp_default(),
                                                r_true_default = 20L,
                                                seed_data = 1L) {
  stopifnot(noise_regime %in% smoke_noise_levels())
  set.seed(seed_data)

  r_true <- max(1L, min(as.integer(r_true_default), as.integer(n_train) - 1L, as.integer(p), as.integer(q)))
  sx <- smoke_moderate_profile(r_true)
  sy <- smoke_moderate_profile(r_true)
  P <- smoke_random_orthonormal(p, r_true)
  C <- smoke_random_orthonormal(q, r_true)

  Ttrain <- matrix(stats::rnorm(n_train * r_true), nrow = n_train, ncol = r_true)
  Ttest <- matrix(stats::rnorm(n_test * r_true), nrow = n_test, ncol = r_true)

  Xsig_train <- sweep(Ttrain, 2, sx, "*") %*% t(P)
  Xsig_test <- sweep(Ttest, 2, sx, "*") %*% t(P)
  Ysig_train <- sweep(Ttrain, 2, sy, "*") %*% t(C)
  Ysig_test <- sweep(Ttest, 2, sy, "*") %*% t(C)

  target_snr <- smoke_noise_targets[[noise_regime]]
  x_noise_unit_train <- matrix(stats::rnorm(n_train * p), nrow = n_train, ncol = p)
  x_noise_unit_test <- matrix(stats::rnorm(n_test * p), nrow = n_test, ncol = p)
  y_noise_unit_train <- matrix(stats::rnorm(n_train * q), nrow = n_train, ncol = q)
  y_noise_unit_test <- matrix(stats::rnorm(n_test * q), nrow = n_test, ncol = q)

  x_scaled <- smoke_scale_noise_to_snr(rbind(Xsig_train, Xsig_test), rbind(x_noise_unit_train, x_noise_unit_test), target_snr)
  y_scaled <- smoke_scale_noise_to_snr(rbind(Ysig_train, Ysig_test), rbind(y_noise_unit_train, y_noise_unit_test), target_snr)

  X_full <- rbind(Xsig_train, Xsig_test) + x_scaled$noise
  Y_full <- rbind(Ysig_train, Ysig_test) + y_scaled$noise

  Xtrain_raw <- X_full[seq_len(n_train), , drop = FALSE]
  Xtest_raw <- X_full[n_train + seq_len(n_test), , drop = FALSE]
  Ytrain_raw <- Y_full[seq_len(n_train), , drop = FALSE]
  Ytest_raw <- Y_full[n_train + seq_len(n_test), , drop = FALSE]

  Xc <- smoke_center_train_test(Xtrain_raw, Xtest_raw)
  Yc <- smoke_center_train_test(Ytrain_raw, Ytest_raw)

  effective_ncomp <- smoke_capacity_limit(n_train, p, q, requested_ncomp)
  capacity_limited <- effective_ncomp < as.integer(requested_ncomp)

  list(
    task_type = "regression",
    Xtrain = Xc$train,
    Xtest = Xc$test,
    Ytrain = Yc$train,
    Ytest = Yc$test,
    meta = list(
      n_train = as.integer(n_train),
      n_test = as.integer(n_test),
      p = as.integer(p),
      q = as.integer(q),
      K = NA_integer_,
      r_true = as.integer(r_true),
      requested_ncomp = as.integer(requested_ncomp),
      effective_ncomp = as.integer(effective_ncomp),
      capacity_limited = isTRUE(capacity_limited),
      noise_regime = noise_regime,
      signal_var_X = x_scaled$signal_var,
      noise_var_X = x_scaled$noise_var,
      signal_var_Y = y_scaled$signal_var,
      noise_var_Y = y_scaled$noise_var,
      realized_snr_X = x_scaled$realized_snr,
      realized_snr_Y = y_scaled$realized_snr,
      noise_target_snr = target_snr,
      noise_rank = smoke_noise_rank(noise_regime),
      class_separation = NA_real_,
      seed_data = as.integer(seed_data)
    )
  )
}

synthetic_smoke_generate_classification <- function(n_train,
                                                    n_test,
                                                    p,
                                                    K,
                                                    noise_regime = "medium_noise",
                                                    requested_ncomp = smoke_requested_ncomp_default(),
                                                    r_true_default = 20L,
                                                    seed_data = 1L,
                                                    max_attempts = 25L) {
  stopifnot(noise_regime %in% smoke_noise_levels())
  target_snr <- smoke_noise_targets[[noise_regime]]

  r_true <- max(1L, min(as.integer(r_true_default), as.integer(p), as.integer(K)))
  sx <- smoke_moderate_profile(r_true)
  levels_k <- paste0("cls_", seq_len(K))

  for (attempt in seq_len(max_attempts)) {
    set.seed(seed_data + attempt - 1L)
    prototypes <- matrix(stats::rnorm(K * r_true), nrow = K, ncol = r_true)
    prototypes <- sweep(prototypes, 2, apply(prototypes, 2, sd), "/")
    prototypes <- prototypes / (sqrt(rowSums(prototypes^2)) + 1e-8)
    prototypes <- prototypes * 3

    train_base <- smoke_balanced_indices(n_train, K)
    test_base <- smoke_balanced_indices(n_test, K)
    train_base <- sample(train_base, length(train_base), replace = FALSE)
    test_base <- sample(test_base, length(test_base), replace = FALSE)

    latent_sd <- 0.35
    Ttrain <- prototypes[train_base, , drop = FALSE] + latent_sd * matrix(stats::rnorm(n_train * r_true), nrow = n_train, ncol = r_true)
    Ttest <- prototypes[test_base, , drop = FALSE] + latent_sd * matrix(stats::rnorm(n_test * r_true), nrow = n_test, ncol = r_true)

    logits_signal_train <- smoke_distance_logits(Ttrain, prototypes)
    logits_signal_test <- smoke_distance_logits(Ttest, prototypes)
    logits_noise_unit_train <- matrix(stats::rnorm(n_train * K), nrow = n_train, ncol = K)
    logits_noise_unit_test <- matrix(stats::rnorm(n_test * K), nrow = n_test, ncol = K)
    logits_scaled <- smoke_scale_noise_to_snr(
      rbind(logits_signal_train, logits_signal_test),
      rbind(logits_noise_unit_train, logits_noise_unit_test),
      target_snr
    )
    logits_full <- rbind(logits_signal_train, logits_signal_test) + logits_scaled$noise
    logits_train <- logits_full[seq_len(n_train), , drop = FALSE]
    logits_test <- logits_full[n_train + seq_len(n_test), , drop = FALSE]

    ytrain <- factor(levels_k[max.col(logits_train, ties.method = "first")], levels = levels_k)
    ytest <- factor(levels_k[max.col(logits_test, ties.method = "first")], levels = levels_k)
    if (nlevels(droplevels(ytrain)) < K || nlevels(droplevels(ytest)) < K) next

    P <- smoke_random_orthonormal(p, r_true)
    Xsig_train <- sweep(Ttrain, 2, sx, "*") %*% t(P)
    Xsig_test <- sweep(Ttest, 2, sx, "*") %*% t(P)
    x_noise_unit_train <- matrix(stats::rnorm(n_train * p), nrow = n_train, ncol = p)
    x_noise_unit_test <- matrix(stats::rnorm(n_test * p), nrow = n_test, ncol = p)
    x_scaled <- smoke_scale_noise_to_snr(
      rbind(Xsig_train, Xsig_test),
      rbind(x_noise_unit_train, x_noise_unit_test),
      target_snr
    )
    X_full <- rbind(Xsig_train, Xsig_test) + x_scaled$noise
    Xtrain_raw <- X_full[seq_len(n_train), , drop = FALSE]
    Xtest_raw <- X_full[n_train + seq_len(n_test), , drop = FALSE]
    Xc <- smoke_center_train_test(Xtrain_raw, Xtest_raw)

    effective_ncomp <- smoke_capacity_limit(n_train, p, K, requested_ncomp)
    capacity_limited <- effective_ncomp < as.integer(requested_ncomp)

    return(list(
      task_type = "classification",
      Xtrain = Xc$train,
      Xtest = Xc$test,
      Ytrain = ytrain,
      Ytest = ytest,
      Ytrain_dummy = smoke_dummy_matrix(ytrain),
      Ytest_dummy = smoke_dummy_matrix(ytest),
      meta = list(
        n_train = as.integer(n_train),
        n_test = as.integer(n_test),
        p = as.integer(p),
        q = as.integer(K),
        K = as.integer(K),
        r_true = as.integer(r_true),
        requested_ncomp = as.integer(requested_ncomp),
        effective_ncomp = as.integer(effective_ncomp),
        capacity_limited = isTRUE(capacity_limited),
        noise_regime = noise_regime,
        signal_var_X = x_scaled$signal_var,
        noise_var_X = x_scaled$noise_var,
        signal_var_Y = logits_scaled$signal_var,
        noise_var_Y = logits_scaled$noise_var,
        realized_snr_X = x_scaled$realized_snr,
        realized_snr_Y = logits_scaled$realized_snr,
        noise_target_snr = target_snr,
        noise_rank = smoke_noise_rank(noise_regime),
        class_separation = smoke_logit_margin(rbind(logits_train, logits_test)),
        seed_data = as.integer(seed_data + attempt - 1L)
      )
    ))
  }

  stop("Failed to generate non-degenerate classification data with all classes represented.")
}

synthetic_smoke_methods <- function(cuda_ok = FALSE,
                                    n_train,
                                    p,
                                    q,
                                    K = NA_integer_,
                                    include_gpu = FALSE,
                                    include_r = FALSE,
                                    include_pls_pkg = FALSE) {
  q_eff <- if (is.na(K)) q else K
  allow_pls_pkg <- isTRUE(include_pls_pkg) &&
    isTRUE(n_train <= 1000L && p <= 200L && q_eff <= 200L && (is.na(K) || K <= 20L))
  allow_r <- isTRUE(include_r) &&
    isTRUE(n_train <= 500L && p <= 100L && q_eff <= 100L && (is.na(K) || K <= 10L))

  out <- data.table::rbindlist(list(
    data.table::data.table(engine = "Rcpp", algorithm = "plssvd", svd_method = "irlba", fast_profile = "default"),
    data.table::data.table(engine = "Rcpp", algorithm = "plssvd", svd_method = "cpu_rsvd", fast_profile = "default"),
    data.table::data.table(engine = "Rcpp", algorithm = "simpls", svd_method = "irlba", fast_profile = "default"),
    data.table::data.table(engine = "Rcpp", algorithm = "simpls", svd_method = "cpu_rsvd", fast_profile = "default"),
    data.table::data.table(engine = "Rcpp", algorithm = "simpls_fast", svd_method = "irlba", fast_profile = "incdefl"),
    data.table::data.table(engine = "Rcpp", algorithm = "simpls_fast", svd_method = "cpu_rsvd", fast_profile = "incdefl")
  ))

  if (isTRUE(allow_r)) {
    out <- data.table::rbindlist(list(
      out,
      data.table::data.table(engine = "R", algorithm = "plssvd", svd_method = "irlba", fast_profile = "default"),
      data.table::data.table(engine = "R", algorithm = "plssvd", svd_method = "cpu_rsvd", fast_profile = "default"),
      data.table::data.table(engine = "R", algorithm = "simpls", svd_method = "irlba", fast_profile = "default"),
      data.table::data.table(engine = "R", algorithm = "simpls", svd_method = "cpu_rsvd", fast_profile = "default"),
      data.table::data.table(engine = "R", algorithm = "simpls_fast", svd_method = "cpu_rsvd", fast_profile = "incdefl"),
      data.table::data.table(engine = "R", algorithm = "simpls_fast", svd_method = "irlba", fast_profile = "incdefl")
    ), fill = TRUE)
  }

  if (isTRUE(allow_pls_pkg)) {
    out <- data.table::rbindlist(list(
      out,
      data.table::data.table(engine = "pls_pkg", algorithm = "simpls", svd_method = "pls_pkg", fast_profile = "pls_pkg")
    ), fill = TRUE)
  }

  if (isTRUE(cuda_ok) && isTRUE(include_gpu)) {
    out <- data.table::rbindlist(list(
      out,
      data.table::data.table(engine = "GPU", algorithm = "plssvd", svd_method = "gpu_native", fast_profile = "gpu_native"),
      data.table::data.table(engine = "GPU", algorithm = "simpls_fast", svd_method = "gpu_native", fast_profile = "gpu_native")
    ), fill = TRUE)
  }

  unique(out)[, method_id := paste(engine, algorithm, svd_method, fast_profile, sep = "_")][]
}

synthetic_smoke_validate_dataset <- function(ds) {
  stopifnot(all(is.finite(ds$Xtrain)), all(is.finite(ds$Xtest)))
  if (identical(ds$task_type, "classification")) {
    if (!is.factor(ds$Ytrain) || !is.factor(ds$Ytest)) {
      stop("Classification dataset must use factor responses")
    }
    if (nlevels(droplevels(ds$Ytrain)) < 2L || nlevels(droplevels(ds$Ytest)) < 2L) {
      stop("Classification dataset is degenerate")
    }
  } else {
    stopifnot(all(is.finite(ds$Ytrain)), all(is.finite(ds$Ytest)))
  }
  if (!isTRUE(is.list(ds$meta))) stop("Dataset metadata missing")
  if (!is.finite(ds$meta$effective_ncomp) || is.na(ds$meta$effective_ncomp)) {
    stop("Dataset metadata missing effective_ncomp")
  }
  invisible(TRUE)
}

synthetic_smoke_pls_pkg_fit <- function(ds, effective_ncomp) {
  if (!requireNamespace("pls", quietly = TRUE)) {
    stop("pls package not available")
  }

  if (identical(ds$task_type, "classification")) {
    x_names <- paste0("x_", seq_len(ncol(ds$Xtrain)))
    Xtr <- ds$Xtrain
    colnames(Xtr) <- x_names
    Ymm <- smoke_dummy_matrix(ds$Ytrain)
    class_map <- stats::setNames(levels(ds$Ytrain), colnames(Ymm))
    df_train <- data.frame(Ymm, Xtr, check.names = FALSE)
    form <- stats::as.formula(paste0("cbind(", paste(colnames(Ymm), collapse = ","), ") ~ ."))

    t0 <- proc.time()[3]
    mdl <- pls::plsr(form, data = df_train, ncomp = as.integer(effective_ncomp), method = "simpls", scale = FALSE, validation = "none")
    train_ms <- (proc.time()[3] - t0) * 1000

    Xte <- as.data.frame(ds$Xtest)
    colnames(Xte) <- x_names
    t1 <- proc.time()[3]
    pred_arr <- predict(mdl, newdata = Xte, ncomp = as.integer(effective_ncomp))
    predict_ms <- (proc.time()[3] - t1) * 1000

    pred_mat <- pred_arr[, , 1, drop = TRUE]
    pred_idx <- max.col(pred_mat, ties.method = "first")
    pred <- factor(class_map[colnames(Ymm)[pred_idx]], levels = levels(ds$Ytrain))

    list(
      train_ms = as.numeric(train_ms),
      predict_ms = as.numeric(predict_ms),
      total_ms = as.numeric(train_ms + predict_ms),
      accuracy = mean(as.character(pred) == as.character(ds$Ytest), na.rm = TRUE),
      Q2 = NA_real_,
      train_R2 = NA_real_,
      model_size_mb = as.numeric(utils::object.size(mdl)) / (1024^2)
    )
  } else {
    x_names <- paste0("x_", seq_len(ncol(ds$Xtrain)))
    Xtr <- ds$Xtrain
    colnames(Xtr) <- x_names
    ymat <- as.matrix(ds$Ytrain)
    df_train <- data.frame(ymat, Xtr, check.names = FALSE)
    y_cols <- colnames(df_train)[seq_len(ncol(ymat))]
    form <- stats::as.formula(paste0("cbind(", paste(y_cols, collapse = ","), ") ~ ."))

    t0 <- proc.time()[3]
    mdl <- pls::plsr(form, data = df_train, ncomp = as.integer(effective_ncomp), method = "simpls", scale = FALSE, validation = "none")
    train_ms <- (proc.time()[3] - t0) * 1000

    Xte <- as.data.frame(ds$Xtest)
    colnames(Xte) <- x_names
    t1 <- proc.time()[3]
    pred_arr <- predict(mdl, newdata = Xte, ncomp = as.integer(effective_ncomp))
    predict_ms <- (proc.time()[3] - t1) * 1000

    pred_test <- as.matrix(pred_arr[, , 1, drop = TRUE])
    fit_arr <- predict(mdl, newdata = as.data.frame(Xtr), ncomp = as.integer(effective_ncomp))
    fit_train <- as.matrix(fit_arr[, , 1, drop = TRUE])

    list(
      train_ms = as.numeric(train_ms),
      predict_ms = as.numeric(predict_ms),
      total_ms = as.numeric(train_ms + predict_ms),
      accuracy = NA_real_,
      Q2 = smoke_r2_score(ds$Ytest, pred_test),
      train_R2 = smoke_r2_score(ds$Ytrain, fit_train),
      model_size_mb = as.numeric(utils::object.size(mdl)) / (1024^2)
    )
  }
}

synthetic_smoke_fastpls_fit <- function(ds, cfg, effective_ncomp, seed_fit) {
  common_args <- list(
    Xtrain = ds$Xtrain,
    Ytrain = ds$Ytrain,
    ncomp = as.integer(effective_ncomp),
    scaling = "centering",
    seed = as.integer(seed_fit),
    fit = TRUE,
    proj = FALSE
  )

  if (identical(cfg$engine, "GPU")) {
    t0 <- proc.time()[3]
    model <- if (identical(cfg$algorithm, "plssvd")) {
      do.call(fastPLS::plssvd_gpu, c(common_args, list(
        rsvd_oversample = 10L,
        rsvd_power = 1L,
        svds_tol = 0
      )))
    } else {
      do.call(fastPLS::simpls_gpu, c(common_args, list(
        rsvd_oversample = 10L,
        rsvd_power = 1L,
        svds_tol = 0
      )))
    }
    train_ms <- (proc.time()[3] - t0) * 1000
  } else {
    fn <- if (identical(cfg$engine, "R")) fastPLS::pls_r else fastPLS::pls
    args <- c(common_args, list(
      method = cfg$algorithm,
      svd.method = cfg$svd_method,
      rsvd_oversample = 10L,
      rsvd_power = 1L,
      svds_tol = 0,
      irlba_svtol = 1e-6,
      rsvd_tol = 0
    ))
    if (identical(cfg$algorithm, "simpls_fast")) {
      args <- c(args, list(
        fast_incremental = TRUE,
        fast_inc_iters = 2L,
        fast_defl_cache = TRUE,
        fast_center_t = FALSE,
        fast_reorth_v = FALSE,
        fast_block = 8L
      ))
    }
    t0 <- proc.time()[3]
    model <- do.call(fn, smoke_filter_call_args(fn, args))
    train_ms <- (proc.time()[3] - t0) * 1000
  }

  t1 <- proc.time()[3]
  pred <- predict(model, newdata = ds$Xtest, Ytest = ds$Ytest, proj = FALSE)
  predict_ms <- (proc.time()[3] - t1) * 1000

  if (identical(ds$task_type, "classification")) {
    pred_lab <- pred$Ypred
    if (is.data.frame(pred_lab)) pred_lab <- pred_lab[[ncol(pred_lab)]]
    accuracy <- mean(as.character(pred_lab) == as.character(ds$Ytest), na.rm = TRUE)
    train_R2 <- NA_real_
    Q2 <- NA_real_
  } else {
    Q2 <- tail(pred$Q2Y, 1L)
    train_R2 <- if (!is.null(model$R2Y)) tail(model$R2Y, 1L) else {
      pred_train <- predict(model, newdata = ds$Xtrain, Ytest = ds$Ytrain, proj = FALSE)
      tail(pred_train$Q2Y, 1L)
    }
    accuracy <- NA_real_
  }

  list(
    train_ms = as.numeric(train_ms),
    predict_ms = as.numeric(predict_ms),
    total_ms = as.numeric(train_ms + predict_ms),
    accuracy = as.numeric(accuracy),
    Q2 = as.numeric(Q2),
    train_R2 = as.numeric(train_R2),
    model_size_mb = as.numeric(utils::object.size(model)) / (1024^2)
  )
}

synthetic_smoke_run_method <- function(ds, cfg, effective_ncomp, seed_fit) {
  if (identical(cfg$engine, "pls_pkg")) {
    synthetic_smoke_pls_pkg_fit(ds, effective_ncomp = effective_ncomp)
  } else {
    synthetic_smoke_fastpls_fit(ds, cfg, effective_ncomp = effective_ncomp, seed_fit = seed_fit)
  }
}

synthetic_smoke_metric_name <- function(task_type) {
  if (identical(task_type, "classification")) "accuracy" else "Q2"
}

synthetic_smoke_metric_value <- function(task_type, fit_res) {
  if (identical(task_type, "classification")) fit_res$accuracy else fit_res$Q2
}

synthetic_smoke_manifest_lines <- function(raw_dt, cuda_ok, timing_reps, out_dir) {
  ok_rows <- raw_dt[status == "ok"]
  methods_run <- if (nrow(ok_rows)) unique(ok_rows$method_id) else character()
  skipped_tbl <- raw_dt[status != "ok", .N, by = .(status, msg)][order(status, msg)]
  skipped_txt <- if (!nrow(skipped_tbl)) {
    "none"
  } else {
    paste(sprintf("%s x%d (%s)", skipped_tbl$status, skipped_tbl$N, skipped_tbl$msg), collapse = "; ")
  }
  c(
    "machine_name = Chiamaka remote PC",
    sprintf("actual_nodename = %s", Sys.info()[["nodename"]] %||% "unknown"),
    sprintf("gpu_available = %s", if (isTRUE(cuda_ok)) "TRUE" else "FALSE"),
    sprintf("timing_reps = %d", as.integer(timing_reps)),
    sprintf("successful_runs = %d", nrow(ok_rows)),
    sprintf("methods_backends_run = %s", if (length(methods_run)) paste(methods_run, collapse = ", ") else "<none>"),
    "skipped_method_rules = pls_pkg only when n_train<=1000,p<=200,q<=200,K<=20; pure R only when n_train<=500,p<=100,q<=100,K<=10",
    sprintf("non_ok_runs = %s", skipped_txt),
    sprintf("output_dir = %s", normalizePath(out_dir, winslash = "/", mustWork = FALSE))
  )
}

synthetic_smoke_validate_raw <- function(raw_dt,
                                         expected_families = names(smoke_family_specs()),
                                         expected_noise_levels = smoke_noise_levels()) {
  required_families <- expected_families %||% names(smoke_family_specs())
  missing_families <- setdiff(required_families, unique(raw_dt$scenario_family))
  if (length(missing_families)) {
    stop("Missing synthetic scenario families in raw output: ", paste(missing_families, collapse = ", "))
  }

  if (anyNA(raw_dt$effective_ncomp)) {
    stop("Raw output contains missing effective_ncomp values.")
  }

  analysis_expect <- list(
    sim_reg_n_p50 = list(analysis = "n_train", values = smoke_grid_n_train()),
    sim_reg_n_p500 = list(analysis = "n_train", values = smoke_grid_n_train()),
    sim_reg_n_p1000_q1000_ncomp500 = list(analysis = "n_train", values = smoke_grid_n_train()),
    sim_reg_p_sweep = list(analysis = "p", values = smoke_grid_p()),
    sim_reg_q_sweep = list(analysis = "q", values = smoke_grid_q()),
    sim_reg_noise_sweep = list(analysis = "noise_regime", values = expected_noise_levels)
  )
  for (fam in intersect(names(analysis_expect), required_families)) {
    sub <- raw_dt[scenario_family == fam]
    expected_analysis <- analysis_expect[[fam]]$analysis
    if (!nrow(sub) || !all(unique(sub$analysis) == expected_analysis)) {
      stop("Scenario family ", fam, " is missing expected analysis ", expected_analysis)
    }
    if (identical(expected_analysis, "noise_regime")) {
      observed_vals <- unique(as.character(sub$analysis_value))
      observed_vals <- observed_vals[order(match(observed_vals, smoke_noise_levels()))]
      expected_vals <- analysis_expect[[fam]]$values
    } else {
      observed_vals <- sort(unique(as.integer(sub$analysis_value)))
      expected_vals <- sort(as.integer(analysis_expect[[fam]]$values))
    }
    if (!identical(observed_vals, expected_vals)) {
      stop("Scenario family ", fam, " is missing requested grid values.")
    }
  }

  axis_checks <- data.table::rbindlist(list(
    raw_dt[scenario_family %in% c("sim_reg_n_p50", "sim_reg_n_p500", "sim_reg_n_p1000_q1000_ncomp500"), .(scenario_family, axis = "n_train", value = n_train)],
    raw_dt[scenario_family == "sim_reg_p_sweep", .(scenario_family, axis = "p", value = p)],
    raw_dt[scenario_family == "sim_reg_q_sweep", .(scenario_family, axis = "q", value = q)],
    raw_dt[scenario_family == "sim_reg_noise_sweep", .(scenario_family, axis = "noise_regime", value = noise_rank)]
  ), fill = TRUE)
  collapsed <- axis_checks[, .(n_unique = data.table::uniqueN(value)), by = .(scenario_family, axis)][n_unique <= 1L]
  if (nrow(collapsed)) {
    stop("At least one scenario family has only one unique x-axis value: ", paste(collapsed$scenario_family, collapse = ", "))
  }

  pq_expected <- intersect(required_families, c("sim_reg_p_sweep", "sim_reg_q_sweep"))
  if (length(pq_expected)) {
    pq <- raw_dt[scenario_family %in% pq_expected]
    if (pq[, data.table::uniqueN(effective_ncomp)] <= 1L) {
      stop("All p/q sweep runs collapsed to the same effective dimensionality unexpectedly.")
    }
  }

  invisible(TRUE)
}
