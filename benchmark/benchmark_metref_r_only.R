#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
  library(data.table)
  library(ggplot2)
  library(pls)
})

set.seed(as.integer(Sys.getenv("FASTPLS_SEED", "123")))

out_dir <- Sys.getenv("FASTPLS_OUTDIR", "metref_r_only_benchmark")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ncomp_grid <- as.integer(strsplit(
  Sys.getenv("FASTPLS_NCOMP_LIST", "2,5,10,20,30,40,50,75,100"),
  ",",
  fixed = TRUE
)[[1]])
ncomp_grid <- sort(unique(ncomp_grid[is.finite(ncomp_grid) & ncomp_grid >= 1L]))
if (!length(ncomp_grid)) ncomp_grid <- c(2L, 5L, 10L, 20L, 30L, 40L, 50L, 75L, 100L)

reps <- as.integer(Sys.getenv("FASTPLS_REPS", "3"))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L

test_n <- as.integer(Sys.getenv("FASTPLS_TEST_N", "100"))
if (!is.finite(test_n) || is.na(test_n) || test_n < 1L) test_n <- 100L

data(MetRef, package = "KODAMA")
u <- MetRef$data
u <- u[, colSums(u) != 0, drop = FALSE]
u <- normalization(u)$newXtrain
y <- factor(MetRef$donor)

test_n <- min(test_n, floor(nrow(u) / 3))
idx_test <- sample(seq_len(nrow(u)), test_n)

Xtrain <- as.matrix(u[-idx_test, , drop = FALSE])
Ytrain <- y[-idx_test]
Xtest <- as.matrix(u[idx_test, , drop = FALSE])
Ytest <- y[idx_test]

plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), nlevels(Ytrain))

methods <- rbindlist(list(
  data.table(
    engine = "R",
    algorithm = rep(c("plssvd", "simpls", "simpls_fast"), each = 3L),
    svd_method = rep(c("arpack", "cpu_rsvd", "irlba"), times = 3L)
  ),
  data.table(
    engine = "pls_pkg",
    algorithm = "simpls",
    svd_method = "pls_pkg"
  )
), fill = TRUE)
methods[, method_id := ifelse(engine == "pls_pkg",
                              "pls_pkg_simpls",
                              paste(engine, algorithm, svd_method, sep = "_"))]

extract_pred_fastpls <- function(model_out) {
  yp <- model_out$Ypred
  if (is.data.frame(yp)) return(as.factor(yp[[1L]]))
  if (is.matrix(yp)) return(as.factor(yp[, 1L]))
  if (is.vector(yp)) return(as.factor(yp))
  if (length(dim(yp)) == 3L) {
    mat <- yp[, , 1L, drop = FALSE]
    cls <- apply(mat, 1L, which.max)
    lev <- model_out$lev
    return(factor(lev[cls], levels = lev))
  }
  stop("Unsupported fastPLS prediction format")
}

fit_pls_pkg <- function(ncomp) {
  class_lev <- levels(Ytrain)
  Ymm <- model.matrix(~ Ytrain - 1)
  colnames(Ymm) <- paste0("cls_", seq_len(ncol(Ymm)))
  x_names <- paste0("x_", seq_len(ncol(Xtrain)))

  Xtr <- Xtrain
  colnames(Xtr) <- x_names
  df_train <- data.frame(Ymm, Xtr, check.names = FALSE)
  form <- as.formula(paste0("cbind(", paste(colnames(Ymm), collapse = ","), ") ~ ."))

  mdl <- pls::plsr(
    form,
    data = df_train,
    ncomp = as.integer(ncomp),
    method = "simpls",
    scale = FALSE,
    validation = "none"
  )
  list(model = mdl, class_lev = class_lev, y_cols = colnames(Ymm), x_names = x_names)
}

predict_pls_pkg <- function(obj, ncomp) {
  Xte <- as.data.frame(Xtest)
  colnames(Xte) <- obj$x_names
  pa <- predict(obj$model, newdata = Xte, ncomp = as.integer(ncomp))
  pm <- pa[, , 1L, drop = FALSE]
  idx <- apply(pm, 1L, which.max)
  raw <- obj$y_cols[idx]
  pred_num <- as.integer(sub("^cls_", "", raw))
  factor(obj$class_lev[pred_num], levels = obj$class_lev)
}

fit_once <- function(engine, algorithm, svd_method, ncomp) {
  if (identical(engine, "pls_pkg")) {
    return(fit_pls_pkg(ncomp))
  }
  fastPLS::pls_r(
    Xtrain, Ytrain, Xtest, Ytest,
    ncomp = as.integer(ncomp),
    method = algorithm,
    svd.method = svd_method,
    scaling = "centering",
    fit = FALSE,
    proj = FALSE
  )
}

predict_once <- function(engine, fit_obj, ncomp) {
  if (identical(engine, "pls_pkg")) {
    return(predict_pls_pkg(fit_obj, ncomp))
  }
  extract_pred_fastpls(fit_obj)
}

rows <- vector("list", length = nrow(methods) * length(ncomp_grid) * reps)
k <- 1L

for (i in seq_len(nrow(methods))) {
  cfg <- methods[i]
  message(sprintf("Running %s (%d/%d)", cfg$method_id, i, nrow(methods)))

  for (nc in ncomp_grid) {
    if (identical(cfg$algorithm, "plssvd") && nc > plssvd_cap) {
      rows[[k]] <- data.table(
        dataset = "metref",
        rep = NA_integer_,
        ncomp = nc,
        engine = cfg$engine,
        algorithm = cfg$algorithm,
        svd_method = cfg$svd_method,
        method_id = cfg$method_id,
        status = "skipped_ncomp_above_plssvd_cap",
        train_ms = NA_real_,
        accuracy = NA_real_,
        xtrain_nrow = nrow(Xtrain),
        xtrain_ncol = ncol(Xtrain),
        ytrain_display_dim = nlevels(Ytrain),
        metric_name = "accuracy",
        msg = sprintf("plssvd cap=%d", plssvd_cap)
      )
      k <- k + 1L
      next
    }

    for (r in seq_len(reps)) {
      err <- NULL
      elapsed_ms <- NA_real_
      acc <- NA_real_

      gc(FALSE)
      t0 <- proc.time()[3L]
      fit <- tryCatch(
        fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, nc),
        error = function(e) {
          err <<- conditionMessage(e)
          NULL
        }
      )
      elapsed_ms <- (proc.time()[3L] - t0) * 1000

      if (is.null(err)) {
        pred <- tryCatch(
          predict_once(cfg$engine, fit, nc),
          error = function(e) {
            err <<- conditionMessage(e)
            NULL
          }
        )
        if (!is.null(pred)) {
          acc <- mean(pred == Ytest)
        }
      }

      rows[[k]] <- data.table(
        dataset = "metref",
        rep = r,
        ncomp = nc,
        engine = cfg$engine,
        algorithm = cfg$algorithm,
        svd_method = cfg$svd_method,
        method_id = cfg$method_id,
        status = if (is.null(err)) "ok" else paste0("error: ", err),
        train_ms = elapsed_ms,
        accuracy = acc,
        xtrain_nrow = nrow(Xtrain),
        xtrain_ncol = ncol(Xtrain),
        ytrain_display_dim = nlevels(Ytrain),
        metric_name = "accuracy",
        msg = if (is.null(err)) "" else err
      )
      k <- k + 1L
    }
  }
}

raw <- rbindlist(rows[seq_len(k - 1L)], fill = TRUE)
summary_dt <- raw[
  status == "ok",
  .(
    train_ms_median = median(train_ms, na.rm = TRUE),
    train_ms_mean = mean(train_ms, na.rm = TRUE),
    accuracy_median = median(accuracy, na.rm = TRUE),
    accuracy_mean = mean(accuracy, na.rm = TRUE),
    reps_ok = .N,
    xtrain_nrow = xtrain_nrow[1L],
    xtrain_ncol = xtrain_ncol[1L],
    ytrain_display_dim = ytrain_display_dim[1L],
    metric_name = metric_name[1L]
  ),
  by = .(dataset, ncomp, engine, algorithm, svd_method, method_id)
]

fwrite(raw, file.path(out_dir, "metref_r_only_raw.csv"))
fwrite(summary_dt, file.path(out_dir, "metref_r_only_summary.csv"))

plot_dt <- melt(
  summary_dt,
  id.vars = c("dataset", "ncomp", "engine", "algorithm", "svd_method", "method_id"),
  measure.vars = c("train_ms_median", "accuracy_median"),
  variable.name = "panel",
  value.name = "value"
)
plot_dt[, panel := factor(panel,
                          levels = c("train_ms_median", "accuracy_median"),
                          labels = c("Median train time (ms)", "Median accuracy"))]
plot_dt[, algorithm := factor(algorithm, levels = c("plssvd", "simpls", "simpls_fast"))]

p <- ggplot(
  plot_dt,
  aes(x = ncomp, y = value, color = svd_method, linetype = engine, shape = engine,
      group = interaction(method_id, panel))
) +
  geom_point(size = 2) +
  facet_grid(panel ~ algorithm, scales = "free_y") +
  scale_color_manual(
    values = c(arpack = "#7570b3", cpu_rsvd = "#d95f02", irlba = "#1b9e77", pls_pkg = "#666666"),
    drop = FALSE
  ) +
  scale_shape_manual(values = c(R = 16, pls_pkg = 15), drop = FALSE) +
  scale_linetype_manual(values = c(R = "solid", pls_pkg = "22"), drop = FALSE) +
  labs(
    title = sprintf(
      "MetRef R-only benchmark | train_n=%d, p=%d, classes=%d",
      nrow(Xtrain), ncol(Xtrain), nlevels(Ytrain)
    ),
    x = "Number of components",
    y = NULL,
    color = "Backend",
    shape = "Engine",
    linetype = "Engine"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

if (data.table::uniqueN(plot_dt$ncomp) > 1L) {
  p <- p + geom_line(linewidth = 0.8)
}

ggsave(
  filename = file.path(out_dir, "metref_r_only_benchmark.png"),
  plot = p,
  width = 13,
  height = 7.5,
  dpi = 170
)

cat("Wrote:\n")
cat(file.path(out_dir, "metref_r_only_raw.csv"), "\n")
cat(file.path(out_dir, "metref_r_only_summary.csv"), "\n")
cat(file.path(out_dir, "metref_r_only_benchmark.png"), "\n")
