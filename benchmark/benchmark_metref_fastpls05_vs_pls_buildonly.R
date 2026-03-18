suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
  library(pls)
  library(data.table)
  library(ggplot2)
})

set.seed(as.integer(Sys.getenv("FASTPLS_SEED", "123")))
reps <- as.integer(Sys.getenv("FASTPLS_REPS", "10"))
if (is.na(reps) || reps < 1L) reps <- 10L
ncomp_env <- Sys.getenv("FASTPLS_NCOMP_LIST", "2,5,10,15,20,22,30,50,100")
ncomp_list <- as.integer(strsplit(ncomp_env, ",", fixed = TRUE)[[1]])
ncomp_list <- sort(unique(ncomp_list[is.finite(ncomp_list) & ncomp_list >= 1L]))
if (!length(ncomp_list)) ncomp_list <- c(2L, 5L, 10L, 15L, 20L, 22L, 30L, 50L, 100L)

out_dir <- Sys.getenv("FASTPLS_OUTDIR", "metref_fastpls050_vs_pls_buildonly")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Data from KODAMA
if (!"MetRef" %in% data(package = "KODAMA")$results[, "Item"]) {
  stop("MetRef dataset not found in KODAMA package")
}
data("MetRef", package = "KODAMA")
X <- MetRef$data
X <- X[, colSums(X) != 0, drop = FALSE]
X <- normalization(X)$newXtrain
Y <- as.factor(MetRef$donor)

# Fixed split for reproducibility (same split for all methods)
ss <- sample(seq_len(nrow(X)), 100)
Xtrain <- as.matrix(X[-ss, , drop = FALSE])
Ytrain <- Y[-ss]
Xtest <- as.matrix(X[ss, , drop = FALSE])
Ytest <- Y[ss]

# helper for accuracy
acc <- function(truth, pred) mean(as.character(truth) == as.character(pred), na.rm = TRUE)

# fastPLS runners (build time only in timed block)
fit_fastpls <- function(method_name, ncomp_eff, fast_args = list()) {
  call_args <- c(
    list(
      Xtrain = Xtrain,
      Ytrain = Ytrain,
      ncomp = ncomp_eff,
      method = method_name,
      svd.method = "cpu_exact",
      scaling = "centering"
    ),
    fast_args
  )
  t0 <- proc.time()[3]
  model <- do.call(fastPLS::pls, call_args)
  elapsed_ms <- (proc.time()[3] - t0) * 1000

  pred <- predict(model, Xtest, Ytest = Ytest, proj = FALSE)$Ypred[[1]]
  list(time_ms = elapsed_ms, accuracy = acc(Ytest, pred))
}

# pls package runner: classify by one-vs-rest dummy regression + argmax
fit_pls_pkg <- function(ncomp_eff) {
  class_lev <- levels(Ytrain)
  Ymm <- model.matrix(~ Ytrain - 1)
  colnames(Ymm) <- paste0("cls_", seq_len(ncol(Ymm)))
  class_map <- setNames(class_lev, colnames(Ymm))
  x_names <- paste0("x_", seq_len(ncol(Xtrain)))
  colnames(Xtrain) <- x_names
  df_train <- data.frame(Ymm, Xtrain, check.names = FALSE)
  form <- as.formula(paste0("cbind(", paste(colnames(Ymm), collapse = ","), ") ~ ."))

  t0 <- proc.time()[3]
  mdl <- pls::plsr(form, data = df_train, ncomp = ncomp_eff, method = "simpls", scale = FALSE, validation = "none")
  elapsed_ms <- (proc.time()[3] - t0) * 1000

  Xtest_df <- as.data.frame(Xtest)
  colnames(Xtest_df) <- x_names
  pred_arr <- predict(mdl, newdata = Xtest_df, ncomp = ncomp_eff)
  pred_mat <- pred_arr[, , 1, drop = FALSE]
  pred_idx <- apply(pred_mat, 1, which.max)
  lev <- colnames(Ymm)
  pred <- factor(class_map[lev[pred_idx]], levels = class_lev)

  list(time_ms = elapsed_ms, accuracy = acc(Ytest, pred))
}

plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), nlevels(Ytrain))

methods <- list(
  list(method_id = "fastPLS_simpls", runner = function(nc) fit_fastpls("simpls", nc)),
  list(method_id = "fastPLS_plssvd", runner = function(nc) fit_fastpls("plssvd", min(nc, plssvd_cap))),
  list(method_id = "fastPLS_simpls_fast", runner = function(nc) fit_fastpls("simpls_fast", nc)),
  list(method_id = "fastPLS_simpls_fast_incdefl", runner = function(nc) fit_fastpls(
    "simpls_fast", nc,
    fast_args = list(
      fast_incremental = TRUE,
      fast_inc_iters = 2L,
      fast_defl_cache = TRUE,
      fast_center_t = FALSE,
      fast_reorth_v = FALSE,
      fast_block = 8L
    )
  )),
  list(method_id = "pls_pkg_simpls", runner = function(nc) fit_pls_pkg(nc))
)

rows <- vector("list", length(methods) * reps * length(ncomp_list))
k <- 1L
for (m in methods) {
  for (nc in ncomp_list) {
    cat(sprintf("Running %s (ncomp=%d) ...\n", m$method_id, nc))
    flush.console()
    for (r in seq_len(reps)) {
      out <- tryCatch(m$runner(nc), error = function(e) list(time_ms = NA_real_, accuracy = NA_real_, err = conditionMessage(e)))
      rows[[k]] <- data.table(
        method_id = m$method_id,
        rep = r,
        ncomp_requested = nc,
        ncomp_effective = if (m$method_id == "fastPLS_plssvd") min(nc, plssvd_cap) else nc,
        train_time_ms = out$time_ms,
        accuracy = out$accuracy,
        status = if (!is.null(out$err)) out$err else "ok"
      )
      k <- k + 1L
    }
  }
}
res <- rbindlist(rows, fill = TRUE)

sumtab <- res[, .(
  reps_ok = sum(status == "ok", na.rm = TRUE),
  train_time_ms_median = median(train_time_ms, na.rm = TRUE),
  train_time_ms_mean = mean(train_time_ms, na.rm = TRUE),
  train_time_ms_sd = sd(train_time_ms, na.rm = TRUE),
  accuracy_median = median(accuracy, na.rm = TRUE),
  accuracy_mean = mean(accuracy, na.rm = TRUE)
), by = .(method_id, ncomp_requested, ncomp_effective)][order(train_time_ms_median)]

fwrite(res, file.path(out_dir, "metref_buildonly_replicates.csv"))
fwrite(sumtab, file.path(out_dir, "metref_buildonly_summary.csv"))

p1 <- ggplot(sumtab, aes(x = ncomp_effective, y = train_time_ms_median, color = method_id)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  labs(title = "MetRef - Model Building Time vs Components", x = "ncomp (effective)", y = "Median Train Time (ms)", color = "Method") +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "metref_buildonly_train_time_vs_ncomp.png"), p1, width = 11, height = 6.5, dpi = 150)

p2 <- ggplot(sumtab, aes(x = ncomp_effective, y = accuracy_mean, color = method_id)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  labs(title = "MetRef - Accuracy vs Components", x = "ncomp (effective)", y = "Mean Accuracy", color = "Method") +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "metref_buildonly_accuracy_vs_ncomp.png"), p2, width = 11, height = 6.5, dpi = 150)

writeLines(capture.output(print(sumtab)), file.path(out_dir, "metref_buildonly_summary.txt"))
cat("Done. Output dir:", out_dir, "\n")
print(sumtab)
