set.seed(20260914)

n_train_per_class <- 45L
n_test_per_class <- 15L
p <- 40L
n <- 2L * (n_train_per_class + n_test_per_class)

labels <- factor(
    rep(c("reference", "case"), each = n / 2L),
    levels = c("reference", "case")
)
latent <- matrix(rnorm(n * 3L), nrow = n, ncol = 3L)
loadings <- matrix(rnorm(3L * p), nrow = 3L, ncol = p)
X <- latent %*% loadings + matrix(rnorm(n * p, sd = 0.65), nrow = n)

case_rows <- labels == "case"
X[case_rows, seq_len(8L)] <- X[case_rows, seq_len(8L)] +
    matrix(rep(seq(0.8, 1.5, length.out = 8L), each = sum(case_rows)),
           nrow = sum(case_rows))

feature_names <- sprintf("marker_%02d", seq_len(p))
colnames(X) <- feature_names
rownames(X) <- sprintf("sample_%03d", seq_len(n))

training_rows <- c(seq_len(n_train_per_class),
                   n / 2L + seq_len(n_train_per_class))
test_rows <- setdiff(seq_len(n), training_rows)

fastpls_example <- list(
    X_train = X[training_rows, , drop = FALSE],
    y_train = droplevels(labels[training_rows]),
    X_test = X[test_rows, , drop = FALSE],
    y_test = droplevels(labels[test_rows]),
    feature_names = feature_names
)

dir.create("data", showWarnings = FALSE)
save(fastpls_example, file = "data/fastpls_example.rda", compress = "xz")
