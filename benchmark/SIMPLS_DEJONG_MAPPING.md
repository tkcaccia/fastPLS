# Accelerated SIMPLS mapping to the de Jong update

This executable-style pseudocode documents the deterministic SIMPLS path used
for estimator-preservation validation. The low-rank solver supplies only the
current dominant direction. The SIMPLS normalization, orthogonalization,
deflation, coefficient update, and component order remain sequential.

```r
simpls_component_path <- function(X, Y, ncomp, dominant_left) {
  X <- sweep(X, 2L, colMeans(X), "-")
  Y <- sweep(Y, 2L, colMeans(Y), "-")
  S <- crossprod(X, Y)

  R <- matrix(0, ncol(X), ncomp)
  Q <- matrix(0, ncol(Y), ncomp)
  V <- matrix(0, ncol(X), ncomp)
  B <- matrix(0, ncol(X), ncol(Y))
  Yhat <- matrix(0, nrow(X), ncol(Y))

  for (component in seq_len(ncomp)) {
    # de Jong direction extraction from the current deflated cross-covariance.
    r <- dominant_left(S)

    # Standard SIMPLS score normalization and loadings.
    t <- X %*% r
    t_norm <- sqrt(drop(crossprod(t)))
    t <- t / t_norm
    r <- r / t_norm
    p <- drop(crossprod(X, t))
    q <- drop(crossprod(Y, t))

    # Standard SIMPLS orthogonalization and rank-one deflation.
    v <- p
    if (component > 1L) {
      previous <- V[, seq_len(component - 1L), drop = FALSE]
      v <- v - previous %*% crossprod(previous, v)
    }
    v <- v / sqrt(drop(crossprod(v)))
    S <- S - v %*% crossprod(v, S)

    # Incremental execution: algebraically identical to R[,1:k] %*% t(Q[,1:k]).
    R[, component] <- r
    Q[, component] <- q
    V[, component] <- v
    B <- B + r %*% t(q)
    Yhat <- Yhat + t %*% t(q)
  }

  list(R = R, Q = Q, V = V, B = B, fitted_centered = Yhat)
}
```

## Mapping of implementation optimizations

| Optimization | de Jong quantity preserved | Change in execution |
|---|---|---|
| One maximal component path | Components 1 to `k` in their original order | Requested prefixes are snapshots rather than independent refits |
| Cached `X'X` for eligible tall matrices | `p = X't`, `||t||`, and the accepted `r` | Reuses an algebraically equivalent cross-product |
| Cached deflation row product | `S <- S - v(v'S)` | Evaluates `v'S` once before the rank-one update |
| Incremental coefficient update | `B_k = R_k Q_k'` | Uses `B_k = B_{k-1} + r_k q_k'` |
| Incremental fitted-response update | `Yhat_k = T_k Q_k'` | Uses `Yhat_k = Yhat_{k-1} + t_k q_k'` |
| Matrix-free cross-covariance | The global operator `S = X'Y` | Evaluates `S z = X'(Y z)` and `S' u = Y'(X u)` without storing `S` |

For deterministic IRLBA validation, each deflated component requests a fresh
IRLBA direction from the current operator. Randomized workspace reuse belongs
only to the rSVD route and is evaluated separately as an approximation.
