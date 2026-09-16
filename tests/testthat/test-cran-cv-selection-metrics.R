cv_selection_data <- function() {
    set.seed(814L)
    observations <- 36L
    predictors <- matrix(rnorm(observations * 10L), observations, 10L)
    classes <- factor(rep(c("a", "b", "c"), each = 12L))
    predictors[, 1L] <- predictors[, 1L] + as.integer(classes) * 0.5
    response <- 1.4 * predictors[, 1L] - 0.7 * predictors[, 2L] +
        rnorm(observations, sd = 0.8)
    multivariate <- cbind(
        response,
        -0.5 * predictors[, 1L] + predictors[, 3L] +
            rnorm(observations, sd = 0.7),
        predictors[, 4L] + 0.4 * predictors[, 5L] +
            rnorm(observations, sd = 0.6)
    )
    list(
        x = predictors,
        classification = classes,
        univariate = response,
        multivariate = multivariate
    )
}

cv_classification_metrics <- c(
    "auto", "accuracy", "balanced_accuracy", "lift_accuracy",
    "macro_precision", "macro_recall", "macro_f1", "kappa", "R2Y",
    "Q2Y"
)

cv_regression_metrics <- c(
    "auto", "R2Y", "Q2Y", "RMSD", "MAE", "MAPE_percent", "RPD",
    "Pearson_r", "Spearman_r"
)

cv_resolved_metric <- function(metric, classification) {
    if (!identical(metric, "auto")) {
        return(metric)
    }
    if (classification) "accuracy" else "RMSD"
}

cv_expected_index <- function(values, metric, classification) {
    finite <- is.finite(values)
    if (!any(finite)) {
        return(1L)
    }
    metric <- tolower(cv_resolved_metric(metric, classification))
    if (metric %in% tolower(c("RMSD", "MAE", "MAPE_percent"))) {
        return(which.min(ifelse(finite, values, Inf)))
    }
    which.max(ifelse(finite, values, -Inf))
}

cv_run_single_selection <- function(x, y, metric, classification) {
    arguments <- list(
        Xdata = x,
        Ydata = y,
        ncomp = 1:3,
        kfold = 3L,
        method = "simpls",
        backend = "cpu",
        selection = metric,
        seed = 91L,
        fit = FALSE
    )
    if (classification) {
        arguments$classifier <- "lda"
    }
    do.call(pls.single.cv, arguments)
}

cv_run_double_selection <- function(x, y, metric, classification) {
    arguments <- list(
        Xdata = x,
        Ydata = y,
        ncomp = 1:3,
        kfold_inner = 3L,
        kfold_outer = 3L,
        runn = 1L,
        method = "simpls",
        backend = "cpu",
        selection = metric,
        seed = 91L
    )
    if (classification) {
        arguments$classifier <- "lda"
    }
    do.call(pls.double.cv, arguments)
}

test_that("single CV applies every documented metric in the right direction", {
    data <- cv_selection_data()
    tasks <- list(
        classification = list(
            response = data$classification,
            metrics = cv_classification_metrics,
            classification = TRUE
        ),
        univariate = list(
            response = data$univariate,
            metrics = cv_regression_metrics,
            classification = FALSE
        ),
        multivariate = list(
            response = data$multivariate,
            metrics = cv_regression_metrics,
            classification = FALSE
        )
    )

    for (task in tasks) {
        for (metric in task$metrics) {
            result <- cv_run_single_selection(
                data$x, task$response, metric, task$classification
            )
            expected <- cv_expected_index(
                result$selection_values, metric, task$classification
            )
            expect_identical(result$best_index, as.integer(expected))
            expect_identical(
                result$best_ncomp,
                as.integer(result$ncomp[[expected]])
            )
            expect_equal(
                result$best_metric_value,
                result$selection_values[[expected]],
                tolerance = 0
            )
            expect_identical(
                result$best_metric_name,
                cv_resolved_metric(metric, task$classification)
            )
            if (identical(metric, "R2Y")) {
                expect_false(is.null(result$Yfit))
            }
        }
    }
})

test_that("nested CV applies every metric to each inner component path", {
    data <- cv_selection_data()
    tasks <- list(
        classification = list(
            response = data$classification,
            metrics = cv_classification_metrics,
            classification = TRUE
        ),
        univariate = list(
            response = data$univariate,
            metrics = cv_regression_metrics,
            classification = FALSE
        ),
        multivariate = list(
            response = data$multivariate,
            metrics = cv_regression_metrics,
            classification = FALSE
        )
    )

    for (task in tasks) {
        for (metric in task$metrics) {
            result <- cv_run_double_selection(
                data$x, task$response, metric, task$classification
            )
            run <- result$results[[1L]]
            expect_true(is.finite(run$metric_value))
            expect_identical(
                run$metric_name,
                cv_resolved_metric(metric, task$classification)
            )
            for (fold_index in seq_along(run$inner)) {
                inner <- run$inner[[fold_index]]
                values <- inner$selection_values
                if (is.null(values)) {
                    values <- inner$metric_value
                }
                expected <- cv_expected_index(
                    values, metric, task$classification
                )
                expect_identical(
                    run$best_ncomp[[fold_index]],
                    as.integer(inner$ncomp[[expected]])
                )
            }
        }
    }
})

test_that("selection rejects metrics from the wrong task before fitting", {
    data <- cv_selection_data()
    for (fun in list(pls.single.cv, pls.double.cv)) {
        expect_error(
            fun(
                data$x, data$univariate,
                ncomp = 1:2,
                selection = "balanced_accuracy",
                backend = "cpu"
            ),
            "not valid for regression"
        )
        expect_error(
            fun(
                data$x, data$classification,
                ncomp = 1:2,
                selection = "RMSD",
                backend = "cpu"
            ),
            "not valid for classification"
        )
    }
})

test_that("response-wise reporting does not change multivariate selection", {
    data <- cv_selection_data()
    aggregate <- pls.single.cv(
        data$x, data$multivariate,
        ncomp = 1:3,
        kfold = 3L,
        selection = "MAE",
        backend = "cpu",
        seed = 29L,
        bycol = FALSE
    )
    response_wise <- pls.single.cv(
        data$x, data$multivariate,
        ncomp = 1:3,
        kfold = 3L,
        selection = "MAE",
        backend = "cpu",
        seed = 29L,
        bycol = TRUE
    )
    expect_identical(response_wise$best_ncomp, aggregate$best_ncomp)
    expect_equal(
        response_wise$selection_values,
        aggregate$selection_values,
        tolerance = 0
    )
    per_response <- response_wise$metrics$cross_validated[[1L]]$per_response
    expect_false(is.null(per_response))
})

test_that("metric selection also ranks classifier and scaling grids", {
    data <- cv_selection_data()
    classification <- pls.single.cv(
        data$x, data$classification,
        ncomp = 1:2,
        kfold = 3L,
        classifier = c("argmax", "lda"),
        selection = "macro_f1",
        backend = "cpu",
        seed = 7L
    )
    ok <- classification$tuning_summary$status == "ok"
    expected <- which(ok)[which.max(
        classification$tuning_summary$best_metric_value[ok]
    )]
    expect_identical(classification$best_grid_id, expected)

    regression <- pls.double.cv(
        data$x, data$multivariate,
        ncomp = 1:2,
        kfold_inner = 3L,
        kfold_outer = 3L,
        scaling = c("centering", "autoscaling"),
        selection = "MAE",
        backend = "cpu",
        seed = 7L
    )
    for (inner in regression$results[[1L]]$inner) {
        ok <- inner$tuning_summary$status == "ok"
        expected <- which(ok)[which.min(
            inner$tuning_summary$best_metric_value[ok]
        )]
        expect_identical(inner$best_grid_id, expected)
    }
})

test_that("nested macro recall shares the compiled balanced-accuracy route", {
    data <- cv_selection_data()
    arguments <- list(
        Xdata = data$x,
        Ydata = data$classification,
        ncomp = 1:3,
        kfold_inner = 3L,
        kfold_outer = 3L,
        classifier = "lda",
        backend = "cpu",
        seed = 37L
    )
    balanced <- do.call(
        pls.double.cv,
        c(arguments, list(selection = "balanced_accuracy"))
    )
    recall <- do.call(
        pls.double.cv,
        c(arguments, list(selection = "macro_recall"))
    )
    expect_identical(recall$Ypred, balanced$Ypred)
    expect_identical(
        recall$results[[1L]]$best_ncomp,
        balanced$results[[1L]]$best_ncomp
    )
    expect_equal(
        recall$results[[1L]]$metric_value,
        balanced$results[[1L]]$metric_value,
        tolerance = 0
    )
    expect_identical(recall$results[[1L]]$metric_name, "macro_recall")
})
