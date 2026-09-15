// SPDX-License-Identifier: MIT

#include "reference_backend.hpp"

#include <fastpls/core/opls.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

template<class T>
void check_dense_filter() {
  fastpls::core::Matrix<T> predictors(24, 6);
  fastpls::core::Matrix<T> responses(24, 3);
  for (std::size_t row = 0; row < predictors.rows(); ++row) {
    for (std::size_t column = 0; column < predictors.columns(); ++column) {
      predictors(row, column) = static_cast<T>(
        std::sin(0.17 * (row + 1) * (column + 1)) + 0.1 * column
      );
    }
    responses(row, 0) = predictors(row, 0) + predictors(row, 1);
    responses(row, 1) = predictors(row, 2) - predictors(row, 3);
    responses(row, 2) = predictors(row, 4) + T(0.25) * predictors(row, 5);
  }
  ReferenceBackend<T> backend;
  const auto fitted = fastpls::core::fit_opls_filter(
    predictors, responses.view(), 2,
    fastpls::core::PredictorScaling::autoscaling, backend
  );
  assert(fitted.completed_components == 2);
  assert(fitted.weights.rows() == predictors.columns());
  assert(fitted.weights.columns() == 2);
  const auto applied = fastpls::core::apply_opls_filter(
    predictors, fitted.predictor_center.data(), fitted.predictor_scale.data(),
    fitted.predictor_center.size(), fitted.weights.view(),
    fitted.loadings.view(), backend
  );
  const T tolerance = std::is_same<T, float>::value ? T(2e-4) : T(2e-10);
  for (std::size_t index = 0; index < applied.size(); ++index) {
    assert(std::abs(applied.data()[index] - fitted.predictors.data()[index]) <
      tolerance);
  }
}

template<class T>
void check_label_filter() {
  fastpls::core::Matrix<T> predictors(30, 5);
  std::vector<std::size_t> labels(30);
  for (std::size_t row = 0; row < predictors.rows(); ++row) {
    labels[row] = row % 3;
    for (std::size_t column = 0; column < predictors.columns(); ++column) {
      predictors(row, column) = static_cast<T>(
        0.35 * labels[row] * (column + 1) +
        std::cos(0.11 * (row + 2) * (column + 1))
      );
    }
  }
  ReferenceBackend<T> backend;
  const auto fitted = fastpls::core::fit_opls_filter_labels(
    predictors, labels.data(), labels.size(), 3, 1,
    fastpls::core::PredictorScaling::centering, backend
  );
  assert(fitted.completed_components == 1);
  for (std::size_t column = 0; column < fitted.weights.columns(); ++column) {
    T norm = T(0);
    for (std::size_t row = 0; row < fitted.weights.rows(); ++row) {
      norm += fitted.weights(row, column) * fitted.weights(row, column);
    }
    assert(std::abs(std::sqrt(norm) - T(1)) < T(2e-4));
  }
}

template<class T>
void check_operator_filter() {
  fastpls::core::Matrix<T> predictors(36, 7);
  fastpls::core::Matrix<T> responses(36, 11);
  for (std::size_t row = 0; row < predictors.rows(); ++row) {
    for (std::size_t column = 0; column < predictors.columns(); ++column) {
      predictors(row, column) = static_cast<T>(
        1.5 + std::sin(0.07 * (row + 1) * (column + 2))
      );
    }
    for (std::size_t column = 0; column < responses.columns(); ++column) {
      responses(row, column) = static_cast<T>(
        predictors(row, column % predictors.columns()) +
        0.2 * std::cos(0.11 * (row + 3) * (column + 1))
      );
    }
  }

  ReferenceBackend<T> backend;
  fastpls::core::RsvdControls controls;
  controls.oversample = 6;
  controls.power = 7;
  controls.seed = 19;
  controls.left_only = true;
  const auto dense = fastpls::core::fit_opls_filter_rsvd<T>(
    predictors, fastpls::core::ConstMatrixView<T>(responses.view()), 1,
    fastpls::core::PredictorScaling::autoscaling, controls, backend
  );

  auto standardized = predictors;
  const auto prepared = fastpls::core::prepare_scaled_dense_operator<T>(
    standardized.view(), fastpls::core::ConstMatrixView<T>(responses.view()),
    fastpls::core::PredictorScaling::autoscaling, backend
  );
  fastpls::core::Matrix<T> centered(
    responses.rows(), responses.columns()
  );
  for (std::size_t column = 0; column < responses.columns(); ++column) {
    for (std::size_t row = 0; row < responses.rows(); ++row) {
      centered(row, column) = responses(row, column) -
        prepared.response_mean[column];
    }
  }
  fastpls::core::Matrix<T> sample_gram(
    responses.rows(), responses.rows()
  );
  backend.gemm(
    centered.view(), centered.view(), false, true, sample_gram.view()
  );
  fastpls::core::OperatorRsvdWorkspace<T> workspace;
  const auto matrix_free = fastpls::core::fit_opls_filter_operator<T>(
    std::move(standardized), responses.view(),
    prepared.response_mean.data(), prepared.response_mean.size(), 1,
    controls, backend, sample_gram.view(), workspace
  );
  assert(matrix_free.completed_components == 1);
  const T tolerance = std::is_same<T, float>::value ? T(2e-3) : T(2e-8);
  T squared_difference = T(0);
  T squared_reference = T(0);
  for (std::size_t index = 0; index < dense.predictors.size(); ++index) {
    const T difference =
      matrix_free.predictors.data()[index] - dense.predictors.data()[index];
    squared_difference += difference * difference;
    squared_reference += dense.predictors.data()[index] *
      dense.predictors.data()[index];
  }
  assert(std::sqrt(squared_difference / squared_reference) < tolerance);
}

int main() {
  check_dense_filter<float>();
  check_dense_filter<double>();
  check_label_filter<float>();
  check_label_filter<double>();
  check_operator_filter<float>();
  check_operator_filter<double>();
  return 0;
}
