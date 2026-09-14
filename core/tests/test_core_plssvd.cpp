// SPDX-License-Identifier: MIT
#include <fastpls/core.hpp>

#include "reference_backend.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

template<class T>
void check_reference_backend_solves() {
  ReferenceBackend<T> backend;
  fastpls::core::Matrix<T> matrix(2, 2);
  matrix(0, 0) = T(0);
  matrix(1, 0) = T(1);
  matrix(0, 1) = T(1);
  matrix(1, 1) = T(1);
  fastpls::core::Matrix<T> right(2, 1);
  right(0, 0) = T(1);
  right(1, 0) = T(2);
  fastpls::core::Matrix<T> solution;
  assert(!backend.cholesky_solve(matrix.view(), right.view(), solution));
  assert(backend.general_solve(matrix.view(), right.view(), solution));
  const T tolerance = std::is_same<T, float>::value ? T(1e-5) : T(1e-12);
  assert(std::abs(solution(0, 0) - T(1)) < tolerance);
  assert(std::abs(solution(1, 0) - T(1)) < tolerance);
}

template<class T>
void check_plssvd() {
  fastpls::core::Matrix<T> x(8, 3);
  fastpls::core::Matrix<T> y(8, 2);
  for (std::size_t row = 0; row < x.rows(); ++row) {
    const T value = static_cast<T>(row) - T(3.5);
    x(row, 0) = value;
    x(row, 1) = value * value - T(5.25);
    x(row, 2) = (row % 2 == 0 ? T(-1) : T(1));
    y(row, 0) = T(2) * x(row, 0) - T(0.5) * x(row, 1);
    y(row, 1) = x(row, 2) + T(0.25) * x(row, 0);
  }
  fastpls::core::Matrix<T> crosscov(3, 2);
  fastpls::core::reference_gemm(
    x.view(), y.view(), true, false, crosscov.view()
  );
  const int components[] = {1, 2};
  fastpls::core::PlssvdControls controls;
  controls.rsvd.oversample = 4;
  controls.rsvd.power = 2;
  controls.rsvd.seed = 17;
  ReferenceBackend<T> backend;
  const auto model = fastpls::core::fit_plssvd_preprocessed(
    x.view(), crosscov.view(), components, 2, controls, backend
  );
  assert(model.completed_components == 2);
  assert(model.weights.rows() == 3 && model.weights.columns() == 2);
  assert(model.response_loadings.rows() == 2);
  assert(model.scores.rows() == 8 && model.scores.columns() == 2);
  assert(model.prediction_weights.size() == 2);
  assert(model.prediction_weights[0].rows() == 1);
  assert(model.prediction_weights[1].rows() == 2);

  fastpls::core::Matrix<T> shifted_y = y;
  const T response_mean[] = {T(3), T(-2)};
  for (std::size_t column = 0; column < shifted_y.columns(); ++column) {
    for (std::size_t row = 0; row < shifted_y.rows(); ++row) {
      shifted_y(row, column) += response_mean[column];
    }
  }
  fastpls::core::CenteredCrosscovOperator<T, ReferenceBackend<T>> op(
    x.view(), shifted_y.view(), response_mean, 2, backend
  );
  fastpls::core::OperatorRsvdWorkspace<T> workspace;
  const auto implicit_model = fastpls::core::fit_plssvd_operator<T>(
    x.view(), op, components, 2, controls, backend, workspace
  );
  assert(implicit_model.completed_components == model.completed_components);
  for (std::size_t index = 0; index < model.singular_values.size(); ++index) {
    const T scale = std::max(std::abs(model.singular_values[index]), T(1));
    assert(std::abs(implicit_model.singular_values[index] -
      model.singular_values[index]) / scale <
      (std::is_same<T, float>::value ? T(2e-4) : T(2e-10)));
  }
  fastpls::core::Matrix<T> implicit_prediction(8, 2);
  backend.gemm(
    implicit_model.scores.view(),
    implicit_model.prediction_weights[1].view(), false, false,
    implicit_prediction.view()
  );

  fastpls::core::Matrix<T> prediction(8, 2);
  backend.gemm(
    model.scores.view(), model.prediction_weights[1].view(), false, false,
    prediction.view()
  );
  T error = T(0);
  for (std::size_t column = 0; column < y.columns(); ++column) {
    for (std::size_t row = 0; row < y.rows(); ++row) {
      const T difference = prediction(row, column) - y(row, column);
      error += difference * difference;
      assert(std::abs(prediction(row, column) -
        implicit_prediction(row, column)) <
        (std::is_same<T, float>::value ? T(2e-4) : T(2e-10)));
    }
  }
  assert(std::isfinite(error));
  const T tolerance = std::is_same<T, float>::value ? T(2e-3) : T(2e-7);
  assert(std::abs(error - T(10.5762741916562)) < tolerance);
  assert(std::abs(model.singular_values[0] - T(119.561004785789)) <
         (std::is_same<T, float>::value ? T(2e-3) : T(2e-7)));

  fastpls::core::Matrix<T> tall_x(32, 3);
  fastpls::core::Matrix<T> tall_y(32, 2);
  for (std::size_t row = 0; row < tall_x.rows(); ++row) {
    const T value = static_cast<T>(row) - T(15.5);
    tall_x(row, 0) = value;
    tall_x(row, 1) = std::sin(value);
    tall_x(row, 2) = std::cos(value * T(0.5));
    tall_y(row, 0) = tall_x(row, 0) + T(0.5) * tall_x(row, 1);
    tall_y(row, 1) = tall_x(row, 2) - T(0.25) * tall_x(row, 0);
  }
  fastpls::core::Matrix<T> tall_crosscov(3, 2);
  backend.gemm(
    tall_x.view(), tall_y.view(), true, false, tall_crosscov.view()
  );
  const auto score_free_model = fastpls::core::fit_plssvd_preprocessed(
    tall_x.view(), tall_crosscov.view(), components, 2, controls, backend,
    false
  );
  const auto score_model = fastpls::core::fit_plssvd_preprocessed(
    tall_x.view(), tall_crosscov.view(), components, 2, controls, backend,
    true
  );
  assert(score_free_model.scores.rows() == 0);
  assert(score_free_model.score_gram.rows() == 2);
  assert(score_free_model.prediction_weights.size() == 2);
  const T route_tolerance = std::is_same<T, float>::value ?
    T(5e-4) : T(5e-11);
  for (std::size_t index = 0;
       index < score_free_model.score_gram.size(); ++index) {
    assert(std::abs(score_free_model.score_gram.data()[index] -
                    score_model.score_gram.data()[index]) < route_tolerance);
  }
  for (std::size_t index = 0;
       index < score_free_model.prediction_weights[1].size(); ++index) {
    assert(std::abs(score_free_model.prediction_weights[1].data()[index] -
                    score_model.prediction_weights[1].data()[index]) <
           route_tolerance);
  }
}

int main() {
  check_reference_backend_solves<float>();
  check_reference_backend_solves<double>();
  check_plssvd<float>();
  check_plssvd<double>();
  return 0;
}
