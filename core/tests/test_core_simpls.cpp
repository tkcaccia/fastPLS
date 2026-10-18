// SPDX-License-Identifier: MIT
#include "reference_backend.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>

namespace {

template<class T>
void check() {
  using fastpls::core::Matrix;
  using fastpls::core::SimplsControls;
  using fastpls::core::SimplsWorkspace;

  ReferenceBackend<T> backend;
  Matrix<T> predictors(8, 3);
  const T values[24] = {
    T(-3), T(-2), T(-1), T(0), T(0), T(1), T(2), T(3),
    T(1), T(-1), T(1), T(-1), T(1), T(-1), T(1), T(-1),
    T(-1), T(2), T(-2), T(1), T(1), T(-2), T(2), T(-1)
  };
  for (std::size_t index = 0; index < 24; ++index) {
    predictors.data()[index] = values[index];
  }

  Matrix<T> coefficients(3, 3);
  coefficients(0, 0) = T(1.5);
  coefficients(1, 0) = T(-0.5);
  coefficients(0, 1) = T(-0.75);
  coefficients(1, 1) = T(2);
  coefficients(0, 2) = T(0.25);
  coefficients(1, 2) = T(1.25);
  Matrix<T> response(8, 3);
  backend.gemm(
    predictors.view(), coefficients.view(), false, false, response.view()
  );
  Matrix<T> crosscov(3, 3);
  backend.gemm(
    predictors.view(), response.view(), true, false, crosscov.view()
  );

  SimplsControls controls;
  controls.components = 2;
  controls.maximum_block = 1;
  controls.cache_predictor_crossprod = false;
  controls.reorthogonalize = true;
  controls.store_scores = true;
  controls.rsvd.oversample = 0;
  controls.rsvd.power = 5;
  controls.rsvd.seed = 19;
  SimplsWorkspace<T> workspace;
  auto model = fastpls::core::fit_simpls_preprocessed<T>(
    predictors.view(), crosscov.view(), controls, backend, workspace
  );
  assert(model.completed_components == 2);

  std::vector<T> response_mean(response.columns(), T(0));
  fastpls::core::CenteredCrosscovOperator<T, ReferenceBackend<T>> initial(
    predictors.view(), response.view(), response_mean.data(),
    response_mean.size(), backend
  );
  fastpls::core::ProjectedOperator<
    T,
    fastpls::core::CenteredCrosscovOperator<T, ReferenceBackend<T>>,
    ReferenceBackend<T>
  > projected(initial, controls.components, backend);
  SimplsWorkspace<T> operator_workspace;
  fastpls::core::OperatorRsvdWorkspace<T> rsvd_workspace;
  auto operator_model = fastpls::core::fit_simpls_operator<T>(
    predictors.view(), initial, projected, controls, backend,
    operator_workspace, rsvd_workspace
  );
  assert(operator_model.completed_components == 2);

  Matrix<T> prediction = fastpls::core::predict_simpls_preprocessed<T>(
    predictors.view(), model, 2, backend
  );
  T squared_error = T(0);
  T response_sum_squares = T(0);
  for (std::size_t index = 0; index < response.size(); ++index) {
    const T difference = prediction.data()[index] - response.data()[index];
    squared_error += difference * difference;
    response_sum_squares += response.data()[index] * response.data()[index];
  }
  const T relative_error = std::sqrt(squared_error / response_sum_squares);
  const T tolerance = sizeof(T) == sizeof(float) ? T(2e-4) : T(1e-8);
  assert(relative_error < tolerance);

  Matrix<T> operator_prediction =
    fastpls::core::predict_simpls_preprocessed<T>(
      predictors.view(), operator_model, 2, backend
    );
  T operator_difference = T(0);
  for (std::size_t index = 0; index < prediction.size(); ++index) {
    const T difference =
      prediction.data()[index] - operator_prediction.data()[index];
    operator_difference += difference * difference;
  }
  assert(std::sqrt(operator_difference / response_sum_squares) < tolerance);

  for (std::size_t left = 0; left < 2; ++left) {
    for (std::size_t right = 0; right < 2; ++right) {
      T score_product = T(0);
      T basis_product = T(0);
      for (std::size_t row = 0; row < predictors.rows(); ++row) {
        score_product += model.scores(row, left) * model.scores(row, right);
      }
      for (std::size_t row = 0; row < predictors.columns(); ++row) {
        basis_product += model.deflation_basis(row, left) *
          model.deflation_basis(row, right);
      }
      const T expected = left == right ? T(1) : T(0);
      assert(std::abs(score_product - expected) < tolerance);
      assert(std::abs(basis_product - expected) < tolerance);
    }
  }
}

template<class T>
void check_score_moments(bool cached) {
  using fastpls::core::Matrix;
  ReferenceBackend<T> backend;
  std::mt19937 rng(19);
  std::normal_distribution<double> normal;
  Matrix<T> predictors(40, 64);
  Matrix<T> response(40, 2);
  for (std::size_t column = 0; column < predictors.columns(); ++column) {
    T mean = T(0);
    for (std::size_t row = 0; row < predictors.rows(); ++row) {
      predictors(row, column) = static_cast<T>(normal(rng));
      mean += predictors(row, column);
    }
    mean /= static_cast<T>(predictors.rows());
    for (std::size_t row = 0; row < predictors.rows(); ++row) {
      predictors(row, column) -= mean;
    }
  }
  for (std::size_t row = 0; row < response.rows(); ++row) {
    response(row, 0) = row % 2 == 0 ? T(-0.5) : T(0.5);
    response(row, 1) = -response(row, 0);
  }
  Matrix<T> crosscov(64, 2);
  backend.gemm(
    predictors.view(), response.view(), true, false, crosscov.view()
  );
  fastpls::core::SimplsControls controls;
  controls.components = 24;
  controls.maximum_block = 1;
  controls.reorthogonalize = false;
  controls.cache_predictor_crossprod = cached;
  controls.store_scores = true;
  controls.store_score_moments = true;
  fastpls::core::SimplsWorkspace<T> workspace;
  const auto model = fastpls::core::fit_simpls_preprocessed<T>(
    predictors.view(), crosscov.view(), controls, backend, workspace
  );
  assert(model.completed_components > 1);
  const T tolerance = sizeof(T) == sizeof(float) ? T(2e-5) : T(2e-12);
  for (std::size_t column = 0;
       column < model.completed_components; ++column) {
    for (std::size_t row = 0; row < model.completed_components; ++row) {
      T expected = T(0);
      for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
        expected += model.scores(sample, row) * model.scores(sample, column);
      }
      assert(std::isfinite(model.score_gram(row, column)));
      assert(std::abs(model.score_gram(row, column) - expected) < tolerance);
    }
  }
}

}  // namespace

int main() {
  check<float>();
  check<double>();
  check_score_moments<float>(false);
  check_score_moments<double>(false);
  check_score_moments<float>(true);
  check_score_moments<double>(true);
  return 0;
}
