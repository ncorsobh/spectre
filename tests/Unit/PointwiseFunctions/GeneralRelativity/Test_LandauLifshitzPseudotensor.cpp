// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/ElementMap.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfDetSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/LandauLifshitzPseudotensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace gr {

template <typename Frame = Frame::Inertial>
struct Flux : db::SimpleTag {
  using type = tnsr::IA<DataVector, 3, Frame>;
  static auto flux(
      const tnsr::AA<DataVector, 3, Frame>& landau_lifshitz_pseudotensor) {
    tnsr::IA<DataVector, 3, Frame> result;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j <= 3; ++j) {
        result.get(i, j) = landau_lifshitz_pseudotensor.get(i, j);
      }
    }
    return result;
  }
};

template <typename Solution>
tnsr::AA<DataVector, 3> calc_landau_lifshitz_pseudotensor(
    const Solution& solution, const tnsr::I<DataVector, 3, Frame::Inertial> x,
    const double t) {
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(vars);
  const auto& deriv_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(vars);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(vars);
  const auto& deriv_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(vars);
  const auto& dt_spatial_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(vars);
  const auto& deriv_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  const auto& test_spacetime_metric =
      spacetime_metric(lapse, shift, spatial_metric);

  const auto& test_inverse_spacetime_metric =
      inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto& da_spacetime_metric = derivatives_of_spacetime_metric(
      lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
      spatial_metric, dt_spatial_metric, deriv_spatial_metric);

  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars);

  const auto& phi = gh::phi(lapse, deriv_lapse, shift, deriv_shift,
                            spatial_metric, deriv_spatial_metric);

  const auto& da_det_spatial_metric = gh::spacetime_deriv_of_det_spatial_metric(
      sqrt_det_spatial_metric, inverse_spatial_metric, dt_spatial_metric, phi);

  return landau_lifshitz_pseudotensor(
      test_spacetime_metric, test_inverse_spacetime_metric, da_spacetime_metric,
      lapse, dt_lapse, deriv_lapse, sqrt_det_spatial_metric,
      da_det_spatial_metric);
}

template <typename Solution>
void test_div_landau_lifshitz_pseudotensor(
    const Solution& solution, const size_t grid_size_each_dimension,
    bool time_dependent = false) {
  const Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto coordinate_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto xi = logical_coordinates(mesh);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);

  double t;
  double dt;
  if (time_dependent) {
    t = 0.23;
    dt = 0.05;
  } else {
    // Arbitrary time for time-independent solution.
    t = std::numeric_limits<double>::signaling_NaN();
  }

  const auto& test_landau_lifshitz_pseudotensor =
      calc_landau_lifshitz_pseudotensor(solution, x, std::as_const(t));

  Variables<tmpl::list<Flux<>>> fluxes(num_grid_points);
  get<Flux<>>(fluxes) = Flux<>::flux(test_landau_lifshitz_pseudotensor);

  const auto& div_landau_lifshitz_pseudotensor =
      divergence(fluxes, mesh, inv_jacobian);

  Approx my_approx = Approx::custom().epsilon(1e-7).scale(1.0);
  auto expected_div_ll_pseudotensor = make_with_value<DataVector>(x, 0.);

  if (time_dependent) {
    const auto& test_landau_lifshitz_pseudotensor_minusdt =
        calc_landau_lifshitz_pseudotensor(solution, x, t - 0.5 * dt);
    const auto& test_landau_lifshitz_pseudotensor_plusdt =
        calc_landau_lifshitz_pseudotensor(solution, x, t + 0.5 * dt);
    tnsr::A<DataVector, 3> time_div_landau_lifshitz_pseudotensor;
    for (size_t i = 0; i <= 3; ++i) {
      time_div_landau_lifshitz_pseudotensor.get(i) =
          get<::Tags::div<Flux<>>>(div_landau_lifshitz_pseudotensor).get(i) +
          (test_landau_lifshitz_pseudotensor_plusdt.get(i, 0) -
           test_landau_lifshitz_pseudotensor_minusdt.get(i, 0)) /
              dt;
      CHECK_ITERABLE_CUSTOM_APPROX(time_div_landau_lifshitz_pseudotensor.get(i),
                                   expected_div_ll_pseudotensor, my_approx);
    }
  } else {
    for (size_t i = 0; i <= 3; ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(
          get<::Tags::div<Flux<>>>(div_landau_lifshitz_pseudotensor).get(i),
          expected_div_ll_pseudotensor, my_approx);
    }
  }
}

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.LandauLifshitzPseudotensor",
    "[PointwiseFunctions][Unit]") {
  const double mass = 1.4;
  const std::array<double, 3> spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{1.1, 0.2, 0.4}};
  const Solutions::KerrSchild ks_solution{mass, spin, center};

  const double amplitude = 0.8;
  const double wavelength = 1.6;
  const Solutions::GaugeWave<3> gw_solution{amplitude, wavelength};

  test_div_landau_lifshitz_pseudotensor(ks_solution, 12);
  test_div_landau_lifshitz_pseudotensor(gw_solution, 12, true);
}

}  // namespace gr
