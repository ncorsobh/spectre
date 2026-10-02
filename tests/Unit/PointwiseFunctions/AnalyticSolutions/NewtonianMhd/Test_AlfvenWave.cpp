// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
using AlfvenWave = NewtonianMhd::Solutions::AlfvenWave;
using Density = hydro::Tags::RestMassDensity<DataVector>;
using Pressure = hydro::Tags::Pressure<DataVector>;
using SpecificInternalEnergy = hydro::Tags::SpecificInternalEnergy<DataVector>;
using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>;
using MagneticField =
    hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>;
using DivergenceCleaningField =
    hydro::Tags::DivergenceCleaningField<DataVector>;
using all_tags = tmpl::list<Density, Pressure, SpecificInternalEnergy, Velocity,
                            MagneticField, DivergenceCleaningField>;

constexpr double adiabatic_index = 5.0 / 3.0;
constexpr double background_density = 1.5;
constexpr double background_pressure = 0.8;
constexpr double parallel_magnetic_field = 1.1;
constexpr double amplitude = 0.3;
constexpr double divergence_cleaning_speed = 1.7;
const std::array<double, 3> wavevector{{1.0, 2.0, -1.0}};

AlfvenWave make_solution() {
  return AlfvenWave{wavevector,          background_density,
                    background_pressure, parallel_magnetic_field,
                    amplitude,           adiabatic_index};
}

tnsr::I<DataVector, 3> point(const std::array<double, 3>& coords) {
  tnsr::I<DataVector, 3> x{1_st};
  for (size_t i = 0; i < 3; ++i) {
    x.get(i) = DataVector{1, gsl::at(coords, i)};
  }
  return x;
}

// The nine conservative variables at a point, flattened so that finite
// differences are easy to take.
struct State {
  std::array<double, 9> conservative{};
  // flux[variable][direction]
  std::array<std::array<double, 3>, 9> flux{};
};

State state_at(const AlfvenWave& solution, const std::array<double, 3>& coords,
               const double time) {
  const auto x = point(coords);
  const auto primitives = solution.variables(x, time, all_tags{});

  Scalar<DataVector> mass_density_cons{1_st};
  tnsr::I<DataVector, 3> momentum_density{1_st};
  Scalar<DataVector> energy_density{1_st};
  tnsr::I<DataVector, 3> magnetic_field_cons{1_st};
  Scalar<DataVector> divergence_cleaning_field_cons{1_st};
  NewtonianMhd::ConservativeFromPrimitive::apply(
      make_not_null(&mass_density_cons), make_not_null(&momentum_density),
      make_not_null(&energy_density), make_not_null(&magnetic_field_cons),
      make_not_null(&divergence_cleaning_field_cons), get<Density>(primitives),
      get<Velocity>(primitives), get<SpecificInternalEnergy>(primitives),
      get<MagneticField>(primitives), get<DivergenceCleaningField>(primitives));

  tnsr::I<DataVector, 3> mass_density_flux{1_st};
  tnsr::IJ<DataVector, 3> momentum_density_flux{1_st};
  tnsr::I<DataVector, 3> energy_density_flux{1_st};
  tnsr::IJ<DataVector, 3> magnetic_field_flux{1_st};
  tnsr::I<DataVector, 3> divergence_cleaning_field_flux{1_st};
  NewtonianMhd::ComputeFluxes<false>::apply(
      make_not_null(&mass_density_flux), make_not_null(&momentum_density_flux),
      make_not_null(&energy_density_flux), make_not_null(&magnetic_field_flux),
      make_not_null(&divergence_cleaning_field_flux), momentum_density,
      energy_density, magnetic_field_cons, divergence_cleaning_field_cons,
      get<Velocity>(primitives), get<Pressure>(primitives),
      divergence_cleaning_speed);

  State result{};
  result.conservative[0] = get(mass_density_cons)[0];
  result.conservative[4] = get(energy_density)[0];
  result.conservative[8] = get(divergence_cleaning_field_cons)[0];
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result.conservative, 1 + i) = momentum_density.get(i)[0];
    gsl::at(result.conservative, 5 + i) = magnetic_field_cons.get(i)[0];
  }
  for (size_t j = 0; j < 3; ++j) {
    gsl::at(gsl::at(result.flux, 0), j) = mass_density_flux.get(j)[0];
    gsl::at(gsl::at(result.flux, 4), j) = energy_density_flux.get(j)[0];
    gsl::at(gsl::at(result.flux, 8), j) =
        divergence_cleaning_field_flux.get(j)[0];
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(gsl::at(result.flux, 1 + i), j) =
          momentum_density_flux.get(j, i)[0];
      gsl::at(gsl::at(result.flux, 5 + i), j) =
          magnetic_field_flux.get(j, i)[0];
    }
  }
  return result;
}

// The whole point of an analytic solution: it must satisfy the evolution
// equations. With no background field and no source terms these read
// dt U + d_j F^j(U) = 0 for all nine conservative variables, so this checks the
// solution and the fluxes against each other.
void test_satisfies_evolution_equations() {
  const auto solution = make_solution();
  const double step = 1.0e-5;
  const double time = 0.37;
  const std::array<std::array<double, 3>, 3> sample_points{
      {{{0.1, -0.2, 0.4}}, {{0.7, 0.3, -0.6}}, {{-0.5, 0.9, 0.2}}}};

  for (const auto& center : sample_points) {
    const auto forward_in_time = state_at(solution, center, time + step);
    const auto backward_in_time = state_at(solution, center, time - step);

    std::array<double, 9> divergence_of_flux{};
    for (size_t j = 0; j < 3; ++j) {
      auto forward = center;
      auto backward = center;
      gsl::at(forward, j) += step;
      gsl::at(backward, j) -= step;
      const auto state_forward = state_at(solution, forward, time);
      const auto state_backward = state_at(solution, backward, time);
      for (size_t variable = 0; variable < 9; ++variable) {
        gsl::at(divergence_of_flux, variable) +=
            (gsl::at(gsl::at(state_forward.flux, variable), j) -
             gsl::at(gsl::at(state_backward.flux, variable), j)) /
            (2.0 * step);
      }
    }

    // The individual terms are O(10) here, so the residual is a part in 1e7 or
    // better; the floor is set by the O(h^2) truncation error of the central
    // differences, which is a few times 1e-8 for this wavevector.
    const Approx custom_approx = Approx::custom().epsilon(1.0e-6).scale(1.0);
    for (size_t variable = 0; variable < 9; ++variable) {
      const double time_derivative =
          (gsl::at(forward_in_time.conservative, variable) -
           gsl::at(backward_in_time.conservative, variable)) /
          (2.0 * step);
      CHECK(time_derivative + gsl::at(divergence_of_flux, variable) ==
            custom_approx(0.0));
    }
  }
}

// Circular polarization means the magnitudes of B and v do not vary, which is
// what keeps this an exact solution at finite amplitude.
void test_magnitudes_are_uniform() {
  const auto solution = make_solution();
  const double expected_magnetic_field_squared =
      square(parallel_magnetic_field) * (1.0 + square(amplitude));
  const double expected_velocity_squared =
      square(amplitude * parallel_magnetic_field) / background_density;

  const std::array<std::array<double, 3>, 4> sample_points{
      {{{0.0, 0.0, 0.0}},
       {{0.3, -0.7, 0.2}},
       {{1.1, 0.4, -0.9}},
       {{-0.6, 0.15, 0.8}}}};
  for (const auto& coords : sample_points) {
    const auto values = solution.variables(point(coords), 0.21, all_tags{});
    double magnetic_field_squared = 0.0;
    double velocity_squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      magnetic_field_squared += square(get<MagneticField>(values).get(i)[0]);
      velocity_squared += square(get<Velocity>(values).get(i)[0]);
    }
    CHECK(magnetic_field_squared == approx(expected_magnetic_field_squared));
    CHECK(velocity_squared == approx(expected_velocity_squared));
  }
}

// The wave profile translates rigidly along the propagation direction at the
// Alfven speed.
void test_translates_at_the_alfven_speed() {
  const auto solution = make_solution();
  const double time = 0.44;
  double wavevector_magnitude = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    wavevector_magnitude += square(gsl::at(wavevector, i));
  }
  wavevector_magnitude = sqrt(wavevector_magnitude);

  const std::array<double, 3> coords{{0.2, -0.3, 0.5}};
  std::array<double, 3> shifted{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(shifted, i) =
        gsl::at(coords, i) - (solution.alfven_speed() * time *
                              gsl::at(wavevector, i) / wavevector_magnitude);
  }

  const auto evolved = solution.variables(point(coords), time, all_tags{});
  const auto initial = solution.variables(point(shifted), 0.0, all_tags{});
  for (size_t i = 0; i < 3; ++i) {
    CHECK(get<MagneticField>(evolved).get(i)[0] ==
          approx(get<MagneticField>(initial).get(i)[0]));
    CHECK(get<Velocity>(evolved).get(i)[0] ==
          approx(get<Velocity>(initial).get(i)[0]));
  }
}

// The background field must be the static, curl-free and divergence-free part
// of the solution: uniform, aligned with the propagation direction, and equal
// to the full field minus the transverse perturbation.
void test_background_field_splits_off_the_parallel_part() {
  const auto solution = make_solution();
  using BackgroundField = NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>;

  const std::array<std::array<double, 3>, 4> sample_points{
      {{{0.0, 0.0, 0.0}},
       {{0.3, -0.7, 0.2}},
       {{1.1, 0.4, -0.9}},
       {{-0.6, 0.15, 0.8}}}};
  const auto reference = solution.variables(point(sample_points[0]), 0.0,
                                            tmpl::list<BackgroundField>{});
  for (const double time : {0.0, 0.37, 1.4}) {
    for (const auto& coords : sample_points) {
      const auto background = solution.variables(point(coords), time,
                                                 tmpl::list<BackgroundField>{});
      double background_squared = 0.0;
      double parallel_projection = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        // Static in space and time.
        CHECK(get<BackgroundField>(background).get(i)[0] ==
              approx(get<BackgroundField>(reference).get(i)[0]));
        background_squared +=
            square(get<BackgroundField>(background).get(i)[0]);
        parallel_projection +=
            get<BackgroundField>(background).get(i)[0] * gsl::at(wavevector, i);
      }
      // Aligned with the propagation direction and of magnitude B_parallel.
      CHECK(background_squared == approx(square(parallel_magnetic_field)));
      double wavevector_norm = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        wavevector_norm += square(gsl::at(wavevector, i));
      }
      CHECK(parallel_projection ==
            approx(parallel_magnetic_field * sqrt(wavevector_norm)));

      // What is left after removing it is purely transverse, so the evolved
      // perturbation carries no component along the propagation direction.
      const auto total =
          solution.variables(point(coords), time, tmpl::list<MagneticField>{});
      double perturbation_projection = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        perturbation_projection +=
            (get<MagneticField>(total).get(i)[0] -
             get<BackgroundField>(background).get(i)[0]) *
            gsl::at(wavevector, i);
      }
      CHECK(perturbation_projection == approx(0.0));

      // The evolved variable the splitting reports must be exactly what is
      // left after removing the background.
      const auto evolved = solution.variables(
          point(coords), time,
          tmpl::list<NewtonianMhd::Tags::MagneticFieldCons<>>{});
      for (size_t i = 0; i < 3; ++i) {
        CHECK(get<NewtonianMhd::Tags::MagneticFieldCons<>>(evolved).get(i)[0] ==
              approx(get<MagneticField>(total).get(i)[0] -
                     get<BackgroundField>(background).get(i)[0]));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.Solutions.AlfvenWave",
                  "[Unit][PointwiseFunctions]") {
  test_satisfies_evolution_equations();
  test_magnitudes_are_uniform();
  test_translates_at_the_alfven_speed();
  test_background_field_splits_off_the_parallel_part();

  const auto created =
      TestHelpers::test_factory_creation<evolution::initial_data::InitialData,
                                         AlfvenWave>(
          "AlfvenWave:\n"
          "  WaveVector: [1.0, 2.0, -1.0]\n"
          "  BackgroundDensity: 1.5\n"
          "  BackgroundPressure: 0.8\n"
          "  ParallelMagneticField: 1.1\n"
          "  Amplitude: 0.3\n"
          "  AdiabaticIndex: 1.6666666666666667\n");
  CHECK(dynamic_cast<const AlfvenWave&>(*created) == make_solution());
  test_serialization(make_solution());
}
