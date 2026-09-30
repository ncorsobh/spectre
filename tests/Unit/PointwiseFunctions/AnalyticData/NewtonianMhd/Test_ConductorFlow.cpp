// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianMhd/ConductorFlow.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
using ConductorFlow = NewtonianMhd::AnalyticData::ConductorFlow;
using BackgroundField = NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>;
using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>;
using MagneticField =
    hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>;

constexpr double conductor_radius = 1.5;
constexpr double asymptotic_velocity = 0.1;
constexpr double field_strength = 1.3;

ConductorFlow make_solution(const double dipole_strength,
                            const bool magnetized) {
  return ConductorFlow{
      5.0 / 3.0,       2.0, 1.0, asymptotic_velocity, field_strength,
      dipole_strength, 0.3, 0.7, conductor_radius,    magnetized};
}

tnsr::I<DataVector, 3> point(const std::array<double, 3>& coords) {
  tnsr::I<DataVector, 3> x{1_st};
  for (size_t i = 0; i < 3; ++i) {
    x.get(i) = DataVector{1, gsl::at(coords, i)};
  }
  return x;
}

// The single-tag `variables` overloads are private, so always ask for the full
// set and pick out what is needed.
using all_tags = tmpl::list<Velocity, MagneticField, BackgroundField>;

tuples::tagged_tuple_from_typelist<all_tags> all_variables(
    const ConductorFlow& solution, const tnsr::I<DataVector, 3>& x) {
  return solution.variables(x, all_tags{});
}

std::array<double, 3> background_field_at(const ConductorFlow& solution,
                                          const std::array<double, 3>& coords) {
  const auto field =
      get<BackgroundField>(all_variables(solution, point(coords)));
  return {{get<0>(field)[0], get<1>(field)[0], get<2>(field)[0]}};
}

// The background field must be divergence free and curl free everywhere outside
// the conductor, which is what makes the B0/B1 splitting of the fluxes exact.
void test_background_field_is_harmonic() {
  const auto solution = make_solution(0.8, true);
  const double step = 1.0e-5;
  const std::array<std::array<double, 3>, 3> sample_points{
      {{{2.0, 0.5, -1.0}}, {{-3.0, 2.5, 4.0}}, {{0.0, 0.0, 5.0}}}};

  for (const auto& center : sample_points) {
    std::array<std::array<double, 3>, 3> derivative{};
    for (size_t j = 0; j < 3; ++j) {
      auto forward = center;
      auto backward = center;
      gsl::at(forward, j) += step;
      gsl::at(backward, j) -= step;
      const auto field_forward = background_field_at(solution, forward);
      const auto field_backward = background_field_at(solution, backward);
      for (size_t i = 0; i < 3; ++i) {
        // derivative[i][j] = d B^i / d x^j
        gsl::at(gsl::at(derivative, i), j) =
            (gsl::at(field_forward, i) - gsl::at(field_backward, i)) /
            (2.0 * step);
      }
    }

    const double divergence =
        derivative[0][0] + derivative[1][1] + derivative[2][2];
    Approx custom_approx = Approx::custom().epsilon(1.0e-8).scale(1.0);
    CHECK(divergence == custom_approx(0.0));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < i; ++j) {
        CHECK(gsl::at(gsl::at(derivative, i), j) ==
              custom_approx(gsl::at(gsl::at(derivative, j), i)));
      }
    }
  }
}

// Without the conductor's own dipole the reaction dipole is chosen exactly so
// that the radial field vanishes on the surface.
void test_radial_field_vanishes_on_surface() {
  const auto solution = make_solution(0.0, true);
  const std::array<std::array<double, 3>, 3> directions{
      {{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.6, 0.0, 0.8}}}};
  for (const auto& direction : directions) {
    std::array<double, 3> surface_point{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(surface_point, i) = conductor_radius * gsl::at(direction, i);
    }
    const auto field = background_field_at(solution, surface_point);
    double radial_component = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      radial_component += gsl::at(field, i) * gsl::at(direction, i);
    }
    CHECK(radial_component == approx(0.0));
  }
}

// The whole static field is reported both as the total field and as the
// background available for splitting, so an executable that splits it off ends
// up evolving B1 = 0.
void test_total_field_is_the_background() {
  const auto solution = make_solution(0.8, true);
  const auto values = all_variables(solution, point({{2.0, -1.0, 3.0}}));
  const auto& background = get<BackgroundField>(values);
  const auto& total = get<MagneticField>(values);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(total.get(i)[0] == approx(background.get(i)[0]));
  }
}

void test_velocity_profiles() {
  const auto stokes = make_solution(0.0, true);
  const auto potential = make_solution(0.0, false);

  // Far from the conductor both profiles approach the uniform wind along +z.
  const auto far_point = point({{0.0, 0.0, 1.0e8}});
  for (const auto& solution : {stokes, potential}) {
    const auto velocity = get<Velocity>(all_variables(solution, far_point));
    // The Stokes profile approaches the uniform wind only as R_0 / r.
    Approx custom_approx = Approx::custom().epsilon(1.0e-6).scale(1.0);
    CHECK(get<0>(velocity)[0] == custom_approx(0.0));
    CHECK(get<1>(velocity)[0] == custom_approx(0.0));
    CHECK(get<2>(velocity)[0] == custom_approx(asymptotic_velocity));
  }

  // On the surface neither profile lets fluid through, and the Stokes profile
  // additionally has no tangential slip.
  const std::array<double, 3> direction{{0.6, 0.0, 0.8}};
  std::array<double, 3> surface_coords{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(surface_coords, i) =
        conductor_radius * (1.0 + 1.0e-12) * gsl::at(direction, i);
  }
  const auto surface_point = point(surface_coords);
  for (const auto& solution : {stokes, potential}) {
    const auto velocity = get<Velocity>(all_variables(solution, surface_point));
    double radial_velocity = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      radial_velocity += velocity.get(i)[0] * gsl::at(direction, i);
    }
    Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
    CHECK(radial_velocity == custom_approx(0.0));
  }
  const auto stokes_velocity =
      get<Velocity>(all_variables(stokes, surface_point));
  for (size_t i = 0; i < 3; ++i) {
    Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
    CHECK(stokes_velocity.get(i)[0] == custom_approx(0.0));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.AnalyticData.ConductorFlow",
                  "[Unit][PointwiseFunctions]") {
  test_background_field_is_harmonic();
  test_radial_field_vanishes_on_surface();
  test_total_field_is_the_background();
  test_velocity_profiles();

  const auto created =
      TestHelpers::test_factory_creation<evolution::initial_data::InitialData,
                                         ConductorFlow>(
          "ConductorFlow:\n"
          "  AdiabaticIndex: 1.6666666666666667\n"
          "  BackgroundDensity: 2.0\n"
          "  BackgroundPressure: 1.0\n"
          "  AsymptoticVelocity: 0.1\n"
          "  MagneticFieldStrength: 1.3\n"
          "  ConductorInternalField: 0.8\n"
          "  MomentTiltAngle: 0.3\n"
          "  MomentAzimuthal: 0.7\n"
          "  ConductorRadius: 1.5\n"
          "  Magnetized: True\n");
  CHECK(dynamic_cast<const ConductorFlow&>(*created) ==
        make_solution(0.8, true));
  test_serialization(make_solution(0.8, true));
}
