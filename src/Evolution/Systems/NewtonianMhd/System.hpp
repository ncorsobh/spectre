// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/Characteristics.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Evolution/Systems/NewtonianMhd/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to evolving the Newtonian magnetohydrodynamics system
///
/// The magnetic field is split as \f$B^i = B_0^i + B_1^i\f$ with \f$B_0\f$ a
/// static, curl-free and divergence-free background; only \f$B_1\f$ is evolved.
/// Setting \f$B_0 = 0\f$ recovers standard Newtonian MHD.  Divergence cleaning
/// uses the hyperbolic (GLM) scheme.
namespace NewtonianMhd {

template <size_t Dim, bool UseBackgroundMagneticField = false>
struct System {
  /// Whether the static background field \f$B_0\f$ is split off from the
  /// evolved \f$B_1\f$.
  /// When false every \f$B_0\f$ code path is removed at compile time.
  static constexpr bool use_background_magnetic_field =
      UseBackgroundMagneticField;

  static std::string name() { return "NewtonianMhd"; }

  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = Dim;

  using boundary_conditions_base = BoundaryConditions::BoundaryCondition<Dim>;

  using variables_tag = ::Tags::Variables<tmpl::list<
      Tags::MassDensityCons, Tags::MomentumDensity<Dim>, Tags::EnergyDensity,
      Tags::MagneticFieldCons<Dim>, Tags::DivergenceCleaningFieldCons>>;
  using flux_variables =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity, Tags::MagneticFieldCons<Dim>,
                 Tags::DivergenceCleaningFieldCons>;
  using non_conservative_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<>;
  using primitive_variables_tag = ::Tags::Variables<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::MagneticField<DataVector, Dim>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>>;

  using compute_volume_time_derivative_terms =
      TimeDerivativeTerms<Dim, UseBackgroundMagneticField>;

  using conservative_from_primitive = ConservativeFromPrimitive<Dim>;
  using primitive_from_conservative = PrimitiveFromConservative<Dim>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed<Dim>;
};

}  // namespace NewtonianMhd
