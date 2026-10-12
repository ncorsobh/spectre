// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to evolving the Newtonian Euler system
namespace NewtonianEuler {

/// The tags appended to the evolved, flux and primitive variable lists when the
/// magnetic field is evolved, and an empty list otherwise.
template <bool EvolveMagneticField, typename... MagneticTags>
using magnetic_tags_if =
    tmpl::conditional_t<EvolveMagneticField, tmpl::list<MagneticTags...>,
                        tmpl::list<>>;

template <size_t Dim, bool EvolveMagneticField = false>
struct System {
  static_assert(Dim == 3 or not EvolveMagneticField,
                "The magnetic field has three components in any spatial "
                "dimension, so MHD is only instantiated at Dim == 3. Use the "
                "Cartoon basis for lower-dimensional MHD problems.");

  /// Whether the magnetic field and the GLM divergence-cleaning field are
  /// evolved. When false every magnetic code path is removed at compile time
  /// and this is the Newtonian Euler system.
  static constexpr bool evolve_magnetic_field = EvolveMagneticField;

  static std::string name() {
    return EvolveMagneticField ? "NewtonianMhd" : "NewtonianEuler";
  }

  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = Dim;

  using boundary_conditions_base = BoundaryConditions::BoundaryCondition<Dim>;

  using flux_variables = tmpl::append<
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity>,
      magnetic_tags_if<EvolveMagneticField, Tags::MagneticFieldCons<>,
                       Tags::DivergenceCleaningFieldCons>>;
  using variables_tag = ::Tags::Variables<flux_variables>;
  using non_conservative_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<>;
  // Compute item for pressure not currently implemented in SpECTRE,
  // so its simple tag is passed along with the primitive variables.
  using primitive_variables_tag = ::Tags::Variables<tmpl::append<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>>,
      magnetic_tags_if<EvolveMagneticField,
                       hydro::Tags::MagneticField<DataVector, 3>,
                       hydro::Tags::DivergenceCleaningField<DataVector>>>>;

  using compute_volume_time_derivative_terms = TimeDerivativeTerms<Dim>;

  using conservative_from_primitive = ConservativeFromPrimitive<Dim>;
  using primitive_from_conservative = PrimitiveFromConservative<Dim>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed<Dim>;
};

}  // namespace NewtonianEuler
