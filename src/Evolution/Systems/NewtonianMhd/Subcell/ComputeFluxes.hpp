// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::subcell {
/*!
 * \brief Helper that calls `ComputeFluxes` on variables held on a subcell
 * face.
 *
 * Unlike the generic tag-list version used by other systems, the fluxes need
 * the divergence-cleaning speed, which is a constant of the element rather than
 * a field, so it is passed separately. The background magnetic field is taken
 * from `vars` where the splitting is enabled; where it is disabled it is not
 * part of `vars` at all.
 */
template <size_t Dim, bool UseBackgroundMagneticField, typename TagsList>
void compute_fluxes(const gsl::not_null<Variables<TagsList>*> vars,
                    const double divergence_cleaning_speed) {
  const auto call = [&vars, &divergence_cleaning_speed](
                        const auto&... background_magnetic_field) {
    NewtonianMhd::ComputeFluxes<Dim, UseBackgroundMagneticField>::apply(
        make_not_null(
            &get<::Tags::Flux<Tags::MassDensityCons, tmpl::size_t<Dim>,
                              Frame::Inertial>>(*vars)),
        make_not_null(
            &get<::Tags::Flux<Tags::MomentumDensity<Dim>, tmpl::size_t<Dim>,
                              Frame::Inertial>>(*vars)),
        make_not_null(&get<::Tags::Flux<Tags::EnergyDensity, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(*vars)),
        make_not_null(
            &get<::Tags::Flux<Tags::MagneticFieldCons<Dim>, tmpl::size_t<Dim>,
                              Frame::Inertial>>(*vars)),
        make_not_null(
            &get<::Tags::Flux<Tags::DivergenceCleaningFieldCons,
                              tmpl::size_t<Dim>, Frame::Inertial>>(*vars)),
        get<Tags::MomentumDensity<Dim>>(*vars), get<Tags::EnergyDensity>(*vars),
        get<Tags::MagneticFieldCons<Dim>>(*vars),
        get<Tags::DivergenceCleaningFieldCons>(*vars),
        get<hydro::Tags::SpatialVelocity<DataVector, Dim>>(*vars),
        get<hydro::Tags::Pressure<DataVector>>(*vars),
        divergence_cleaning_speed, background_magnetic_field...);
  };
  if constexpr (UseBackgroundMagneticField) {
    call(get<Tags::BackgroundMagneticField<Dim>>(*vars));
  } else {
    call();
  }
}
}  // namespace NewtonianMhd::subcell
