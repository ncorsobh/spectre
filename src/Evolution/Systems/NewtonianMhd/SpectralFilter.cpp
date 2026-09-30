// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using newtonian_mhd_tags = tmpl::list<
    NewtonianMhd::Tags::MassDensityCons, NewtonianMhd::Tags::MomentumDensity<>,
    NewtonianMhd::Tags::EnergyDensity, NewtonianMhd::Tags::MagneticFieldCons<>,
    NewtonianMhd::Tags::DivergenceCleaningFieldCons>;
}  // namespace

#define INSTANTIATE(_, data)                                      \
  template class Filters::Hypercube<3, newtonian_mhd_tags>;       \
  template class Filters::None<3, newtonian_mhd_tags>;            \
  template struct evolution::dg::Initialization::SpectralFilters< \
      3, newtonian_mhd_tags>;

INSTANTIATE(~, ~)

template class Filters::HollowCylinder<newtonian_mhd_tags>;
template class Filters::FilledCylinder<newtonian_mhd_tags>;

#undef INSTANTIATE
