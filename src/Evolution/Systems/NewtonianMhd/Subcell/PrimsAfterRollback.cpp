// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Subcell/PrimsAfterRollback.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::subcell {
void PrimsAfterRollback::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                   MagneticField, DivergenceCleaningField>>*>
        prim_vars,
    const bool did_rollback, const Mesh<3>& subcell_mesh,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3>& magnetic_field_cons,
    const Scalar<DataVector>& divergence_cleaning_field_cons,
    const EquationsOfState::EquationOfState<false, 2>& equation_of_state) {
  if (did_rollback) {
    const size_t num_grid_points = subcell_mesh.number_of_grid_points();
    if (prim_vars->number_of_grid_points() != num_grid_points) {
      prim_vars->initialize(num_grid_points);
    }
    NewtonianMhd::PrimitiveFromConservative::apply(
        make_not_null(&get<MassDensity>(*prim_vars)),
        make_not_null(&get<Velocity>(*prim_vars)),
        make_not_null(&get<SpecificInternalEnergy>(*prim_vars)),
        make_not_null(&get<Pressure>(*prim_vars)),
        make_not_null(&get<MagneticField>(*prim_vars)),
        make_not_null(&get<DivergenceCleaningField>(*prim_vars)),
        mass_density_cons, momentum_density, energy_density,
        magnetic_field_cons, divergence_cleaning_field_cons, equation_of_state);
  }
}

#define INSTANTIATION(r, data) INSTANTIATION(~, ~)
#undef INSTANTIATION
}  // namespace NewtonianMhd::subcell
