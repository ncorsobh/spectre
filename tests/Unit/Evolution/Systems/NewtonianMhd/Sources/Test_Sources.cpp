// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Source.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace {
constexpr size_t dim = 3;

// `TimeDerivativeTerms` hands the source zeroed accumulators and expects it to
// add to them, so a source that adds nothing must leave them at zero.
void test_no_source() {
  const size_t num_points = 3;
  const NewtonianMhd::Sources::NoSource<> source{};
  const EquationsOfState::IdealFluid<false> equation_of_state{5.0 / 3.0};

  Scalar<DataVector> source_mass_density{DataVector{num_points, 0.0}};
  tnsr::I<DataVector, dim> source_momentum_density{DataVector{num_points, 0.0}};
  Scalar<DataVector> source_energy_density{DataVector{num_points, 0.0}};
  tnsr::I<DataVector, dim> source_magnetic_field{DataVector{num_points, 0.0}};
  Scalar<DataVector> source_divergence_cleaning_field{
      DataVector{num_points, 0.0}};

  const Scalar<DataVector> mass_density_cons{DataVector{num_points, 3.0}};
  const tnsr::I<DataVector, dim> momentum_density{DataVector{num_points, 0.4}};
  const Scalar<DataVector> energy_density{DataVector{num_points, 6.0}};
  const tnsr::I<DataVector, dim> magnetic_field{DataVector{num_points, 0.3}};
  const Scalar<DataVector> divergence_cleaning_field{
      DataVector{num_points, 0.5}};
  const tnsr::I<DataVector, dim> velocity{DataVector{num_points, 0.2}};
  const Scalar<DataVector> pressure{DataVector{num_points, 1.0}};
  const tnsr::I<DataVector, dim> coords{DataVector{num_points, 1.0}};

  source(make_not_null(&source_mass_density),
         make_not_null(&source_momentum_density),
         make_not_null(&source_energy_density),
         make_not_null(&source_magnetic_field),
         make_not_null(&source_divergence_cleaning_field), mass_density_cons,
         momentum_density, energy_density, magnetic_field,
         divergence_cleaning_field, velocity, pressure,
         NewtonianMhd::NoBackgroundMagneticField{}, equation_of_state, coords,
         0.0);

  const DataVector zero{num_points, 0.0};
  CHECK(get(source_mass_density) == zero);
  CHECK(get(source_energy_density) == zero);
  CHECK(get(source_divergence_cleaning_field) == zero);
  for (size_t i = 0; i < dim; ++i) {
    CHECK(source_momentum_density.get(i) == zero);
    CHECK(source_magnetic_field.get(i) == zero);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.Sources", "[Unit][Evolution]") {
  test_no_source();
}
