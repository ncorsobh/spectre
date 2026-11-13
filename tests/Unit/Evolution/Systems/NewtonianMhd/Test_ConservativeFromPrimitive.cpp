// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <size_t Dim>
void test_conservative_from_primitive(const DataVector& used_for_size) {
  // `apply` is overloaded on whether the magnetic field is evolved, so the
  // signature has to be named to take its address.
  using Hydro =
      void (*)(gsl::not_null<Scalar<DataVector>*>,
               gsl::not_null<tnsr::I<DataVector, Dim>*>,
               gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
               const tnsr::I<DataVector, Dim>&, const Scalar<DataVector>&);
  pypp::check_with_random_values<3>(
      static_cast<Hydro>(
          &NewtonianMhd::ConservativeFromPrimitive<Dim, false>::apply),
      "ConservativeFromPrimitive",
      {"mass_density_cons", "momentum_density", "energy_density"},
      {{{-1.0, 1.0}, {-2.0, 2.0}, {-3.0, 3.0}}}, used_for_size);
}

void test_conservative_from_primitive_mhd(const DataVector& used_for_size) {
  using Mhd =
      void (*)(gsl::not_null<Scalar<DataVector>*>,
               gsl::not_null<tnsr::I<DataVector, 3>*>,
               gsl::not_null<Scalar<DataVector>*>,
               gsl::not_null<tnsr::I<DataVector, 3>*>,
               gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
               const tnsr::I<DataVector, 3>&, const Scalar<DataVector>&,
               const tnsr::I<DataVector, 3>&, const Scalar<DataVector>&);
  pypp::check_with_random_values<5>(
      static_cast<Mhd>(
          &NewtonianMhd::ConservativeFromPrimitive<3, true>::apply),
      "ConservativeFromPrimitive",
      {"mhd_mass_density_cons", "mhd_momentum_density", "mhd_energy_density",
       "mhd_magnetic_field_cons", "mhd_divergence_cleaning_field_cons"},
      {{{-1.0, 1.0}, {-2.0, 2.0}, {-3.0, 3.0}, {-1.5, 1.5}, {-1.0, 1.0}}},
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianMhd.ConservativeFromPrimitive",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianMhd"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_conservative_from_primitive, (1, 2, 3))
  test_conservative_from_primitive_mhd(dv);
}
