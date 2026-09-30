// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

template <bool UseBg, typename Correction>
void test_conservation(const gsl::not_null<std::mt19937*> gen,
                       const size_t num_pts, const Correction& correction) {
  const EquationsOfState::IdealFluid<false> equation_of_state{1.3};
  const tuples::TaggedTuple<hydro::Tags::EquationOfState<false, 2>,
                            NewtonianMhd::Tags::DivergenceCleaningSpeed>
      volume_data{equation_of_state.get_clone(), 1.5};
  const tuples::TaggedTuple<
      helpers::Tags::Range<NewtonianMhd::Tags::MassDensityCons>,
      helpers::Tags::Range<hydro::Tags::SpecificInternalEnergy<DataVector>>>
      ranges{std::array{1.0e-2, 1.0}, std::array{1.0e-2, 1.0}};

  helpers::test_boundary_correction_conservation<NewtonianMhd::System<UseBg>>(
      gen, correction,
      Mesh<3 - 1>{num_pts, Spectral::Basis::Legendre,
                  Spectral::Quadrature::Gauss},
      volume_data, ranges);
}

template <bool UseBg>
void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts) {
  test_conservation<UseBg>(gen, num_pts,
                           NewtonianMhd::BoundaryCorrections::Hll<UseBg>{});
  test_conservation<UseBg>(gen, num_pts,
                           NewtonianMhd::BoundaryCorrections::Rusanov<UseBg>{});

  const auto hll = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      NewtonianMhd::BoundaryCorrections::Hll<UseBg>>("Hll:");
  test_conservation<UseBg>(
      gen, num_pts,
      dynamic_cast<const NewtonianMhd::BoundaryCorrections::Hll<UseBg>&>(*hll));

  const auto rusanov = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      NewtonianMhd::BoundaryCorrections::Rusanov<UseBg>>("Rusanov:");
  test_conservation<UseBg>(
      gen, num_pts,
      dynamic_cast<const NewtonianMhd::BoundaryCorrections::Rusanov<UseBg>&>(
          *rusanov));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.BoundaryCorrections",
                  "[Unit][Evolution]") {
  PUPable_reg(SINGLE_ARG(NewtonianMhd::BoundaryCorrections::Hll<true>));
  PUPable_reg(SINGLE_ARG(NewtonianMhd::BoundaryCorrections::Rusanov<true>));
  PUPable_reg(SINGLE_ARG(NewtonianMhd::BoundaryCorrections::Hll<false>));
  PUPable_reg(SINGLE_ARG(NewtonianMhd::BoundaryCorrections::Rusanov<false>));

  MAKE_GENERATOR(gen);
  test<true>(make_not_null(&gen), 5);
  test<false>(make_not_null(&gen), 5);
}
