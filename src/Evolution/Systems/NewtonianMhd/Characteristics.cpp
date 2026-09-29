// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Characteristics.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace NewtonianMhd {

template <size_t Dim, bool UseBackgroundMagneticField>
void fast_magnetosonic_speed(
    const gsl::not_null<Scalar<DataVector>*> fast_speed,
    const Scalar<DataVector>& mass_density,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
        background_magnetic_field) {
  set_number_of_grid_points(fast_speed, mass_density);
  get(*fast_speed) = get(sound_speed_squared);
  for (size_t i = 0; i < Dim; ++i) {
    if constexpr (UseBackgroundMagneticField) {
      get(*fast_speed) +=
          square(background_magnetic_field.get(i) + magnetic_field.get(i)) /
          get(mass_density);
    } else {
      get(*fast_speed) += square(magnetic_field.get(i)) / get(mass_density);
    }
  }
  get(*fast_speed) = sqrt(get(*fast_speed));
}

template <size_t Dim, bool UseBackgroundMagneticField>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, (2 * Dim) + 3>*> char_speeds,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const tnsr::i<DataVector, Dim>& normal,
    const double divergence_cleaning_speed,
    const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
        background_magnetic_field) {
  constexpr size_t num_speeds = (2 * Dim) + 3;

  Scalar<DataVector> fast_speed{};
  fast_magnetosonic_speed<Dim, UseBackgroundMagneticField>(
      make_not_null(&fast_speed), mass_density, sound_speed_squared,
      magnetic_field, background_magnetic_field);

  // Normal component of the total Alfven speed, c_An = |B_tot . n| / sqrt(rho)
  DataVector normal_alfven_speed(get(mass_density).size(), 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    normal_alfven_speed += magnetic_field.get(i) * normal.get(i);
    if constexpr (UseBackgroundMagneticField) {
      normal_alfven_speed += background_magnetic_field.get(i) * normal.get(i);
    }
  }
  normal_alfven_speed = abs(normal_alfven_speed) / sqrt(get(mass_density));

  auto& speeds = *char_speeds;
  speeds =
      make_array<num_speeds>(DataVector(get(dot_product(velocity, normal))));

  speeds[0] = -divergence_cleaning_speed;
  speeds[num_speeds - 1] = divergence_cleaning_speed;
  speeds[1] -= get(fast_speed);
  speeds[num_speeds - 2] += get(fast_speed);
  if constexpr (Dim >= 2) {
    speeds[2] -= normal_alfven_speed;
    speeds[num_speeds - 3] += normal_alfven_speed;
  }
  if constexpr (Dim >= 3) {
    // The slow magnetosonic speed is bounded above by both the sound speed and
    // the normal Alfven speed; this bound is exact in the limits of parallel
    // and perpendicular propagation.
    const DataVector slow_speed =
        blaze::min(sqrt(get(sound_speed_squared)), normal_alfven_speed);
    speeds[3] -= slow_speed;
    speeds[num_speeds - 4] += slow_speed;
  }
}

template <size_t Dim, bool UseBackgroundMagneticField>
std::array<DataVector, (2 * Dim) + 3> characteristic_speeds(
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const tnsr::i<DataVector, Dim>& normal,
    const double divergence_cleaning_speed,
    const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
        background_magnetic_field) {
  std::array<DataVector, (2 * Dim) + 3> char_speeds{};
  characteristic_speeds<Dim, UseBackgroundMagneticField>(
      make_not_null(&char_speeds), mass_density, velocity, sound_speed_squared,
      magnetic_field, normal, divergence_cleaning_speed,
      background_magnetic_field);
  return char_speeds;
}

}  // namespace NewtonianMhd

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define USE_BG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template void                                                                \
  NewtonianMhd::fast_magnetosonic_speed<DIM(data), USE_BG(data)>(              \
      gsl::not_null<Scalar<DataVector>*> fast_speed,                           \
      const Scalar<DataVector>& mass_density,                                  \
      const Scalar<DataVector>& sound_speed_squared,                           \
      const tnsr::I<DataVector, DIM(data)>& magnetic_field,                    \
      NewtonianMhd::BackgroundMagneticFieldArgument<DIM(data), USE_BG(data)>   \
          background_magnetic_field);                                          \
  template void NewtonianMhd::characteristic_speeds<DIM(data), USE_BG(data)>(  \
      gsl::not_null<std::array<DataVector, (2 * DIM(data)) + 3>*> char_speeds, \
      const Scalar<DataVector>& mass_density,                                  \
      const tnsr::I<DataVector, DIM(data)>& velocity,                          \
      const Scalar<DataVector>& sound_speed_squared,                           \
      const tnsr::I<DataVector, DIM(data)>& magnetic_field,                    \
      const tnsr::i<DataVector, DIM(data)>& normal,                            \
      double divergence_cleaning_speed,                                        \
      NewtonianMhd::BackgroundMagneticFieldArgument<DIM(data), USE_BG(data)>   \
          background_magnetic_field);                                          \
  template std::array<DataVector, (2 * DIM(data)) + 3>                         \
  NewtonianMhd::characteristic_speeds<DIM(data), USE_BG(data)>(                \
      const Scalar<DataVector>& mass_density,                                  \
      const tnsr::I<DataVector, DIM(data)>& velocity,                          \
      const Scalar<DataVector>& sound_speed_squared,                           \
      const tnsr::I<DataVector, DIM(data)>& magnetic_field,                    \
      const tnsr::i<DataVector, DIM(data)>& normal,                            \
      double divergence_cleaning_speed,                                        \
      NewtonianMhd::BackgroundMagneticFieldArgument<DIM(data), USE_BG(data)>   \
          background_magnetic_field);                                          \
  template struct NewtonianMhd::Tags::FastMagnetosonicSpeedCompute<            \
      DIM(data), USE_BG(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (true, false))

#undef DIM
#undef USE_BG
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template struct NewtonianMhd::Tags::ComputeLargestCharacteristicSpeed<DIM( \
      data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
