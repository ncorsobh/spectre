// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/AnalyticalThermal.hpp"

#include <algorithm>
#include <blaze/Blaze.h>
#include <cmath>
#include <utility>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
// Used for the symmetry energy to transition to a constant value at low
// densities
template <typename DataType>
DataType tanh_transition_function(const DataType& rest_mass_density,
                                  double onset, double scale) {
  return tanh(scale * (rest_mass_density - onset));
}

}  // namespace
namespace EquationsOfState {
template <typename ColdEquationOfState>
AnalyticalThermal<ColdEquationOfState>::AnalyticalThermal(
    ColdEquationOfState cold_eos, const double S0, const double L,
    const double gamma, const double n0, const double alpha)
    : cold_eos_(std::move(cold_eos)),
      S0_(S0),
      L_(L),
      gamma_(gamma),
      n0_(n0),
      alpha_(alpha) {
  eta_ = get_eta();
}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquationOfState>,
                                     AnalyticalThermal<ColdEquationOfState>,
                                     double, 3)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquationOfState>,
                                     AnalyticalThermal<ColdEquationOfState>,
                                     DataVector, 3)

template <typename ColdEquationOfState>
std::unique_ptr<
    EquationOfState<AnalyticalThermal<ColdEquationOfState>::is_relativistic, 3>>
AnalyticalThermal<ColdEquationOfState>::get_clone() const {
  auto clone = std::make_unique<AnalyticalThermal<ColdEquationOfState>>(*this);
  return std::unique_ptr<EquationOfState<is_relativistic, 3>>(std::move(clone));
}

template <typename ColdEquationOfState>
bool AnalyticalThermal<ColdEquationOfState>::operator==(
    const AnalyticalThermal<ColdEquationOfState>& rhs) const {
  return cold_eos_ == rhs.cold_eos_ and S0_ == rhs.S0_ and L_ == rhs.L_ and
         gamma_ == rhs.gamma_ and n0_ == rhs.n0_ and alpha_ == rhs.alpha_;
}
template <typename ColdEquationOfState>
bool AnalyticalThermal<ColdEquationOfState>::operator!=(
    const AnalyticalThermal<ColdEquationOfState>& rhs) const {
  return not(*this == rhs);
}

template <typename ColdEquationOfState>
bool AnalyticalThermal<ColdEquationOfState>::is_equal(
    const EquationOfState<is_relativistic, 3>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const AnalyticalThermal<ColdEquationOfState>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename ColdEquationOfState>
AnalyticalThermal<ColdEquationOfState>::AnalyticalThermal(CkMigrateMessage* msg)
    : EquationOfState<is_relativistic, 3>(msg) {}

template <typename ColdEquationOfState>
void AnalyticalThermal<ColdEquationOfState>::pup(PUP::er& p) {
  EquationOfState<is_relativistic, 3>::pup(p);
  p | cold_eos_;
  p | S0_;
  p | L_;
  p | gamma_;
  p | n0_;
  p | alpha_;
  p | eta_;
}
template <class ColdEos>
double AnalyticalThermal<ColdEos>::get_eta() const {
  return 5.0 / 9.0 * (L_ - 3 * S0_ * gamma_) /
         ((cbrt(0.25) - 1) * (2.0 / 3.0 - gamma_) *
          baryonic_fermi_internal_energy(saturation_density_));
}
template <class ColdEos>
template <class DataType>
DataType AnalyticalThermal<ColdEos>::baryonic_fermi_internal_energy(
    const DataType& rest_mass_density) const {
  return square(hbar_over_baryon_mass_to_four_thirds_) / 2.0 *
         pow(3 * square(M_PI) * rest_mass_density, 2.0 / 3.0);
}
template <class ColdEos>
template <typename DataType>
DataType AnalyticalThermal<ColdEos>::radiation_f_from_temperature(
    const DataType& temperature) const {
  // (See Eq. 4 of PHYS. REV. D 104, 063016 (2021))
  // .5 and 1.0 MeV are given in the paper.
  auto clamp = [](const auto& elt) {
    return std::clamp(elt - 0.5 / baryon_mass_in_mev_, 0.0,
                      0.5 / baryon_mass_in_mev_);
  };
  if constexpr (std::is_same_v<DataType, double>) {
    return 1 + 3.5 * baryon_mass_in_mev_ * clamp(temperature);
  } else {
    auto fs = make_with_value<DataType>(temperature, 0.0);
    std::transform(temperature.begin(), temperature.end(), fs.begin(), clamp);
    // Don't return a blaze expression without a definite type
    fs = 3.5 * baryon_mass_in_mev_ * fs + 1.0;
    return fs;
  }
}

template <class ColdEos>
template <class DataType>
DataType AnalyticalThermal<ColdEos>::thermal_internal_energy(
    const DataType& rest_mass_density, const DataType& temperature,
    const DataType& electron_fraction) const {
  // Following box 1
  const DataType fs = radiation_f_from_temperature(temperature);
  const DataType radiation = 4.0 * stefan_boltzmann_sigma_ * fs *
                             pow(temperature, 4) / rest_mass_density;

  // Factor out a `temperature` from the ideal and degenerate terms
  const double ideal = 1.5;

  const DataType baryon_degenerate = (a_degeneracy(
      rest_mass_density, make_with_value<DataType>(rest_mass_density, 0.5),
      dirac_effective_mass(rest_mass_density)));
  // Electron fraction fixed to 1 here to match paper results
  const DataType electron_degenerate =
      electron_fraction *
      a_degeneracy(static_cast<DataType>(electron_fraction * rest_mass_density),
                   make_with_value<DataType>(rest_mass_density, 1.0),
                   make_with_value<DataType>(rest_mass_density,
                                             electron_mass_over_baryon_mass_));
  const DataType degenerate =
      (baryon_degenerate + electron_degenerate) * temperature;
  return radiation + temperature * ideal * degenerate / (ideal + degenerate);
}
template <class ColdEos>
template <class DataType>
DataType AnalyticalThermal<ColdEos>::thermal_pressure(
    const DataType& rest_mass_density, const DataType& temperature,
    const DataType& electron_fraction) const {
  // Following box 2
  const DataType fs = radiation_f_from_temperature(temperature);
  const DataType radiation =
      4.0 / 3.0 * stefan_boltzmann_sigma_ * fs * pow(temperature, 4);
  const double ideal = 1.0;
  // The paper supresses some notation in this calculation, this is checked
  // against the results of the paper
  const DataType degenerate =
      -rest_mass_density * temperature *
      (a_degeneracy_density_derivative(
           rest_mass_density, make_with_value<DataType>(rest_mass_density, 0.5),
           dirac_effective_mass(rest_mass_density)) +
       a_degeneracy_density_derivative(
           static_cast<DataType>(electron_fraction * rest_mass_density),
           make_with_value<DataType>(rest_mass_density, 1.0),
           electron_mass_over_baryon_mass_, true) *
           square(electron_fraction));
  return radiation + rest_mass_density * temperature * ideal * degenerate /
                         (ideal + degenerate);
}
template <class ColdEos>
template <class DataType>
DataType AnalyticalThermal<ColdEos>::thermal_pressure_density_derivative(
    const DataType& rest_mass_density, const DataType& temperature,
    const DataType& electron_fraction) const {
  const DataType fs = radiation_f_from_temperature(temperature);
  const DataType radiation = 16.0 * stefan_boltzmann_sigma_ * fs *
                             pow(temperature, 4) / (9.0 * rest_mass_density);
  // See Eq. 56 of 1902.10735
  const double ideal = 5.0 / 3.0;
  const DataType baryon_effective_mass =
      dirac_effective_mass(rest_mass_density);
  const DataType a_electron = a_degeneracy(static_cast<DataType>(rest_mass_density * electron_fraction), 
            make_with_value<DataType>(rest_mass_density,1.0),
                                           electron_mass_over_baryon_mass_);
  const DataType a_SM =
      a_degeneracy(rest_mass_density, make_with_value<DataType>(rest_mass_density, 0.5), baryon_effective_mass);
  auto compute_degeneracy_pressure_derivative = [this, &a_SM, &a_electron, &baryon_effective_mass,
  &rest_mass_density, &temperature, &electron_fraction ](){
  const DataType a_electron_density_deriv =
      a_degeneracy_density_derivative(static_cast<DataType>(rest_mass_density * electron_fraction), make_with_value<DataType>(rest_mass_density,1.0),
                                          electron_mass_over_baryon_mass_,
                                          true);
  const DataType a_SM_density_deriv = a_degeneracy_density_derivative(
      rest_mass_density, make_with_value<DataType>(rest_mass_density, 0.5), baryon_effective_mass);
  const DataType degenerate_temperature_dependent =
      -2 * (a_SM_density_deriv + a_electron_density_deriv * square(electron_fraction)) * square(rest_mass_density) * square(temperature) * 
    
      (1.0/rest_mass_density -   (a_SM_density_deriv + a_electron_density_deriv *
       electron_fraction) /
      (a_SM + a_electron * electron_fraction));
  const DataType A_baryon = -alpha_ * (1 - square(baryon_effective_mass));
   
  const DataType C_baryon =
      pow(3.0 * square(M_PI)  * rest_mass_density,
          2.0 / 3.0) *
      square(hbar_over_baryon_mass_to_four_thirds_ / baryon_effective_mass);
  const DataType B_baryon = 1 / (1 + C_baryon);
  const DataType rho_times_a_SM_second_log_density_deriv =
      1.0 * a_SM_density_deriv * (rest_mass_density * a_SM_density_deriv / a_SM - 1) +
      2.0 * a_SM / 3.0 / rest_mass_density * B_baryon *
          (3.0 * square(A_baryon) - 1.0 / 3.0 * B_baryon * square(3.0 * A_baryon + C_baryon) +
           1.0 / 3.0 * C_baryon + 3 * alpha_ *
          square(baryon_effective_mass) * A_baryon);
  const DataType C_electron = pow(3.0 * square(M_PI) * electron_fraction * rest_mass_density, 2.0/3.0) * 
  square(hbar_over_baryon_mass_to_four_thirds_) / square(electron_mass_over_baryon_mass_);  
  const DataType B_electron = 1 / (1 + C_electron); 
  const DataType rho_times_a_electron_second_log_density_deriv =
      a_electron_density_deriv * (rest_mass_density * a_electron_density_deriv / a_electron  - 1) +
      2.0 * a_electron / 9.0 / rest_mass_density * B_electron * C_electron * (1 - B_electron* C_electron);
  const DataType degenerate =
      degenerate_temperature_dependent -
      square(temperature) * rest_mass_density * (rho_times_a_SM_second_log_density_deriv +
                     rho_times_a_electron_second_log_density_deriv * electron_fraction);
return degenerate;
};
  // Here we use piecewise definition of the thermal pressure 
  // rather than the smoothed version (as in the paper) 
 if constexpr(std::is_same_v<DataType, double>){
    return   temperature * (a_electron * electron_fraction + a_SM) * square(temperature) > .5 * temperature ? ideal * temperature :  
  compute_degeneracy_pressure_derivative();}
else{
  return radiation +  select(step_function((a_electron*electron_fraction + a_SM) * square(temperature) - 1.5 * temperature ), ideal * temperature, 
  compute_degeneracy_pressure_derivative());}
}

template <class ColdEos>
template <class DataType>
DataType AnalyticalThermal<ColdEos>::composition_dependent_internal_energy(
    const DataType& rest_mass_density,
    const DataType& electron_fraction) const {
  // Box 1
  const DataType equilibrium_electron_fraction =
      beta_equalibrium_proton_fraction(rest_mass_density);
  DataType result =
      3.0 * K_ *
      (cbrt(pow(electron_fraction, 4) * rest_mass_density) -
       cbrt(pow(equilibrium_electron_fraction, 4) * rest_mass_density));
  result +=
      symmetry_energy_at_zero_temp(rest_mass_density) * 4 *
      (electron_fraction * (electron_fraction - 1.0) -
       equilibrium_electron_fraction * (equilibrium_electron_fraction - 1.0));
  return result;
}
template <class ColdEos>
template <class DataType>
DataType AnalyticalThermal<ColdEos>::composition_dependent_pressure(
    const DataType& rest_mass_density,
    const DataType& electron_fraction) const {
  const DataType equilibrium_electron_fraction =
      beta_equalibrium_proton_fraction(rest_mass_density);

  return K_ * rest_mass_density *
             (cbrt(pow(electron_fraction, 4) * rest_mass_density) -
              cbrt(pow(equilibrium_electron_fraction, 4) * rest_mass_density)) +
         symmetry_pressure_at_zero_temp(rest_mass_density) * 4 *
             (electron_fraction * (electron_fraction - 1.0) -
              equilibrium_electron_fraction *
                  (equilibrium_electron_fraction - 1.0));
}

template <typename ColdEos>
template <typename DataType>
DataType AnalyticalThermal<ColdEos>::dirac_effective_mass(
    const DataType& rest_mass_density) const {
  return 930.6 / baryon_mass_in_mev_ *
         invsqrt(1 + pow((rest_mass_density / n0_), alpha_ * 2));
}

template <typename ColdEos>
template <typename DataType>
DataType AnalyticalThermal<ColdEos>::symmetry_energy_at_zero_temp(
    const DataType& rest_mass_density) const {
  // Transition to SFHo like-low density values
  double transition_density = 0.5 * saturation_density_;
  // Pointwise take value for each density
  using std::max;
  DataType effective_density = max(rest_mass_density, transition_density);
  // Power law fall off for low-density.  Completely arbitrary, tuned by
  // hand
  double gamma_PL = 1.529;
  // See appendix A of 2107.06804
  auto transition_function = [](const DataType& density) {
    // convert into nuclear-theory units inside
    return (1 + tanh(40 * (.16 / saturation_density_ * density - 0.16))) / 2;
  };
  // Erratum, Eq. (17)
  double common_factor = 0.6 * (cbrt(.25) - 1.0);
  DataType kinetic_symmetry_component =
      common_factor * baryonic_fermi_internal_energy(effective_density);
  double kinetic_symmetry_component_at_saturation =
      common_factor * baryonic_fermi_internal_energy(saturation_density_);
  // Eq. (14)
  DataType common_expression =
      eta_ * kinetic_symmetry_component +
      (S0_ - eta_ * kinetic_symmetry_component_at_saturation) *
          pow(effective_density / saturation_density_, gamma_);
  if (min(rest_mass_density) >= transition_density) {
    return common_expression;
  } else {
    if constexpr (std::is_same_v<DataType, double>) {
      return rest_mass_density >= transition_density
                 ? common_expression
                 : (1 - transition_function(rest_mass_density)) *
                           common_expression +
                       transition_function(rest_mass_density) *
                           (common_expression +
                            symmetry_pressure_at_zero_temp(rest_mass_density) /
                                (transition_density * (gamma_PL - 1.0)) *
                                (pow(rest_mass_density / transition_density,
                                     gamma_PL - 1.0) -
                                 1.0));
    } else {
      // Complicated expression for low density
      return select(
          step_function(rest_mass_density - transition_density),
          common_expression,
          (1 - transition_function(rest_mass_density)) * common_expression +
              transition_function(rest_mass_density) *
                  (common_expression +
                   symmetry_pressure_at_zero_temp(rest_mass_density) /
                       (transition_density * (gamma_PL - 1.0)) *
                       (pow(rest_mass_density / transition_density,
                            gamma_PL - 1.0) -
                        1.0)));
    }
  }
}
template <class ColdEos>
template <class DataType>
DataType AnalyticalThermal<ColdEos>::beta_equalibrium_proton_fraction(
    const DataType& rest_mass_density) const {
  // Eq. (20) recast with z = (1 - 2 Y_{p,beta}) can be solved
  // via Cardano's formula.  I don't know if this is fast
  // but it's certainly simpler than unpacking and rootfinding
  DataType A = 64 / (3 * square(M_PI) * rest_mass_density) *
               cube(symmetry_energy_at_zero_temp(rest_mass_density) /
                    hbar_over_baryon_mass_to_four_thirds_);
  // z^3 + pz + q = 0; p = 1/(2A), q = -1/(2A)
  DataType p = 1 / (2 * A);
  // q = -p
  DataType sqrtDelta = p * sqrt(0.25 + p / 27.0);
  // return (1-z) / 2
  return (1.0 - cbrt(p * 0.5 + sqrtDelta) - cbrt(p * 0.5 - sqrtDelta)) / 2.0;
}
template <class ColdEos>
template <typename DataType, typename MassType>
DataType AnalyticalThermal<ColdEos>::a_degeneracy(
    const DataType& rest_mass_density, const DataType& particle_fraction,
    const MassType& mass) const {
  const DataType kinetic_common_factor =
      pow(3 * square(M_PI) * particle_fraction * rest_mass_density, 2.0 / 3.0) *
      square(hbar_over_baryon_mass_to_four_thirds_);
  return square(M_PI) / 2 * sqrt(kinetic_common_factor + square(mass)) /
         kinetic_common_factor;
}
template <class ColdEos>
template <class DataType, class MassType>
DataType AnalyticalThermal<ColdEos>::a_degeneracy_density_derivative(
    const DataType& rest_mass_density, const DataType& particle_fraction,
    const MassType& mass, bool is_electron) const {
  MassType log_mass_derivative_log_density =
      is_electron ? make_with_value<MassType>(mass, 0.0)
                  : -alpha_ * (1 - square(mass));
  DataType kinetic_common_factor =
      pow(3.0 * square(M_PI) * particle_fraction * rest_mass_density,
          2.0 / 3.0) *
      square(hbar_over_baryon_mass_to_four_thirds_ / mass);

  return -2.0 * a_degeneracy(rest_mass_density, particle_fraction, mass) /
         (3 * rest_mass_density) *
         (1.0 - .5 * (1.0 / (1.0 + kinetic_common_factor) *
                      (kinetic_common_factor +
                       3.0 * log_mass_derivative_log_density)));
}
template <class ColdEos>
template <class DataType>
DataType AnalyticalThermal<ColdEos>::symmetry_pressure_at_zero_temp(
    const DataType& rest_mass_density) const {
  double transition_density = 0.5 * saturation_density_;
  // Pointwise take value for each density
  using std::max;
  DataType effective_density = max(rest_mass_density, transition_density);
  // Power law fall off for low-density.  Completely arbitrary, tuned by
  // hand
  double gamma_PL = 1.529;
  // See appendix A of 2107.06804
  auto transition_function = [](const DataType& density) {
    // convert into nuclear-theory units inside
    return (1 + tanh(40 * (.16 / saturation_density_ * density - 0.16))) / 2;
  };
  double common_factor = 0.6 * (pow(.25, 1.0 / 3.0) - 1.0);
  DataType kinetic_symmetry_component =
      common_factor * baryonic_fermi_internal_energy(rest_mass_density);
  double kinetic_symmetry_component_at_saturation =
      common_factor * baryonic_fermi_internal_energy(saturation_density_);
  DataType common_expression =
      effective_density *
      (2.0 * eta_ / 3.0 * kinetic_symmetry_component +
       (S0_ - eta_ * kinetic_symmetry_component_at_saturation) *
           pow(effective_density / saturation_density_, gamma_) * gamma_);
  if (min(rest_mass_density) >= transition_density) {
    // Either DataVector completely above transition density or double above it
    return common_expression;
  } else {
    if constexpr (std::is_same_v<DataType, double>) {
      return transition_function(rest_mass_density) * common_expression *
             pow(rest_mass_density / transition_density, gamma_PL);
    } else {
      // DataVector
      return select(step_function(rest_mass_density - transition_density),
                    common_expression,
                    transition_function(rest_mass_density) * common_expression *
                        pow(rest_mass_density / transition_density, gamma_PL));
    }
  }
}
template <class ColdEos>
template <class DataType>
DataType
AnalyticalThermal<ColdEos>::symmetry_pressure_density_derivative_at_zero_temp(
    const DataType& rest_mass_density) const {
 double transition_density = 0.5 * saturation_density_;
  // Pointwise take value for each density
  using std::max;
  DataType effective_density = max(rest_mass_density, transition_density);
  // Power law fall off for low-density.  Completely arbitrary, tuned by
  // hand
  double gamma_PL = 1.529;
  // See appendix A of 2107.06804
  auto transition_function = [](const DataType& density) {
    // convert into nuclear-theory units inside
    return (1 + tanh(40 * (.16 / saturation_density_ * density - 0.16))) / 2;
  };
  auto derivative_transition_function = [](const DataType& density){
    return 40 * 0.16 / saturation_density_ * pow(1 / cosh(40 * (.16 / saturation_density_ * density - 0.16)), 2) / 2;
  };
  double common_factor = 0.6 * (pow(.25, 1.0 / 3.0) - 1.0);
  DataType kinetic_symmetry_component =
      common_factor * baryonic_fermi_internal_energy(effective_density);
  double kinetic_symmetry_component_at_saturation =
      common_factor * baryonic_fermi_internal_energy(saturation_density_);

  DataType common_expression =
       
         (10.0 / 9.0 * eta_ * kinetic_symmetry_component +
          (S0_ - eta_ * kinetic_symmetry_component_at_saturation) *
              (gamma_ * (1.0 + gamma_)) *
              pow(effective_density / saturation_density_, gamma_));
  if (min(rest_mass_density) >= transition_density) {
    // Either DataVector completely above transition density or double above it
    return common_expression;
  } else {
    if constexpr (std::is_same_v<DataType, double>) {
      return transition_function(rest_mass_density) * common_expression *
             pow(rest_mass_density / transition_density, gamma_PL);
    } else {
      // DataVector
      return select(step_function(rest_mass_density - transition_density),
                    common_expression,
                   symmetry_pressure_at_zero_temp(effective_density) *(gamma_PL * transition_function(rest_mass_density) * 
                        pow(rest_mass_density / transition_density, (gamma_PL-1)) + derivative_transition_function(rest_mass_density)));
    }
  }
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> AnalyticalThermal<ColdEquationOfState>::
    pressure_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature,
        const Scalar<DataType>& electron_fraction) const {
  return Scalar<DataType>{

      get(cold_eos_.pressure_from_density(rest_mass_density)) +
      composition_dependent_pressure(get(rest_mass_density),
                                     get(electron_fraction)) +
      thermal_pressure(get(rest_mass_density), get(temperature),
                       get(electron_fraction))};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
AnalyticalThermal<ColdEquationOfState>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& electron_fraction) const {
  // Have to rootfind for temperature
  const Scalar<DataType> temperature = temperature_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  return pressure_from_density_and_temperature(rest_mass_density, temperature,
                                               electron_fraction);
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> AnalyticalThermal<ColdEquationOfState>::
    specific_internal_energy_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature,
        const Scalar<DataType>& electron_fraction) const {
  return Scalar<DataType>{
      get(cold_eos_.specific_internal_energy_from_density(rest_mass_density)) +
      composition_dependent_internal_energy(get(rest_mass_density),
                                            get(electron_fraction)) +
      thermal_internal_energy(get(rest_mass_density), get(temperature),
                              get(electron_fraction))};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> AnalyticalThermal<ColdEquationOfState>::
    temperature_from_density_and_energy_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& specific_internal_energy,
        const Scalar<DataType>& electron_fraction) const {
  const auto cold_energy =
      get(cold_eos_.specific_internal_energy_from_density(rest_mass_density));
  const auto composition_energy = composition_dependent_internal_energy(
      get(rest_mass_density), get(electron_fraction));
  const DataType thermal_energy_needed =
      get(specific_internal_energy) - cold_energy - composition_energy;
  const auto radiation_prefactor = [this, &rest_mass_density](
                                       const double& temperature, size_t i) {
    if constexpr (std::is_same_v<DataType, double>) {
      return static_cast<double>(4 * stefan_boltzmann_sigma_ *
                                 radiation_f_from_temperature(temperature) /
                                 get(rest_mass_density));
    } else {
      return static_cast<double>(4 * stefan_boltzmann_sigma_ *
                                 radiation_f_from_temperature(temperature) /
                                 get(rest_mass_density)[i]);
    }
  };
  double ideal_prefactor = 1.5;
  const DataType degenerate_prefactor =
      (a_degeneracy(get(rest_mass_density),
                    make_with_value<DataType>(get(rest_mass_density), 0.5),
                    dirac_effective_mass(get(rest_mass_density))) +
       a_degeneracy(static_cast<DataType>(get(rest_mass_density) *
                                          get(electron_fraction)),
                    make_with_value<DataType>(get(rest_mass_density), 1.0),
                    electron_mass_over_baryon_mass_) *
           get(electron_fraction));
  // Coefficients (temperature dependent) on rootfindng polynomial
  // A_Radiation * T^4 + A_{ideal-degen} * T^2 - epsilon_thermal = 0
  auto miss = [&thermal_energy_needed, &degenerate_prefactor, &ideal_prefactor,
               &radiation_prefactor](const double temperature, size_t i = 0) {
    if constexpr (std::is_same_v<DataType, double>) {
      std::vector<DataType> coefficients{
          -thermal_energy_needed,
          (degenerate_prefactor * ideal_prefactor) /
              (temperature * degenerate_prefactor + ideal_prefactor),
          radiation_prefactor(temperature, i)};
      return static_cast<double>(
          evaluate_polynomial(coefficients, square(temperature)));
    } else {
      std::vector<double> coefficients{
          {-thermal_energy_needed[i],
           1 / (temperature * degenerate_prefactor[i] + ideal_prefactor) *
               (degenerate_prefactor[i] * ideal_prefactor),
           radiation_prefactor(temperature, i)}};
      return static_cast<double>(
          evaluate_polynomial(coefficients, square(temperature)));
    }
  };
  // This could be made stricter
  double lower_bound = 0.0;
  double upper_bound = LIKELY(max(thermal_energy_needed)) < 1.0e2
                           ? 2000.0
                           : std::numeric_limits<double>::max();
  return Scalar<DataType>{RootFinder::toms748(
      miss, make_with_value<DataType>(rest_mass_density, lower_bound),
      make_with_value<DataType>(rest_mass_density, upper_bound), 1.0e-14,
      1.0e-15)};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> AnalyticalThermal<ColdEquationOfState>::
    sound_speed_squared_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature,
        const Scalar<DataType>& electron_fraction) const {
  const DataType pressure_derivative_cold =
      get(cold_eos_.chi_from_density(rest_mass_density));
  const DataType equilibrium_electron_fraction =
      beta_equalibrium_proton_fraction(get(rest_mass_density));
  const DataType pressure_derivative_composition =
      (K_ * 4.0 / 3.0) *
          (cbrt(pow(get(electron_fraction), 4) * get(rest_mass_density)) -
           cbrt(pow(equilibrium_electron_fraction, 4) *
                get(rest_mass_density))) +
      symmetry_pressure_density_derivative_at_zero_temp(
          get(rest_mass_density)) *
          4.0 *
          (get(electron_fraction) * (get(electron_fraction) - 1.0) -
           equilibrium_electron_fraction *
               (equilibrium_electron_fraction - 1.0));

  const DataType thermal_pressure_derivative =
      thermal_pressure_density_derivative(
          get(rest_mass_density), get(temperature), get(electron_fraction));
  const auto result = Scalar<DataType>{
      1.0 /
      (1.0 +
       get(pressure_from_density_and_temperature(rest_mass_density, temperature,
                                                 electron_fraction)) /
           get(rest_mass_density) +
       get(specific_internal_energy_from_density_and_temperature(
           rest_mass_density, temperature, electron_fraction))) *
      (pressure_derivative_cold +  pressure_derivative_composition +
       thermal_pressure_derivative)};
  // Don't trust the above calculation
  //return make_with_value<Scalar<DataType>>(rest_mass_density, 1.0);
  return result;
}
}  // namespace EquationsOfState

template class EquationsOfState::AnalyticalThermal<
    EquationsOfState::PolytropicFluid<true>>;
template class EquationsOfState::AnalyticalThermal<
    EquationsOfState::Enthalpy<EquationsOfState::Spectral>>;
