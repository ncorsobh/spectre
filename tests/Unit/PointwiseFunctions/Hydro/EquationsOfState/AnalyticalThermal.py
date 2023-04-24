# Distributed under the MIT License.
# See LICENSE.txt for details.

# from PolytropicFluid import polytropic_pressure_from_density
# from PolytropicFluid import polytropic_specific_internal_energy_from_density


from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

import matplotlib.cm as cm
from matplotlib.colors import Normalize

cmap = cm.viridis


class Polynomial:
    """
    A polynomial which can have coefficients which depend on the variable the
    polynomial will be evaluated on
    """

    def __init__(self, coeffs):
        self.coeffs_ = coeffs

    def __call__(self, x):
        return reduce(
            lambda acc, elt: x * (acc + elt),
            reversed(self.coeffs_),
            np.zeros_like(x),
        )


class PolytropicFluid:
    def __init__(self, polytropic_constant, polytropic_exponent):
        self.polytropic_constant_ = polytropic_constant
        self.polytropic_exponent_ = polytropic_exponent

    def pressure_from_density(self, rest_mass_density):
        return self.polytropic_constant_ * rest_mass_density ** (
            self.polytropic_exponent_
        )

    def specific_internal_energy_from_density(self, rest_mass_density):
        return (
            self.polytropic_constant_
            * rest_mass_density ** (self.polytropic_exponent_ - 1)
            / (self.polytropic_exponent_ - 1.0)
        )

    def chi_from_density(self, rest_mass_density):
        return (
            self.polytropic_constant_
            * self.polytropic_exponent_
            * rest_mass_density ** (self.polytropic_exponent_ - 1.0)
        )


# Defined same as in `AnalyticalThermal.hpp`
hbar_over_baryon_mass_to_four_thirds_ = 1.5060
saturation_density_ = 4.34e-4
stefan_boltzmann_sigma_ = (
    pow(np.pi, 2) / 60 * pow(1 / hbar_over_baryon_mass_to_four_thirds_, 3)
)
electron_mass_over_baryon_mass_ = 5.44e-4
baryon_mass_in_mev_ = 939.57  # Neutron mass for consistency with the paper
K_ = (
    np.cbrt(3 * np.square(np.pi)) * hbar_over_baryon_mass_to_four_thirds_ * 0.25
)
# Power-law exponent for return to SFHo low density
# composition dependence
gamma_PL_ = 1.527


def low_density_transition_function(rho):
    # See appendix A of 2107.06804
    X = 40 * 0.16 / saturation_density_
    rho_transition = 0.025 * saturation_density_ / 0.16
    return (1 + np.tanh(X * (rho - rho_transition))) / 2


def low_density_transition_function_derivative(rho):
    X = 40 * 0.16 / saturation_density_
    rho_transition = 0.025 * saturation_density_ / 0.16
    return X / 2 * (1 / np.cosh(X * (rho - rho_transition)) ** 2)


def baryonic_fermi_internal_energy(rest_mass_density):
    return (
        1
        / 2
        * (hbar_over_baryon_mass_to_four_thirds_) ** 2
        * pow(3 * np.pi**2 * rest_mass_density, 2 / 3)
    )


def E_kin_sym(rest_mass_density):
    return (
        3
        / 5
        * (2 ** (-2 / 3) - 1)
        * baryonic_fermi_internal_energy(rest_mass_density)
    )


def fs_of_temperature(temperature):
    """
    Radiation coefficient in thermal energy,
    related to number of Degrees of freedom.
    """
    return np.where(
        temperature < 0.5 / baryon_mass_in_mev_,
        1,
        np.where(
            temperature < 1 / baryon_mass_in_mev_,
            -0.75 + 3.5 * temperature / baryon_mass_in_mev_,
            2.75,
        ),
    )


class AnalyticalThermal:
    def __init__(self, S0, L, gamma, n0, alpha, cold_eos):
        self.S0_ = S0
        self.L_ = L
        self.gamma_ = gamma
        self.n0_ = n0
        self.alpha_ = alpha
        self.cold_eos_ = cold_eos
        self.get_eta()

    def get_eta(self):
        # See Erratum

        self.eta_ = (
            5
            / 9
            * (self.L_ - 3 * self.S0_ * self.gamma_)
            / (
                (2 ** (-2 / 3) - 1)
                * (2 / 3 - self.gamma_)
                * baryonic_fermi_internal_energy(saturation_density_)
            )
        )

    def get_symmetry_energy(self, rest_mass_density):
        transition_density = 0.5 * saturation_density_
        effective_density = np.maximum(rest_mass_density, transition_density)
        gamma_PL = gamma_PL_
        common_expression = self.eta_ * E_kin_sym(effective_density) + (
            self.S0_ - self.eta_ * E_kin_sym(saturation_density_)
        ) * pow(effective_density / saturation_density_, self.gamma_)
        return np.where(
            rest_mass_density >= transition_density,
            common_expression,
            (
                (1 - low_density_transition_function(rest_mass_density))
                * 0.5
                * common_expression
                + low_density_transition_function(rest_mass_density)
                * (
                    common_expression
                    + self.get_symmetry_pressure(transition_density)
                    / (transition_density * (gamma_PL - 1.0))
                    * (
                        (rest_mass_density / transition_density)
                        ** (gamma_PL - 1.0)
                        - 1.0
                    )
                )
            ),
        )

    def get_equilibrium_charge_fraction(self, rest_mass_density):
        # Calculated at zero T, assuming it doesn't change much
        coeff = (
            64
            / (3 * np.pi**2 * rest_mass_density)
            * (
                self.get_symmetry_energy(rest_mass_density)
                / hbar_over_baryon_mass_to_four_thirds_
            )
            ** 3
        )
        # define delta = (1 - 2 *Y_e)
        # (1-delta) = 2 * coeff * delta**3
        # delta**3 + delta/(2 * coeff) - 1/(2 * coeff) = 0
        p = 1 / (2 * coeff)
        q = -1 / (2 * coeff)
        # Cardano formula
        delta = np.cbrt(-q / 2 + np.sqrt(q**2 / 4 + p**3 / 27)) + np.cbrt(
            -q / 2 - np.sqrt(q**2 / 4 + p**3 / 27)
        )
        return (1 - delta) / 2

    def get_composition_dependent_energy(
        self, rest_mass_density, electron_fraction
    ):
        # Eq.12b, 12c, Box 1
        K = (3 * np.pi**2) ** (1 / 3) * (
            hbar_over_baryon_mass_to_four_thirds_ / 4
        )
        equilibrium_electron_fraction = self.get_equilibrium_charge_fraction(
            rest_mass_density
        )
        symmetry_energy = self.get_symmetry_energy(rest_mass_density)
        return symmetry_energy * (
            (1 - 2 * electron_fraction) ** 2
            - (1 - 2 * equilibrium_electron_fraction) ** 2
        ) + 3 * K * (
            electron_fraction ** (4 / 3)
            - equilibrium_electron_fraction ** (4 / 3)
        ) * rest_mass_density ** (
            1 / 3
        )

    def get_dirac_specific_mass(self, rest_mass_density):
        """
        Dirac mass per unit rest mass (:
        """
        return (
            930.6
            / baryon_mass_in_mev_
            * (1 + (rest_mass_density / (self.n0_)) ** (2 * self.alpha_))
            ** (-0.5)
        )

    def get_log_derivative_dirac_specific_mass(self, rest_mass_density):
        return -self.alpha_ * (
            1 - self.get_dirac_specific_mass(rest_mass_density) ** 2
        )

    def get_a_thermal(
        self, rest_mass_density, electron_fraction, dirac_specific_mass
    ):
        fermi_energy_common_factor = (
            3 * np.pi**2 * electron_fraction * rest_mass_density
        ) ** (2 / 3) * hbar_over_baryon_mass_to_four_thirds_**2
        return (
            np.pi**2
            / 2
            * np.sqrt(fermi_energy_common_factor + dirac_specific_mass**2)
            / fermi_energy_common_factor
        )

    def get_temperature_dependent_energy(
        self, rest_mass_density, temperature, electron_fraction
    ):
        fs = fs_of_temperature(temperature)
        radiation_part = (
            4
            * stefan_boltzmann_sigma_
            * fs
            * temperature**4
            / rest_mass_density
        )
        ideal_gas_part = 3 * temperature / 2
        baryon_degeneracy_a = self.get_a_thermal(
            rest_mass_density=rest_mass_density,
            electron_fraction=np.full_like(rest_mass_density, 0.5),
            dirac_specific_mass=self.get_dirac_specific_mass(
                rest_mass_density=rest_mass_density
            ),
        )
        electron_degeneracy_a = self.get_a_thermal(
            rest_mass_density=electron_fraction * rest_mass_density,
            electron_fraction=np.ones_like(rest_mass_density),
            dirac_specific_mass=electron_mass_over_baryon_mass_,
        ) * electron_fraction ** (1)

        degeneracy_part = (
            baryon_degeneracy_a + electron_degeneracy_a
        ) * temperature**2
        return (
            radiation_part
            + (ideal_gas_part**-1 + degeneracy_part**-1) ** -1
        )

    def specific_internal_energy_from_density_and_temperature(
        self, rest_mass_density, temperature, electron_fraction
    ):
        return (
            self.cold_eos_.specific_internal_energy_from_density(
                rest_mass_density
            )
            + self.get_composition_dependent_energy(
                rest_mass_density=rest_mass_density,
                electron_fraction=electron_fraction,
            )
            + self.get_temperature_dependent_energy(
                rest_mass_density=rest_mass_density,
                temperature=temperature,
                electron_fraction=electron_fraction,
            )
        )

    def get_symmetry_pressure(self, rest_mass_density):
        gamma_PL = gamma_PL_
        transition_density = 0.5 * saturation_density_
        effective_density = np.maximum(rest_mass_density, transition_density)
        common_expression = effective_density * (
            self.eta_ * 2 / 3 * E_kin_sym(rest_mass_density=effective_density)
            + (self.S0_ - self.eta_ * E_kin_sym(saturation_density_))
            * (effective_density / saturation_density_) ** (self.gamma_)
            * self.gamma_
        )
        return np.where(
            rest_mass_density > transition_density,
            common_expression,
            low_density_transition_function(rest_mass_density)
            * common_expression
            * (rest_mass_density / transition_density) ** (gamma_PL),
        )

    def get_composition_dependent_pressure(
        self, rest_mass_density, electron_fraction
    ):
        equilibrium_electron_fraction = self.get_equilibrium_charge_fraction(
            rest_mass_density=rest_mass_density
        )
        symmetry_pressure = self.get_symmetry_pressure(
            rest_mass_density=rest_mass_density
        )
        return K_ * (
            electron_fraction ** (4 / 3)
            - equilibrium_electron_fraction ** (4 / 3)
        ) * rest_mass_density ** (4 / 3) + symmetry_pressure * (
            (1 - 2 * electron_fraction) ** 2
            - (1 - 2 * equilibrium_electron_fraction) ** 2
        )

    def get_derivative_of_a_thermal(
        self,
        rest_mass_density,
        electron_fraction,
        dirac_specific_mass,
        is_electron=False,
    ):
        fermi_energy_common_factor = (
            3 * np.pi**2 * electron_fraction * rest_mass_density
        ) ** (2 / 3) * hbar_over_baryon_mass_to_four_thirds_**2
        return (
            -2
            * self.get_a_thermal(
                rest_mass_density=rest_mass_density,
                electron_fraction=electron_fraction,
                dirac_specific_mass=dirac_specific_mass,
            )
            / (3 * rest_mass_density)
            * (
                1.0
                - 1.0
                / 2.0
                * dirac_specific_mass**2
                / (dirac_specific_mass**2 + fermi_energy_common_factor)
                * (
                    fermi_energy_common_factor / dirac_specific_mass**2
                    + 3
                    * (
                        0.0
                        if is_electron
                        else self.get_log_derivative_dirac_specific_mass(
                            rest_mass_density=rest_mass_density
                        )
                    )
                )
            )
        )

    def get_symmetry_pressure_density_derivative(self, rest_mass_density):
        gamma_PL = gamma_PL_
        transition_density = 0.5 * saturation_density_
        effective_density = np.maximum(rest_mass_density, transition_density)
        common_expression = 10.0 / 9.0 * self.eta_ * E_kin_sym(
            rest_mass_density=effective_density
        ) + (self.S0_ - self.eta_ * E_kin_sym(saturation_density_)) * (
            effective_density / saturation_density_
        ) ** (
            self.gamma_
        ) * self.gamma_ * (
            self.gamma_ + 1
        )
        return np.where(
            rest_mass_density > transition_density,
            common_expression,
            self.get_symmetry_pressure(effective_density)
            * (
                gamma_PL
                * low_density_transition_function(rest_mass_density)
                * (rest_mass_density / transition_density) ** (gamma_PL - 1)
                + low_density_transition_function_derivative(rest_mass_density)
            ),
        )

    def get_pressure_density_derivative(
        self, rest_mass_density, temperature, electron_fraction
    ):
        cold_equilibrium = self.cold_eos_.chi_from_density(rest_mass_density)
        equilibrium_electron_fraction = self.get_equilibrium_charge_fraction(
            rest_mass_density=rest_mass_density
        )

        cold_composition_dependent = K_ * 4.0 / 3.0 * (
            np.cbrt(electron_fraction**4 * rest_mass_density)
            - np.cbrt(equilibrium_electron_fraction**4 * rest_mass_density)
        ) + self.get_symmetry_pressure_density_derivative(rest_mass_density) * (
            (1 - 2 * electron_fraction) ** 2
            - (1 - 2 * equilibrium_electron_fraction) ** 2
        )
        thermal_radiation = (
            16
            * fs_of_temperature(temperature)
            * stefan_boltzmann_sigma_
            * temperature**4
            / (9 * rest_mass_density)
        )
        thermal_ideal = 5 * temperature / 3
        # All of this to get the derivative of the degeneracy part of the
        # thermal pressure
        baryon_dirac_mass = self.get_dirac_specific_mass(rest_mass_density)
        derivative_a_thermal_baryon = self.get_derivative_of_a_thermal(
            rest_mass_density=rest_mass_density,
            electron_fraction=np.full_like(rest_mass_density, 0.5),
            dirac_specific_mass=baryon_dirac_mass,
        )

        derivative_a_thermal_electron = self.get_derivative_of_a_thermal(
            rest_mass_density=electron_fraction * rest_mass_density,
            electron_fraction=np.ones_like(electron_fraction),
            dirac_specific_mass=electron_mass_over_baryon_mass_,
            is_electron=True,
        )
        a_thermal_baryon = self.get_a_thermal(
            rest_mass_density=rest_mass_density,
            electron_fraction=np.full_like(rest_mass_density, 0.5),
            dirac_specific_mass=baryon_dirac_mass,
        )
        a_thermal_electron = self.get_a_thermal(
            rest_mass_density=electron_fraction * rest_mass_density,
            electron_fraction=np.ones_like(rest_mass_density),
            dirac_specific_mass=electron_mass_over_baryon_mass_,
        )
        # Construct the derivatives of the a_derivatives
        A_baryon = -self.alpha_ * (1 - baryon_dirac_mass**2)
        baryon_fermi_energy_common_factor = (
            3 * np.pi**2 * rest_mass_density
        ) ** (2 / 3) * hbar_over_baryon_mass_to_four_thirds_**2
        electron_fermi_energy_common_factor = (
            3 * np.pi**2 * electron_fraction * rest_mass_density
        ) ** (2 / 3) * hbar_over_baryon_mass_to_four_thirds_**2

        B_baryon = 1 / (
            1 + baryon_fermi_energy_common_factor / baryon_dirac_mass**2
        )
        C_baryon = baryon_fermi_energy_common_factor / baryon_dirac_mass**2

        A_electron = 0.0
        B_electron = 1 / (
            1
            + electron_fermi_energy_common_factor
            / electron_mass_over_baryon_mass_**2
        )
        C_electron = (
            electron_fermi_energy_common_factor
            / electron_mass_over_baryon_mass_**2
        )
        derivative_A_baryon = (
            2
            * self.alpha_
            / rest_mass_density
            * baryon_dirac_mass**2
            * A_baryon
        )
        second_derivative_a_thermal_baryon = (
            1.0
            * derivative_a_thermal_baryon
            * (
                derivative_a_thermal_baryon / a_thermal_baryon
                - 1 / rest_mass_density
            )
            + 2
            * a_thermal_baryon
            / (3 * rest_mass_density**2)
            * B_baryon
            * (
                3 * A_baryon**2
                - 1 / 3 * B_baryon * (3 * A_baryon + C_baryon) ** 2
                + 1 / 3 * C_baryon
                + 3 * rest_mass_density / 2 * derivative_A_baryon
            )
        )
        second_derivative_a_thermal_electron = derivative_a_thermal_electron * (
            derivative_a_thermal_electron / a_thermal_electron
            - 1 / rest_mass_density
        ) + 2 * a_thermal_electron / (
            9 * rest_mass_density**2
        ) * B_electron * C_electron * (
            1 - B_electron * C_electron
        )
        thermal_degeneracy_pressure = -(
            1
            * (
                derivative_a_thermal_baryon
                + electron_fraction**2 * derivative_a_thermal_electron
            )
            * rest_mass_density**2
            * temperature**2
        )
        thermal_degeneracy = -(
            second_derivative_a_thermal_baryon
            + electron_fraction * second_derivative_a_thermal_electron
        ) * rest_mass_density**2 * temperature**2 + (
            2
            * thermal_degeneracy_pressure
            * (
                1 / rest_mass_density
                - (
                    derivative_a_thermal_baryon
                    + derivative_a_thermal_electron * electron_fraction
                )
                / (a_thermal_baryon + a_thermal_electron * electron_fraction)
            )
        )
        return (
            cold_equilibrium
            + cold_composition_dependent
            + thermal_radiation
            + np.where(
                (a_thermal_electron * electron_fraction + a_thermal_baryon)
                * temperature**2
                > 1.5 * temperature,
                thermal_ideal,
                thermal_degeneracy,
            )
        )

    def get_temperature_dependent_pressure(
        self, rest_mass_density, temperature, electron_fraction
    ):
        fs = fs_of_temperature(temperature)
        electron_factor = electron_fraction
        # electron_factor=1
        radiation_part = 4 / 3 * stefan_boltzmann_sigma_ * fs * temperature**4
        ideal_part = rest_mass_density * temperature
        dirac_specific_mass = self.get_dirac_specific_mass(
            rest_mass_density=rest_mass_density
        )
        baryon_degeneracy_a = self.get_derivative_of_a_thermal(
            rest_mass_density=rest_mass_density,
            electron_fraction=np.full_like(rest_mass_density, 0.5),
            dirac_specific_mass=dirac_specific_mass,
        )
        electron_degeneracy_a = (
            self.get_derivative_of_a_thermal(
                rest_mass_density=electron_fraction * rest_mass_density,
                electron_fraction=np.ones_like(rest_mass_density),
                dirac_specific_mass=electron_mass_over_baryon_mass_,
                is_electron=True,
            )
            * electron_fraction
        )
        degeneracy_part = (
            1
            * (baryon_degeneracy_a + electron_factor * electron_degeneracy_a)
            * rest_mass_density**2
            * temperature**2
        )

        return radiation_part + 1 / (-1 / degeneracy_part + 1 / ideal_part)

    def pressure_from_density_and_temperature(
        self, rest_mass_density, temperature, electron_fraction
    ):
        return (
            self.cold_eos_.pressure_from_density(rest_mass_density)
            + self.get_composition_dependent_pressure(
                rest_mass_density=rest_mass_density,
                electron_fraction=electron_fraction,
            )
            + self.get_temperature_dependent_pressure(
                rest_mass_density=rest_mass_density,
                temperature=temperature,
                electron_fraction=electron_fraction,
            )
        )

    def temperature_from_density_and_energy(
        self, rest_mass_density, specific_internal_energy, electron_fraction
    ):
        thermal_energy = (
            specific_internal_energy
            - self.cold_eos_.specific_internal_energy_from_density(
                rest_mass_density=rest_mass_density
            )
            - self.get_composition_dependent_energy(
                rest_mass_density=rest_mass_density,
                electron_fraction=electron_fraction,
            )
        )
        thermal_energy_from_temperature_difference = (
            lambda temperature: self.get_temperature_dependent_energy(
                rest_mass_density=rest_mass_density,
                temperature=temperature,
                electron_fraction=electron_fraction,
            )
            - specific_internal_energy
        )
        temp_from_eps = root(
            thermal_energy_from_temperature_difference,
            x0=2 / 3 * specific_internal_energy,
        ).x
        return (
            temp_from_eps if rest_mass_density.shape != () else temp_from_eps[0]
        )

    def sound_speed_squared_from_density_and_temperature(
        self, rest_mass_density, temperature, electron_fraction
    ):
        internal_energy = (
            self.specific_internal_energy_from_density_and_temperature(
                rest_mass_density=rest_mass_density,
                temperature=temperature,
                electron_fraction=electron_fraction,
            )
        )
        pressure = self.pressure_from_density_and_temperature(
            rest_mass_density=rest_mass_density,
            temperature=temperature,
            electron_fraction=electron_fraction,
        )
        derivative_pressure_density = self.get_pressure_density_derivative(
            rest_mass_density=rest_mass_density,
            temperature=temperature,
            electron_fraction=electron_fraction,
        )
        return derivative_pressure_density / (
            1 + internal_energy + pressure / rest_mass_density
        )
        # return np.ones_like(rest_mass_density)


# Needed for python testing interface
class AnalyticalThermalPolytrope(AnalyticalThermal):
    def __init__(
        self, polytropic_constant, polytropic_exponent, S0, L, gamma, n0, alpha
    ):
        super().__init__(
            S0,
            L,
            gamma,
            n0,
            alpha,
            PolytropicFluid(
                polytropic_constant=polytropic_constant,
                polytropic_exponent=polytropic_exponent,
            ),
        )


def analytical_thermal_polytrope_pressure_from_density_and_temperature(
    rest_mass_density, temperature, electron_fraction, *args
):
    return AnalyticalThermalPolytrope(
        *args
    ).pressure_from_density_and_temperature(
        rest_mass_density=rest_mass_density,
        temperature=temperature,
        electron_fraction=electron_fraction,
    )


def analytical_thermal_polytrope_energy_from_density_and_temperature(
    rest_mass_density, temperature, electron_fraction, *args
):
    return AnalyticalThermalPolytrope(
        *args
    ).specific_internal_energy_from_density_and_temperature(
        rest_mass_density=rest_mass_density,
        temperature=temperature,
        electron_fraction=electron_fraction,
    )


def analytical_thermal_polytrope_temperature_from_density_and_energy(
    rest_mass_density, specific_internal_energy, electron_fraction, *args
):
    return AnalyticalThermalPolytrope(
        *args
    ).temperature_from_density_and_energy(
        rest_mass_density=rest_mass_density,
        specific_internal_energy=specific_internal_energy,
        electron_fraction=electron_fraction,
    )


def analytical_thermal_polytrope_sound_speed_squared_from_density_and_temperature(
    rest_mass_density, temperature, electron_fraction, *args
):
    return AnalyticalThermalPolytrope(
        *args
    ).sound_speed_squared_from_density_and_temperature(
        rest_mass_density=rest_mass_density,
        temperature=temperature,
        electron_fraction=electron_fraction,
    )


# if __name__ == "__main__":
if False:
    eos = AnalyticalThermal(
        37.39 / 938,
        118.49 / 938,
        0.62,
        0.00045 * 0.12 / 0.16,
        0.80,
        PolytropicFluid(100, 2.0),
    )
    rhonuc = saturation_density_
    N = 120
    rhos = np.geomspace(1e-12 * rhonuc, 30 * rhonuc, N)
    rhos_fixed = np.full(N, 6 * rhonuc)
    Yps_fixed = np.full_like(rhos_fixed, 0.5)
    temperature = np.linspace(1e-5, 6e-1, N)

    def compare_thermal_pressure():
        for alpha in [0.2, 0.8, 1.2]:
            local_eos = AnalyticalThermal(
                46.39 / 938,
                100.49 / 938,
                1.02,
                0.00045 * 0.12 / 0.16,
                alpha,
                PolytropicFluid(100, 2.0),
            )
            temperature_fixed = 1 / baryon_mass_in_mev_ * np.full_like(rhos, 10)
            fs = fs_of_temperature(temperature_fixed)
            radiation_part = (
                4 / 3 * stefan_boltzmann_sigma_ * fs * temperature_fixed**4
            )
            ideal_part = rhos * temperature_fixed
            plt.loglog(
                rhos * 0.16 / 0.00045,
                938.5
                * 0.16
                / 0.00045
                * local_eos.get_temperature_dependent_pressure(
                    rhos, np.full_like(rhos, temperature_fixed), Yps_fixed
                ),
                color=cmap(alpha / 1.2),
            )
        ax = plt.gca()
        plt.xlim(2e-3, 7)
        plt.ylim(1e-2, 1.5)
        plt.tick_params(
            labeltop=True,
            labelright=True,
            bottom=True,
            top=True,
            left=True,
            right=True,
        )
        plt.grid(which="both")
        plt.ylabel(r"thermal_pressure $[\rm{MeV}/ \rm{fm}^3]$")
        plt.xlabel(r"rest mass density $[\rm{fm}^{-3}]$")
        plt.savefig("rest_mass_density_vs_thermal_pressure_by_alpha.pdf")
        plt.clf()
        alpha_sfho = 0.6
        sfho_eos = AnalyticalThermal(
            31.57 / baryon_mass_in_mev_,
            47.10 / baryon_mass_in_mev_,
            0.41,
            saturation_density_ * 0.08 / 0.16,
            alpha=alpha_sfho,
            cold_eos=PolytropicFluid(100, 2.0),
        )

        sfho_rho = saturation_density_ * np.array([6.25e-6, 6.25e-3, 1.0, 2.5])
        sfho_temp = np.full_like(sfho_rho, 10 / baryon_mass_in_mev_)
        sfho_yp = np.full_like(sfho_rho, 0.25)
        print("sfho_temp^4", sfho_temp**4)
        print("sfho_fs_of_temp", fs_of_temperature(sfho_temp))

        print(
            "sfho_eos pressure",
            sfho_eos.get_temperature_dependent_pressure(
                sfho_rho, sfho_temp, sfho_yp
            )
            / saturation_density_
            * 0.16
            * baryon_mass_in_mev_,
        )

        print(
            "sfho_eos energy",
            sfho_eos.get_temperature_dependent_energy(
                sfho_rho, sfho_temp, sfho_yp
            )
            * baryon_mass_in_mev_,
        )
        plt.plot(
            sfho_rho,
            sfho_eos.get_temperature_dependent_pressure(
                sfho_rho, sfho_temp, sfho_yp
            )
            / saturation_density_
            * 0.16
            * baryon_mass_in_mev_,
            color="deepskyblue",
            label="my code (w/correction)",
        )
        plt.plot(
            sfho_rho,
            [7.949760e-04, 1.007952e-02, 4.587350e-01, 5.289027e-01],
            color="darkorange",
            linestyle="--",
            label="tabulated ('official'`)",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"rest mass density $(1/M_{\odot}^2)")
        plt.ylabel(r"pressure $(\rm{MeV}/\rm{fm}^3)$")
        plt.legend()
        plt.savefig(
            "rest_mass_density_vs_pressure_with_correction.pdf",
            bbox_inches="tight",
        )
        plt.clf()
        plt.plot(
            sfho_rho,
            (
                sfho_eos.get_temperature_dependent_energy(
                    sfho_rho, sfho_temp, sfho_yp
                )
                * baryon_mass_in_mev_
                - np.array([2.369940e03, 1.630152e01, 3.593102e00, 2.015691e00])
            )
            / np.array([2.369940e03, 1.630152e01, 3.593102e00, 2.015691e00]),
            color="darkorange",
            linestyle="--",
            label="error (my code - 'official')/official",
        )
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel(r"rest mass density $(1/M_{\odot}^2)")
        plt.ylabel(r"$\Delta e / e$")
        plt.legend()
        plt.savefig(
            "rest_mass_density_vs_energy_other_correction.pdf",
            bbox_inches="tight",
        )


def compare_composition_dependent_energy():
    alpha_sfho = 0.6
    sfho_eos = AnalyticalThermal(
        31.57 / baryon_mass_in_mev_,
        47.10 / baryon_mass_in_mev_,
        gamma=0.41,
        n0=saturation_density_ * 0.08 / 0.16,
        alpha=alpha_sfho,
        cold_eos=PolytropicFluid(100.0, 2.0),
    )
    sfho_eos_2 = AnalyticalThermalPolytrope(
        100.0,
        2.0,
        31.57 / baryon_mass_in_mev_,
        47.10 / baryon_mass_in_mev_,
        gamma=0.41,
        n0=saturation_density_ * 0.08 / 0.16,
        alpha=alpha_sfho,
    )

    sfho_rho = saturation_density_ * np.array([6.25e-6, 6.25e-3, 1.0, 2.5])
    # sfho_temp = np.full_like(sfho_rho, 10/baryon_mass_in_mev_)
    sfho_yp = np.full_like(sfho_rho, 0.25)
    test_rho = np.linspace(
        0.50 * saturation_density_, 0.55 * saturation_density_, 50
    )
    print(
        "sfho_eos composition dependent energy",
        sfho_eos.get_composition_dependent_energy(sfho_rho, sfho_yp)
        * baryon_mass_in_mev_,
    )
    print(
        "sfho_eos temperature_dependent_energ?",
        sfho_eos_2.get_composition_dependent_energy(sfho_rho, sfho_yp)
        * baryon_mass_in_mev_,
    )

    print(
        "sfho_eos composition dependent pressure",
        sfho_eos.get_composition_dependent_pressure(sfho_rho, sfho_yp)
        * 0.16
        / saturation_density_
        * baryon_mass_in_mev_,
    )


# compare_composition_dependent_energy()


def compare_thermal_energy():
    alpha_sfho = 0.6
    sfho_eos = AnalyticalThermal(
        31.57 / baryon_mass_in_mev_,
        47.10 / baryon_mass_in_mev_,
        gamma=0.41,
        n0=saturation_density_ * 0.08 / 0.16,
        alpha=alpha_sfho,
        cold_eos=PolytropicFluid(100, 2.0),
    )
    sfho_eos_2 = AnalyticalThermalPolytrope(
        100.0,
        2.0,
        31.57 / baryon_mass_in_mev_,
        47.10 / baryon_mass_in_mev_,
        gamma=0.41,
        n0=saturation_density_ * 0.08 / 0.16,
        alpha=alpha_sfho,
    )

    sfho_rho = saturation_density_ * np.array([6.25e-6, 6.25e-3, 1.0, 2.5])
    sfho_temp = np.full_like(sfho_rho, 10 / baryon_mass_in_mev_)
    sfho_yp = np.full_like(sfho_rho, 0.25)
    test_rho = np.linspace(
        0.50 * saturation_density_, 0.55 * saturation_density_, 50
    )
    print(
        "sfho_eos temperature dependent energy",
        sfho_eos.get_temperature_dependent_energy(sfho_rho, sfho_temp, sfho_yp)
        * baryon_mass_in_mev_,
    )

    print(
        "sfho_eos temperature dependent pressure",
        sfho_eos.get_temperature_dependent_pressure(
            sfho_rho, sfho_temp, sfho_yp
        )
        * 0.16
        / saturation_density_
        * baryon_mass_in_mev_,
    )


# compare_thermal_energy()


def compare_sound_speed_squared():
    alpha_sfho = 0.6
    sfho_eos = AnalyticalThermal(
        31.57 / baryon_mass_in_mev_,
        47.10 / baryon_mass_in_mev_,
        gamma=0.41,
        n0=saturation_density_ * 0.08 / 0.16,
        alpha=alpha_sfho,
        cold_eos=PolytropicFluid(100, 2.0),
    )
    sfho_eos_2 = AnalyticalThermalPolytrope(
        100.0,
        2.0,
        31.57 / baryon_mass_in_mev_,
        47.10 / baryon_mass_in_mev_,
        gamma=0.41,
        n0=saturation_density_ * 0.08 / 0.16,
        alpha=alpha_sfho,
    )

    sfho_rho = saturation_density_ * np.array([6.25e-6, 6.25e-3, 1.0, 2.5])
    sfho_temp = np.full_like(sfho_rho, 0 / baryon_mass_in_mev_)
    sfho_yp = np.full_like(sfho_rho, 0.05)
    test_rho = np.linspace(
        0.50 * saturation_density_, 0.55 * saturation_density_, 50
    )
    print(
        "sfho_eos sound_speed_squared",
        sfho_eos.sound_speed_squared_from_density_and_temperature(
            sfho_rho, sfho_temp, sfho_yp
        ),
    )


compare_sound_speed_squared()
