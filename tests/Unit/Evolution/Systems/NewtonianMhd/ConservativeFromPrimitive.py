# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density_cons(mass_density, velocity, specific_internal_energy):
    return mass_density


def momentum_density(mass_density, velocity, specific_internal_energy):
    return mass_density * velocity


def energy_density(mass_density, velocity, specific_internal_energy):
    return (
        0.5 * mass_density * np.dot(velocity, velocity)
        + mass_density * specific_internal_energy
    )


def mhd_mass_density_cons(
    mass_density,
    velocity,
    specific_internal_energy,
    magnetic_field,
    divergence_cleaning_field,
):
    return mass_density


def mhd_momentum_density(
    mass_density,
    velocity,
    specific_internal_energy,
    magnetic_field,
    divergence_cleaning_field,
):
    return mass_density * velocity


def mhd_energy_density(
    mass_density,
    velocity,
    specific_internal_energy,
    magnetic_field,
    divergence_cleaning_field,
):
    return (
        0.5 * mass_density * np.dot(velocity, velocity)
        + mass_density * specific_internal_energy
        + 0.5 * np.dot(magnetic_field, magnetic_field)
    )


def mhd_magnetic_field_cons(
    mass_density,
    velocity,
    specific_internal_energy,
    magnetic_field,
    divergence_cleaning_field,
):
    return magnetic_field


def mhd_divergence_cleaning_field_cons(
    mass_density,
    velocity,
    specific_internal_energy,
    magnetic_field,
    divergence_cleaning_field,
):
    return divergence_cleaning_field
