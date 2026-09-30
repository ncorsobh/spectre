# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def ghost_velocity(face_mesh_velocity, normal_covector, int_velocity, no_slip):
    if no_slip:
        velocity = -int_velocity
    else:
        velocity_dot_normal = np.einsum("i,i", normal_covector, int_velocity)
        velocity = int_velocity - 2.0 * velocity_dot_normal * normal_covector

    if face_mesh_velocity is None:
        return velocity
    if no_slip:
        return velocity + 2.0 * face_mesh_velocity
    mesh_velocity_dot_normal = np.einsum(
        "i,i", normal_covector, face_mesh_velocity
    )
    return velocity + 2.0 * mesh_velocity_dot_normal * normal_covector


def ghost_magnetic_field(normal_covector, int_magnetic_field):
    magnetic_field_dot_normal = np.einsum(
        "i,i", normal_covector, int_magnetic_field
    )
    return (
        int_magnetic_field - 2.0 * magnetic_field_dot_normal * normal_covector
    )


def energy_density(
    mass_density, velocity, specific_internal_energy, magnetic_field
):
    return mass_density * (
        0.5 * np.dot(velocity, velocity) + specific_internal_energy
    ) + 0.5 * np.dot(magnetic_field, magnetic_field)


def magnetic_pressure(magnetic_field, background_magnetic_field):
    return 0.5 * np.dot(magnetic_field, magnetic_field) + np.dot(
        background_magnetic_field, magnetic_field
    )


def momentum_density_flux(
    momentum_density,
    velocity,
    pressure,
    magnetic_field,
    background_magnetic_field,
):
    # flux[j][i] = F^j(rho v^i); this one is symmetric.
    total_field = background_magnetic_field + magnetic_field
    flux = (
        np.outer(velocity, momentum_density)
        - np.outer(total_field, magnetic_field)
        - np.outer(magnetic_field, background_magnetic_field)
    )
    return flux + np.identity(len(velocity)) * (
        pressure + magnetic_pressure(magnetic_field, background_magnetic_field)
    )


def energy_density_flux(
    energy_density_value,
    velocity,
    pressure,
    magnetic_field,
    divergence_cleaning_field,
    background_magnetic_field,
):
    total_field = background_magnetic_field + magnetic_field
    return (
        (
            energy_density_value
            + pressure
            + magnetic_pressure(magnetic_field, background_magnetic_field)
        )
        * velocity
        - total_field * np.dot(velocity, magnetic_field)
        - background_magnetic_field * divergence_cleaning_field
    )


def magnetic_field_flux(
    velocity,
    divergence_cleaning_field,
    magnetic_field,
    background_magnetic_field,
):
    # flux[j][i] = F^j(B^i) = v^j B^i - B^j v^i + delta^{ji} psi
    total_field = background_magnetic_field + magnetic_field
    return (
        np.outer(velocity, total_field)
        - np.outer(total_field, velocity)
        + np.identity(len(velocity)) * divergence_cleaning_field
    )
