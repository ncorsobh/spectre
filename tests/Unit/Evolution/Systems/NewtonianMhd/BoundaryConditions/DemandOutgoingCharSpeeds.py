# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(
    face_mesh_velocity,
    normal_covector,
    int_magnetic_field,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_background_magnetic_field,
    use_polytropic_eos,
):
    adiabatic_index = 1.3
    chi = int_specific_internal_energy * (adiabatic_index - 1.0)
    kappa_times_p_over_rho_squared = (
        adiabatic_index - 1.0
    ) ** 2 * int_specific_internal_energy
    sound_speed_squared = chi + kappa_times_p_over_rho_squared

    total_field = int_background_magnetic_field + int_magnetic_field
    fast_speed = np.sqrt(
        sound_speed_squared
        + np.dot(total_field, total_field) / int_mass_density
    )

    normal_dot_velocity = np.einsum("i,i", int_velocity, normal_covector)

    if face_mesh_velocity is None:
        min_char_speed = normal_dot_velocity - fast_speed
    else:
        normal_dot_mesh_velocity = np.einsum(
            "i,i", face_mesh_velocity, normal_covector
        )
        min_char_speed = (
            normal_dot_velocity - normal_dot_mesh_velocity - fast_speed
        )

    if min_char_speed < 0.0:
        return "DemandOutgoingCharSpeeds boundary condition violated"

    return None
