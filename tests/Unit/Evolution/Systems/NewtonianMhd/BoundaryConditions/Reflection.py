# Distributed under the MIT License.
# See LICENSE.txt for details.

import Evolution.Systems.NewtonianMhd.BoundaryConditions.ReflectionImpl as impl
import numpy as np

# Whether the tangential velocity is also reversed. ConductorReflection.py
# imports this module with NO_SLIP patched to True.
NO_SLIP = False


def _ghost(
    face_mesh_velocity,
    normal_covector,
    int_magnetic_field,
    int_divergence_cleaning_field,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
    int_background_magnetic_field,
    divergence_cleaning_speed,
    no_slip,
):
    velocity = impl.ghost_velocity(
        face_mesh_velocity, normal_covector, int_velocity, no_slip
    )
    magnetic_field = impl.ghost_magnetic_field(
        normal_covector, int_magnetic_field
    )
    divergence_cleaning_field = -int_divergence_cleaning_field
    momentum_density = int_mass_density * velocity
    energy_density = impl.energy_density(
        int_mass_density,
        velocity,
        int_specific_internal_energy,
        magnetic_field,
    )
    return {
        "mass_density_cons": int_mass_density,
        "momentum_density": momentum_density,
        "energy_density": energy_density,
        "magnetic_field_cons": magnetic_field,
        "divergence_cleaning_field_cons": divergence_cleaning_field,
        "flux_mass_density": momentum_density,
        "flux_momentum_density": impl.momentum_density_flux(
            momentum_density,
            velocity,
            int_pressure,
            magnetic_field,
            int_background_magnetic_field,
        ),
        "flux_energy_density": impl.energy_density_flux(
            energy_density,
            velocity,
            int_pressure,
            magnetic_field,
            divergence_cleaning_field,
            int_background_magnetic_field,
        ),
        "flux_magnetic_field": impl.magnetic_field_flux(
            velocity,
            divergence_cleaning_field,
            magnetic_field,
            int_background_magnetic_field,
        ),
        "flux_divergence_cleaning_field": (
            divergence_cleaning_speed**2 * magnetic_field
        ),
        "background_magnetic_field": int_background_magnetic_field,
        "velocity": velocity,
        "specific_internal_energy": int_specific_internal_energy,
    }


def error(*args):
    return None


def mass_density_cons(*args):
    return _ghost(*args, NO_SLIP)["mass_density_cons"]


def momentum_density(*args):
    return _ghost(*args, NO_SLIP)["momentum_density"]


def energy_density(*args):
    return _ghost(*args, NO_SLIP)["energy_density"]


def magnetic_field_cons(*args):
    return _ghost(*args, NO_SLIP)["magnetic_field_cons"]


def divergence_cleaning_field_cons(*args):
    return _ghost(*args, NO_SLIP)["divergence_cleaning_field_cons"]


def flux_mass_density(*args):
    return _ghost(*args, NO_SLIP)["flux_mass_density"]


def flux_momentum_density(*args):
    return _ghost(*args, NO_SLIP)["flux_momentum_density"]


def flux_energy_density(*args):
    return _ghost(*args, NO_SLIP)["flux_energy_density"]


def flux_magnetic_field(*args):
    return _ghost(*args, NO_SLIP)["flux_magnetic_field"]


def flux_divergence_cleaning_field(*args):
    return _ghost(*args, NO_SLIP)["flux_divergence_cleaning_field"]


def background_magnetic_field(*args):
    return _ghost(*args, NO_SLIP)["background_magnetic_field"]


def velocity(*args):
    return _ghost(*args, NO_SLIP)["velocity"]


def specific_internal_energy(*args):
    return _ghost(*args, NO_SLIP)["specific_internal_energy"]
