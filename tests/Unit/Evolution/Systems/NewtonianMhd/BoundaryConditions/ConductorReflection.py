# Distributed under the MIT License.
# See LICENSE.txt for details.

import Evolution.Systems.NewtonianMhd.BoundaryConditions.Reflection as reflection

_ghost = reflection._ghost
NO_SLIP = True


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
