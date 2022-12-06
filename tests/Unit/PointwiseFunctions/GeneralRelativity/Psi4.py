# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import math
import cmath

from .ProjectionOperators import transverse_projection_operator
from .WeylPropagating import weyl_propagating_modes


def psi_4(spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
          spatial_metric, inv_spatial_metric, inertial_coords):
    magnitude_inertial = math.sqrt(
        np.einsum("a,b,ab", inertial_coords, inertial_coords, spatial_metric))
    if (magnitude_inertial != 0.0):
        r_hat = np.einsum("a", inertial_coords / magnitude_inertial)
    else:
        r_hat = np.einsum("a", inertial_coords * 0.0)

    lower_r_hat = np.einsum("a,ab", r_hat, spatial_metric)

    inv_projection_tensor = transverse_projection_operator(
        inv_spatial_metric, r_hat)
    projection_tensor = transverse_projection_operator(spatial_metric,
                                                       lower_r_hat)
    projection_up_lo = np.einsum("ab,ac", inv_projection_tensor,
                                 spatial_metric)

    u8_plus = weyl_propagating_modes(spatial_ricci, extrinsic_curvature,
                                     inv_spatial_metric,
                                     cov_deriv_extrinsic_curvature, r_hat,
                                     inv_projection_tensor, projection_tensor,
                                     projection_up_lo, 1)

    x_coord = np.zeros((3))
    x_coord[0] = 1
    magnitude_x = math.sqrt(
        np.einsum("a,b,ab", x_coord, x_coord, spatial_metric))
    x_hat = x_coord / magnitude_x
    y_coord = np.zeros((3))
    y_coord[1] = 1
    magnitude_y = math.sqrt(
        np.einsum("a,b,ab", y_coord, y_coord, spatial_metric))
    y_hat = y_coord / magnitude_y

    m_bar = x_hat - (y_hat * complex(0.0, 1.0))

    return (-0.5 * np.einsum("ab,a,b", u8_plus, m_bar, m_bar))