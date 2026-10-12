// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <cstddef>
#include <cstdint>

/// \cond
class DataVector;
template <size_t Dim>
class Index;
class Matrix;
template <size_t Dim>
class Mesh;
enum class Side : uint8_t;
namespace Spectral {
enum class Parity : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace evolution::dg::subcell::fd {
/*!
 * \ingroup DgSubcellGroup
 * \brief Computes the projection matrix in 1 dimension going from a DG
 * mesh to a conservative finite difference subcell mesh.
 *
 * The parity parameter is required for ZernikeB1 bases, and is ignored for
 * others.
 */
const Matrix& projection_matrix(const Mesh<1>& dg_mesh, size_t subcell_extents,
                                const Spectral::Quadrature& subcell_quadrature,
                                Spectral::Parity parity);

/*!
 * \ingroup DgSubcellGroup
 * \brief Computes the matrix needed for reconstructing the DG solution from
 * the subcell solution.
 *
 * Reconstructing the DG solution from the FD solution is a bit more
 * involved than projecting the DG solution to the FD subcells. Denoting the
 * projection operator by \f$\mathcal{P}\f$ and the reconstruction operator by
 * \f$\mathcal{R}\f$, we desire the property
 *
 * \f{align*}{
 *   \mathcal{R}(\mathcal{P}(u_{\breve{\imath}}
 *   J_{\breve{\imath}}))=u_{\breve{\imath}} J_{\breve{\imath}},
 * \f}
 *
 * where \f$\breve{\imath}\f$ denotes a grid point on the DG grid, \f$u\f$ is
 * the solution on the DG grid, and \f$J\f$ is the determinant of the Jacobian
 * on the DG grid. We also require that the integral of the conserved variables
 * over the subcells is equal to the integral over the DG element. That is,
 *
 * \f{align*}{
 *   \int_{\Omega}u \,d^3x =\int_{\Omega} \underline{u} \,d^3x \Longrightarrow
 *   \int_{\Omega}u J \,d^3\xi=\int_{\Omega} \underline{u} J \,d^3\xi,
 * \f}
 *
 * where \f$\underline{u}\f$ is the solution on the subcells. Because the number
 * of subcell points is larger than the number of DG points, we need to solve a
 * constrained linear least squares problem to reconstruct the DG solution from
 * the subcells.
 *
 * The final reconstruction matrix is given by
 *
 * \f{align*}{
 *   R_{\breve{\jmath}\underline{i}}
 *   &=\left\{(2 \mathcal{P}\otimes\mathcal{P})^{-1}2\mathcal{P} - (2
 *   \mathcal{P}\otimes\mathcal{P})^{-1}\vec{w}\left[\mathbf{w}(2
 *   \mathcal{P}\otimes\mathcal{P})^{-1}\vec{w}\right]^{-1}\mathbf{w}(2
 *   \mathcal{P}\otimes\mathcal{P})^{-1}2\mathcal{P}
 *   + (2 \mathcal{P}\otimes\mathcal{P})^{-1}\vec{w}\left[\mathbf{w}(2
 *   \mathcal{P}\otimes\mathcal{P})^{-1}\vec{w}\right]^{-1}\vec{\underline{w}}
 *   \right\}_{\breve{\jmath}\underline{i}},
 * \f}
 *
 * where \f$\vec{w}\f$ is the vector of integration weights on the DG element,
 * \f$\mathbf{w}=w_{\breve{l}}\delta_{\breve{l}\breve{\jmath}}\f$, and
 * \f$\vec{\underline{w}}\f$ is the vector of integration weights over the
 * subcells. The integration weights \f$\vec{\underline{w}}\f$ on the subcells
 * are those for 6th-order integration on a uniform mesh, which `cell_integral`
 * applies.
 *
 * Writing \f$A=2\mathcal{P}^{\mathsf{T}}\mathcal{P}\f$, the matrix is
 * assembled from one-dimensional factors rather than formed at full size.
 * Interpolation acts one direction at a time, so
 * \f$\mathcal{P}=\bigotimes_d\mathcal{P}_d\f$ (here \f$\bigotimes\f$ is a
 * genuine tensor product, unlike the \f$\otimes\f$ written above). Setting
 * \f$A_d=\mathcal{P}_d^{\mathsf{T}}\mathcal{P}_d\f$, so that
 * \f$A=2\bigotimes_d A_d\f$, and
 *
 * \f{align*}{
 *   B_d = A_d^{-1}\mathcal{P}_d^{\mathsf{T}}, \quad
 *   \vec{y}_d = A_d^{-1}\vec{w}_d, \quad
 *   s_d = \vec{w}_d^{\mathsf{T}}\vec{y}_d, \quad
 *   \vec{\underline{z}}_d = \mathcal{P}_d\vec{y}_d,
 * \f}
 *
 * each piece above factors as
 *
 * \f{align*}{
 *   A^{-1}2\mathcal{P}^{\mathsf{T}} &= \bigotimes_d B_d, \\
 *   A^{-1}\vec{w}\left[\vec{w}^{\mathsf{T}}A^{-1}\vec{w}\right]^{-1}
 *     &= \frac{\bigotimes_d \vec{y}_d}{\prod_d s_d}, \\
 *   \vec{w}^{\mathsf{T}}A^{-1}2\mathcal{P}^{\mathsf{T}}
 *     &= \bigotimes_d \vec{\underline{z}}_d^{\mathsf{T}},
 * \f}
 *
 * so that
 *
 * \f{align*}{
 *   R = \bigotimes_d B_d + \frac{\bigotimes_d \vec{y}_d}{\prod_d s_d}
 *   \left(\bigotimes_d \vec{\underline{w}}_d
 *       - \bigotimes_d \vec{\underline{z}}_d\right)^{\mathsf{T}}.
 * \f}
 *
 * The constraint contributes a single rank-one correction, whose subcell
 * factor is the amount by which the unconstrained least-squares fit misses the
 * required cell integral. The factor of two in \f$A\f$ appears separately in
 * \f$A^{-1}\vec{w}=\frac{1}{2}\bigotimes_d\vec{y}_d\f$ and
 * \f$\vec{w}^{\mathsf{T}}A^{-1}\vec{w}=\frac{1}{2}\prod_d s_d\f$ and cancels in
 * their ratio, which is why none survives in the expressions above. Only
 * \f$D\f$ inversions of size \f$n_d\f$ are needed rather than one of size
 * \f$\prod_d n_d\f$.
 *
 * \note \p subcell_extents must be \f$2n-1\f$ of the DG extents in every
 * dimension, as produced by `fd::mesh`. The matrices are cached on the DG
 * extents alone and the subcell extents are derived from them, so this is a
 * precondition rather than a free parameter; it is checked by an `ASSERT`.
 */
template <size_t Dim>
const Matrix& reconstruction_matrix(const Mesh<Dim>& dg_mesh,
                                    const Index<Dim>& subcell_extents);

/*!
 * \ingroup DgSubcellGroup
 * \brief The definite integral of \p integrand over the cell, using the
 * subcell integration rule that `reconstruction_matrix` conserves.
 *
 * This is the 6th-order rule on a uniform mesh, i.e. the weights
 * \f$\vec{\underline{w}}\f$ of `reconstruction_matrix`. It is **not** the rule
 * that `definite_integral` applies to a cell-centered finite-difference mesh:
 * that one is the midpoint rule, and the two disagree. Reconstruction preserves
 * the integral in the sense
 *
 * \f{align*}{
 *   \int_{\Omega} \mathcal{R}(\underline{u}) \,d^D\xi
 *   = \texttt{cell\_integral}(\underline{u}),
 * \f}
 *
 * with the left-hand side the ordinary DG quadrature of the reconstructed
 * solution.
 *
 * \p subcell_mesh must be a cell-centered finite-difference mesh on
 * \f$[-1,1]^D\f$.
 */
template <size_t Dim>
double cell_integral(const DataVector& integrand,
                     const Mesh<Dim>& subcell_mesh);

/*!
 * \ingroup DgSubcellGroup
 * \brief Computes the 1D reconstruction matrix for a ZernikeB1 DG mesh.
 *
 * Uses the same constrained least-squares formula as the Legendre overload,
 * but with the parity-aware ZernikeB1 interpolation matrix and GaussRadauUpper
 * quadrature weights, notably only for even parity. For odd parity, the
 * collocation points are inconsistent with the constraint equation, so we opt
 * to directly use the pseduo-inverse. Only `DimByDim` reconstruction is
 * supported for ZernikeB1  meshes; this overload is called per-dimension for
 * the radial direction.
 *
 * The `parity` parameter must be `Even` or `Odd` (not `Uninitialized`).
 */
const Matrix& reconstruction_matrix(const Mesh<1>& dg_mesh,
                                    size_t subcell_extents,
                                    Spectral::Parity parity);

/*!
 * \ingroup DgSubcellGroup
 * \brief Computes the projection matrix in 1 dimension going from a DG
 * mesh to a conservative finite difference subcell mesh for only the ghost
 * zones.
 *
 * This is used when a neighbor sends DG volume data and we need to switch to
 * FD. In this case we need to project the DG volume data onto the ghost zone
 * cells.
 *
 * The parity parameter is required for ZernikeB1 bases, and is ignored for
 * others.
 *
 * \note Currently assumes a max ghost zone size of `5` and a minimum ghost zone
 * size of 2.
 */
const Matrix& projection_matrix(const Mesh<1>& dg_mesh, size_t subcell_extents,
                                size_t ghost_zone_size, Side side,
                                Spectral::Parity parity);
}  // namespace evolution::dg::subcell::fd
