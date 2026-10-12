// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Matrices.hpp"

#include <array>
#include <cstddef>
#include <ostream>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/RuntimeCache.hpp"
#include "Utilities/StaticCache.hpp"

namespace evolution::dg::subcell::fd {
const Matrix& projection_matrix(const Mesh<1>& dg_mesh,
                                const size_t subcell_extents,
                                const Spectral::Quadrature& subcell_quadrature,
                                const Spectral::Parity parity) {
  ASSERT(dg_mesh.basis(0) == Spectral::Basis::Legendre or
             dg_mesh.basis(0) == Spectral::Basis::Cartoon or
             dg_mesh.basis(0) == Spectral::Basis::ZernikeB1,
         "FD Subcell projection only supports Legendre, Cartoon, or ZernikeB1 "
         "bases right "
         "now but got basis "
             << dg_mesh.basis(0));
  ASSERT(subcell_quadrature == Spectral::Quadrature::FaceCentered or
             subcell_quadrature == Spectral::Quadrature::CellCentered or
             subcell_quadrature == Spectral::Quadrature::AxialSymmetry or
             subcell_quadrature == Spectral::Quadrature::SphericalSymmetry,
         "subcell_quadrature option in projection_matrix should be "
         "FaceCentered, CellCentered, or a Cartoon quadrature, but got "
             << subcell_quadrature);
  ASSERT(dg_mesh.basis(0) != Spectral::Basis::ZernikeB1 or
             parity != Spectral::Parity::Uninitialized,
         "Parity must be set when using ZernikeB1");

  const static auto zernike_b1_cache = make_static_cache<
      CacheRange<Spectral::minimum_number_of_points<
                     Spectral::Basis::ZernikeB1,
                     Spectral::Quadrature::GaussRadauUpper>,
                 Spectral::maximum_number_of_points<
                     Spectral::Basis::ZernikeB1,
                     Spectral::Quadrature::GaussRadauUpper> +
                     1>,
      CacheRange<Spectral::minimum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered>,
                 Spectral::maximum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered> +
                     1>,
      CacheEnumeration<Spectral::Quadrature, Spectral::Quadrature::CellCentered,
                       Spectral::Quadrature::FaceCentered>,
      CacheEnumeration<Spectral::Parity, Spectral::Parity::Even,
                       Spectral::Parity::Odd>>(
      [](const size_t local_num_dg_points, const size_t local_num_fd_points,
         const Spectral::Quadrature local_subcell_quadrature,
         const Spectral::Parity local_parity) {
        return Spectral::interpolation_matrix<
            Spectral::Basis::ZernikeB1, Spectral::Quadrature::GaussRadauUpper>(
            local_num_dg_points,
            Spectral::collocation_points(
                Mesh<1>{local_num_fd_points, Spectral::Basis::FiniteDifference,
                        local_subcell_quadrature}),
            local_parity);
      });

  const static auto legendre_cache = make_static_cache<
      CacheRange<
          Spectral::minimum_number_of_points<
              Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>,
          Spectral::maximum_number_of_points<
              Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto> +
              1>,
      CacheRange<Spectral::minimum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered>,
                 Spectral::maximum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered> +
                     1>,
      CacheEnumeration<Spectral::Quadrature, Spectral::Quadrature::Gauss,
                       Spectral::Quadrature::GaussLobatto>,
      CacheEnumeration<Spectral::Quadrature, Spectral::Quadrature::CellCentered,
                       Spectral::Quadrature::FaceCentered>>(
      [](const size_t local_num_dg_points, const size_t local_num_fd_points,
         const Spectral::Quadrature dg_quadrature,
         const Spectral::Quadrature local_subcell_quadrature) {
        return Spectral::interpolation_matrix(
            Mesh<1>{local_num_dg_points, Spectral::Basis::Legendre,
                    dg_quadrature},
            Spectral::collocation_points(
                Mesh<1>{local_num_fd_points, Spectral::Basis::FiniteDifference,
                        local_subcell_quadrature}));
      });
  if (dg_mesh.basis(0) == Spectral::Basis::Cartoon) {
    static const Matrix cartoon_matrix{1, 1, 1.0};
    return cartoon_matrix;
  } else if (dg_mesh.basis(0) == Spectral::Basis::ZernikeB1) {
    return zernike_b1_cache(dg_mesh.extents(0), subcell_extents,
                            subcell_quadrature, parity);
  } else {
    return legendre_cache(dg_mesh.extents(0), subcell_extents,
                          dg_mesh.quadrature(0), subcell_quadrature);
  }
}

namespace {
double get_sixth_order_integration_coefficient(const size_t num_pts,
                                               const size_t index) {
  if (num_pts == 3) {
    return gsl::at(std::array<double, 3>{{9. / 8., 3. / 4., 9. / 8.}}, index);
  } else if (num_pts == 4) {
    return gsl::at(
        std::array<double, 4>{{13. / 12., 11. / 12., 11. / 12., 13. / 12.}},
        index);
  } else if (num_pts == 5) {
    return gsl::at(
        std::array<double, 5>{{1375. / 1152., 125. / 288., 335. / 192.,
                               125. / 288., 1375. / 1152.}},
        index);
  } else if (num_pts == 6) {
    return gsl::at(
        std::array<double, 6>{{741. / 640., 417. / 640., 381. / 320.,
                               381. / 320., 417. / 640., 741. / 640.}},
        index);
  } else if (num_pts == 7) {
    return gsl::at(
        std::array<double, 7>{{741. / 640., 3547. / 5760., 8111. / 5760.,
                               611. / 960., 8111. / 5760., 3547. / 5760.,
                               741. / 640.}},
        index);
  } else if (num_pts == 8) {
    return gsl::at(
        std::array<double, 8>{{1663. / 1440., 227. / 360., 323. / 240.,
                               139. / 160., 139. / 160., 323. / 240.,
                               227. / 360., 1663. / 1440.}},
        index);
  } else if (num_pts == 9) {
    return gsl::at(
        std::array<double, 9>{{1663. / 1440., 227. / 360., 1547. / 1152.,
                               245. / 288., 3001. / 2880., 245. / 288.,
                               1547. / 1152., 227. / 360., 1663. / 1440.}},
        index);
  } else if (num_pts == 10) {
    return gsl::at(
        std::array<double, 10>{{1375. / 1152., 125. / 288., 335. / 192.,
                                125. / 288., 1375. / 1152., 1375. / 1152.,
                                125. / 288., 335. / 192., 125. / 288.,
                                1375. / 1152.}},
        index);
  } else if (num_pts == 11) {
    return gsl::at(
        std::array<double, 11>{{1375. / 1152., 125. / 288., 335. / 192.,
                                2483. / 5760., 7183. / 5760., 863. / 960.,
                                7183. / 5760., 2483. / 5760., 335. / 192.,
                                125. / 288., 1375. / 1152.}},
        index);
  } else if (num_pts == 12) {
    return gsl::at(
        std::array<double, 12>{{1375. / 1152., 125. / 288., 335. / 192.,
                                2483. / 5760., 3583. / 2880., 2743. / 2880.,
                                2743. / 2880., 3583. / 2880., 2483. / 5760.,
                                335. / 192., 125. / 288., 1375. / 1152.}},
        index);
  } else if (num_pts == 13) {
    return gsl::at(
        std::array<double, 13>{{1375. / 1152., 125. / 288., 335. / 192.,
                                2483. / 5760., 3583. / 2880., 1823. / 1920.,
                                2897. / 2880., 1823. / 1920., 3583. / 2880.,
                                2483. / 5760., 335. / 192., 125. / 288.,
                                1375. / 1152.}},
        index);
  } else if (num_pts == 14) {
    return gsl::at(
        std::array<double, 14>{{1375. / 1152., 125. / 288., 335. / 192.,
                                2483. / 5760., 3583. / 2880., 1823. / 1920.,
                                5777. / 5760., 5777. / 5760., 1823. / 1920.,
                                3583. / 2880., 2483. / 5760., 335. / 192.,
                                125. / 288., 1375. / 1152.}},
        index);
  } else if (num_pts >= 15) {
    return gsl::at(
        std::array<double, 8>{{1375. / 1152., 125. / 288., 335. / 192.,
                               2483. / 5760., 3583. / 2880., 1823. / 1920.,
                               5777. / 5760., 1.}},

        index <= 7             ? index
        : index >= num_pts - 8 ? num_pts - 1 - index
                               : 7);
  }
  ERROR("Cannot get coefficients for a mesh with only '"
        << num_pts << "' points. We need at least 5 points.");
}

// Function to get the 1d integration coefficient from the 1, 2, or 3d spatial
// location. The 1d coefficient is a product of the 3 coefficients in each
// direction.
template <size_t Dim>
double integration_weight(const Index<Dim>& extents, const Index<Dim>& index) {
  double result = get_sixth_order_integration_coefficient(extents[0], index[0]);
  for (size_t d = 1; d < Dim; ++d) {
    result *= get_sixth_order_integration_coefficient(extents[d], index[d]);
  }
  return result;
}

template <Spectral::Quadrature QuadratureType, size_t Dim>
Matrix reconstruction_matrix_cache_impl_helper(
    const Index<Dim>& dg_extents, const Index<Dim>& subcell_extents) {
  if constexpr (QuadratureType == Spectral::Quadrature::SphericalSymmetry or
                QuadratureType == Spectral::Quadrature::AxialSymmetry) {
    ASSERT(Dim == 1,
           "Cartoon basis should never be the first basis: only a mesh slice "
           "should get here, got Dim = "
               << Dim);
    return Matrix{1, 1, 1.0};
  } else {
    // The reconstruction matrix solves a least-squares fit of the subcell data
    // by the DG basis, constrained to preserve the integral over the cell. In
    // block form the system is
    //
    //   M = [[A, -w], [w^T, 0]],   A = 2 P^T P,
    //
    // with P the projection from the DG grid to the subcell grid and w the DG
    // quadrature weights. Every ingredient is a tensor product of
    // one-dimensional factors: P = (x) P_d, so A^-1 = (1/2) (x) (P_d^T P_d)^-1,
    // and w, the integration weights and the cell widths all factor per
    // dimension. Only the single conservation constraint does not, but a
    // bordered system has a closed-form inverse in terms of A^-1, u = A^-1 w
    // and s = w^T u, so the whole matrix can be assembled from the 1d factors
    // plus one rank-one correction. That avoids inverting the (n+1) x (n+1)
    // system, whose cost grows like the cube of the total number of DG points.
    const size_t num_pts = dg_extents.product();
    const size_t num_subcells = subcell_extents.product();

    // Per dimension: the least-squares reconstruction B_d = (P_d^T P_d)^-1
    // P_d^T, the weight image y_d = (P_d^T P_d)^-1 w_d, its norm s_d =
    // w_d^T y_d, the subcell image z_d = P_d y_d, and the integration weights
    // v_d of the constraint row.
    std::array<Matrix, Dim> one_d_reconstruction{};
    std::array<DataVector, Dim> weight_image{};
    std::array<DataVector, Dim> subcell_image{};
    std::array<DataVector, Dim> constraint_row{};
    std::array<double, Dim> weight_norm{};

    for (size_t d = 0; d < Dim; ++d) {
      const size_t n_d = dg_extents[d];
      const size_t m_d = subcell_extents[d];
      const Matrix p_d =
          Spectral::interpolation_matrix<Spectral::Basis::Legendre,
                                         QuadratureType>(
              n_d,
              Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                           Spectral::Quadrature::CellCentered>(
                  m_d));

      Matrix gram(n_d, n_d, 0.0);
      for (size_t i = 0; i < n_d; ++i) {
        for (size_t j = 0; j < n_d; ++j) {
          double sum = 0.0;
          for (size_t k = 0; k < m_d; ++k) {
            sum += p_d(k, i) * p_d(k, j);
          }
          gram(i, j) = sum;
        }
      }
      const Matrix inv_gram = inv(gram);

      Matrix b_d(n_d, m_d, 0.0);
      for (size_t i = 0; i < n_d; ++i) {
        for (size_t k = 0; k < m_d; ++k) {
          double sum = 0.0;
          for (size_t j = 0; j < n_d; ++j) {
            sum += inv_gram(i, j) * p_d(k, j);
          }
          b_d(i, k) = sum;
        }
      }
      gsl::at(one_d_reconstruction, d) = std::move(b_d);

      const DataVector& w_d =
          Spectral::quadrature_weights<Spectral::Basis::Legendre,
                                       QuadratureType>(n_d);
      DataVector y_d(n_d, 0.0);
      for (size_t i = 0; i < n_d; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n_d; ++j) {
          sum += inv_gram(i, j) * w_d[j];
        }
        y_d[i] = sum;
      }
      double s_d = 0.0;
      for (size_t i = 0; i < n_d; ++i) {
        s_d += w_d[i] * y_d[i];
      }
      gsl::at(weight_norm, d) = s_d;

      DataVector z_d(m_d, 0.0);
      for (size_t k = 0; k < m_d; ++k) {
        double sum = 0.0;
        for (size_t i = 0; i < n_d; ++i) {
          sum += p_d(k, i) * y_d[i];
        }
        z_d[k] = sum;
      }
      gsl::at(subcell_image, d) = std::move(z_d);
      gsl::at(weight_image, d) = std::move(y_d);

      DataVector v_d(m_d, 0.0);
      for (size_t k = 0; k < m_d; ++k) {
        v_d[k] = 2.0 / static_cast<double>(m_d) *
                 get_sixth_order_integration_coefficient(m_d, k);
      }
      gsl::at(constraint_row, d) = std::move(v_d);
    }

    double s_total = 1.0;
    for (size_t d = 0; d < Dim; ++d) {
      s_total *= gsl::at(weight_norm, d);
    }

    // The rank-one correction only depends on the subcell index through
    // `v - z`, so build that once.
    DataVector constraint_defect(num_subcells, 0.0);
    for (IndexIterator<Dim> subcell_it(subcell_extents); subcell_it;
         ++subcell_it) {
      double v_product = 1.0;
      double z_product = 1.0;
      for (size_t d = 0; d < Dim; ++d) {
        v_product *= gsl::at(constraint_row, d)[subcell_it()[d]];
        z_product *= gsl::at(subcell_image, d)[subcell_it()[d]];
      }
      constraint_defect[subcell_it.collapsed_index()] =
          (v_product - z_product) / s_total;
    }

    Matrix reduced_recons_matrix(num_pts, num_subcells);
    for (IndexIterator<Dim> dg_it(dg_extents); dg_it; ++dg_it) {
      double y_product = 1.0;
      for (size_t d = 0; d < Dim; ++d) {
        y_product *= gsl::at(weight_image, d)[dg_it()[d]];
      }
      const size_t dg_index = dg_it.collapsed_index();
      for (IndexIterator<Dim> subcell_it(subcell_extents); subcell_it;
           ++subcell_it) {
        double b_product = 1.0;
        for (size_t d = 0; d < Dim; ++d) {
          b_product *=
              gsl::at(one_d_reconstruction, d)(dg_it()[d], subcell_it()[d]);
        }
        const size_t subcell_index = subcell_it.collapsed_index();
        reduced_recons_matrix(dg_index, subcell_index) =
            b_product + y_product * constraint_defect[subcell_index];
      }
    }
    return reduced_recons_matrix;
  }
}

// The subcell extents are `2 n - 1` of the DG extents in each dimension (see
// `fd::mesh`), so the DG extents alone identify the matrix. Enumerating one
// range per dimension at compile time would mean thousands of instantiations of
// the generator, so a RuntimeCache is used instead: a single instantiation, a
// table indexed arithmetically at runtime, and entries built on first use.
template <Spectral::Quadrature QuadratureType, size_t Dim>
const Matrix& reconstruction_matrix_impl(const Mesh<Dim>& dg_mesh) {
  if constexpr (QuadratureType == Spectral::Quadrature::SphericalSymmetry or
                QuadratureType == Spectral::Quadrature::AxialSymmetry) {
    // A Cartoon direction carries a single grid point, so the reconstruction
    // is the identity whatever the extent. Returning it before the cache below
    // keeps each Cartoon quadrature from reserving a full table per dimension
    // to hold copies of one 1x1 matrix.
    ASSERT(Dim == 1,
           "Cartoon basis should never be the first basis: only a mesh slice "
           "should get here, got Dim = "
               << Dim);
    static const Matrix cartoon_matrix{1, 1, 1.0};
    return cartoon_matrix;
  } else {
    constexpr size_t max_pts =
        Spectral::maximum_number_of_points<Spectral::Basis::Legendre,
                                           QuadratureType> +
        1;
    const auto generator = [](const auto... dg_extents_1d) {
      return reconstruction_matrix_cache_impl_helper<QuadratureType>(
          Index<Dim>{dg_extents_1d...}, Index<Dim>{(2 * dg_extents_1d - 1)...});
    };
    if constexpr (Dim == 1) {
      const static auto cache =
          make_runtime_cache<CacheRange<1_st, max_pts>>(generator);
      return cache(dg_mesh.extents(0));
    } else if constexpr (Dim == 2) {
      const static auto cache =
          make_runtime_cache<CacheRange<1_st, max_pts>,
                             CacheRange<1_st, max_pts>>(generator);
      return cache(dg_mesh.extents(0), dg_mesh.extents(1));
    } else {
      const static auto cache =
          make_runtime_cache<CacheRange<1_st, max_pts>,
                             CacheRange<1_st, max_pts>,
                             CacheRange<1_st, max_pts>>(generator);
      return cache(dg_mesh.extents(0), dg_mesh.extents(1), dg_mesh.extents(2));
    }
  }
}
}  // namespace

template <size_t Dim>
const Matrix& reconstruction_matrix(
    const Mesh<Dim>& dg_mesh,
    [[maybe_unused]] const Index<Dim>& subcell_extents) {
  ASSERT(dg_mesh.basis(0) == Spectral::Basis::Legendre or
             dg_mesh.basis(0) == Spectral::Basis::Cartoon,
         "FD Subcell reconstruction only supports Legendre or Cartoon bases "
         "right now, got "
             << dg_mesh);
  // Extents may differ between dimensions. A Cartoon dimension would need its
  // own 1x1 interpolation and quadrature weight here, which is not implemented
  // for AllDimsAtOnce; DimByDim reconstruction handles Cartoon meshes by
  // slicing, and supports differing extents in the non-Cartoon dimensions.
  ASSERT(dg_mesh == Mesh<Dim>(dg_mesh.extents().indices(), dg_mesh.basis(0),
                              dg_mesh.quadrature(0)),
         "AllDimsAtOnce reconstruction does not support a Cartoon or otherwise "
         "mixed-basis mesh; use DimByDim for those. Got "
             << dg_mesh);
  // The cache below is keyed on the DG extents alone, which is only sound
  // because `fd::mesh` fixes the subcell extents at `2 n - 1` in every
  // dimension.
#ifdef SPECTRE_DEBUG
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT(subcell_extents[d] == 2 * dg_mesh.extents(d) - 1,
           "The subcell extents must be 2 n - 1 of the DG extents in every "
           "dimension, but got subcell extents "
               << subcell_extents << " for DG mesh " << dg_mesh);
  }
#endif

  switch (dg_mesh.quadrature(0)) {
    case Spectral::Quadrature::GaussLobatto:
      return reconstruction_matrix_impl<Spectral::Quadrature::GaussLobatto>(
          dg_mesh);
    case Spectral::Quadrature::Gauss:
      return reconstruction_matrix_impl<Spectral::Quadrature::Gauss>(dg_mesh);
    case Spectral::Quadrature::AxialSymmetry:
      ASSERT(Dim == 1,
             "Cartoon basis should only be used with DimByDim reconstruction, "
             "so only a mesh slice "
             "should get here, got Dim = "
                 << Dim);
      return reconstruction_matrix_impl<Spectral::Quadrature::AxialSymmetry>(
          dg_mesh);
    case Spectral::Quadrature::SphericalSymmetry:
      ASSERT(Dim == 1,
             "Cartoon basis should only be used with DimByDim reconstruction, "
             "so only a mesh slice "
             "should get here, got Dim = "
                 << Dim);
      return reconstruction_matrix_impl<
          Spectral::Quadrature::SphericalSymmetry>(dg_mesh);
    default:
      ERROR(
          "Unsupported quadrature type in FD subcell reconstruction "
          "matrix, got "
          << dg_mesh.quadrature(0));
  };
}

const Matrix& reconstruction_matrix(const Mesh<1>& dg_mesh,
                                    const size_t subcell_extents,
                                    const Spectral::Parity parity) {
  ASSERT(dg_mesh.basis(0) == Spectral::Basis::ZernikeB1 and
             dg_mesh.quadrature(0) == Spectral::Quadrature::GaussRadauUpper,
         "reconstruction_matrix with a Parity argument only supports "
         "ZernikeB1 with GaussRadauUpper quadrature, but got "
             << dg_mesh);
  ASSERT(parity != Spectral::Parity::Uninitialized,
         "Parity must be set when using ZernikeB1");
  ASSERT(subcell_extents >= dg_mesh.extents(0),
         "Subcell extents ("
             << subcell_extents << ") must be >= DG extents ("
             << dg_mesh.extents(0)
             << ") for ZernikeB1 reconstruction to be well-posed (P^T P "
                "would be singular)");

  static const auto cache = make_runtime_cache<
      CacheRange<Spectral::minimum_number_of_points<
                     Spectral::Basis::ZernikeB1,
                     Spectral::Quadrature::GaussRadauUpper>,
                 Spectral::maximum_number_of_points<
                     Spectral::Basis::ZernikeB1,
                     Spectral::Quadrature::GaussRadauUpper> +
                     1>,
      CacheRange<Spectral::minimum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered>,
                 Spectral::maximum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered> +
                     1>,
      CacheEnumeration<Spectral::Parity, Spectral::Parity::Even,
                       Spectral::Parity::Odd>>(
      [](const size_t n_dg, const size_t n_fd,
         const Spectral::Parity local_parity) {
        const Matrix& proj_matrix = projection_matrix(
            Mesh<1>{n_dg, Spectral::Basis::ZernikeB1,
                    Spectral::Quadrature::GaussRadauUpper},
            n_fd, Spectral::Quadrature::CellCentered, local_parity);

        // Compute 2 P^T P (n_dg x n_dg)
        Matrix two_pt_p(n_dg, n_dg, 0.0);
        dgemm_<true>('T', 'N', n_dg, n_dg, n_fd, 2.0, proj_matrix.data(),
                     proj_matrix.spacing(), proj_matrix.data(),
                     proj_matrix.spacing(), 0.0, two_pt_p.data(),
                     two_pt_p.spacing());

        // RHS top block: 2 P^T (n_dg x n_fd)
        Matrix rhs_top(n_dg, n_fd);
        for (size_t k = 0; k < n_fd; ++k) {
          for (size_t l = 0; l < n_dg; ++l) {
            rhs_top(l, k) = 2.0 * proj_matrix(k, l);
          }
        }

        // For odd parity, the conservation constraint
        //
        //   sum_i w_i^DG u(r_i^DG) = sum_j w_j^FD u(r_j^FD)
        //
        // is inconsistent because the DG quadrature points are the roots of
        // Pi^{m=0}(2*n_dg, r) = Q^0_{2*n_dg} - Q^0_{2*n_dg-2}, which are
        // optimised for integrating even-parity (m=0) functions Q^0_k. They
        // are not accurate for odd-parity (m=1) integrands Q^1_k.
        //
        // For example, with n_dg=2 (points 0.447, 1.0; weights 0.833, 0.167)
        // and u(r) = r (the lowest odd-parity basis function):
        //
        //   sum_i w_i^DG * r_i  =  0.0787   (DG quadrature, inaccurate for m=1)
        //   sum_j w_j^FD * r_j  =  0.5000   (6th-order FD, = int_0^1 r dr)
        //
        // Imposing this as a constraint via a Lagrange multiplier forces R to
        // satisfy a false equation, corrupting the reconstruction. The pure
        // pseudoinverse R = (2 P^T P)^{-1} 2P^T does not impose a conservation
        // constraint and still satisfies R P = I exactly by construction.
        if (local_parity == Spectral::Parity::Odd) {
          // Use the pure pseudoinverse R = (P^T P)^{-1} P^T, which satisfies
          // R . P = (P^T P)^{-1} (P^T P) = I exactly without relying on any
          // conservation constraint.
          const Matrix inv_two_pt_p = inv(two_pt_p);
          Matrix result(n_dg, n_fd);
          dgemm_<true>('N', 'N', n_dg, n_fd, n_dg, 1.0, inv_two_pt_p.data(),
                       inv_two_pt_p.spacing(), rhs_top.data(),
                       rhs_top.spacing(), 0.0, result.data(), result.spacing());
          return result;
        }

        // Even parity: the conservation constraint is self-consistent (see
        // discussion above), so use Lagrange-constrained least-squares.
        const size_t aug_size = n_dg + 1;
        Matrix lhs(aug_size, aug_size, 0.0);
        for (size_t l = 0; l < n_dg; ++l) {
          for (size_t j = 0; j < n_dg; ++j) {
            lhs(l, j) = two_pt_p(l, j);
          }
        }

        // Lagrange multiplier row and column (conservation constraint)
        const DataVector& dg_weights =
            Spectral::quadrature_weights<Spectral::Basis::ZernikeB1,
                                         Spectral::Quadrature::GaussRadauUpper>(
                n_dg);
        for (size_t i = 0; i < n_dg; ++i) {
          lhs(i, n_dg) = -dg_weights[i];
          lhs(n_dg, i) = dg_weights[i];
        }

        // Fill augmented RHS: rows 0..n_dg-1 hold 2 P^T, row n_dg holds the
        // FD integration weights. The ZernikeB1 physical domain is [0,1], so
        // the cell width is 1/n_fd (not 2/n_fd as for Legendre on [-1,1]).
        Matrix rhs(aug_size, n_fd);
        for (size_t k = 0; k < n_fd; ++k) {
          for (size_t l = 0; l < n_dg; ++l) {
            rhs(l, k) = rhs_top(l, k);
          }
        }
        const double delta = 1.0 / static_cast<double>(n_fd);
        for (size_t i = 0; i < n_fd; ++i) {
          rhs(n_dg, i) =
              delta * get_sixth_order_integration_coefficient(n_fd, i);
        }

        const Matrix inv_lhs = inv(lhs);
        Matrix full_recons(aug_size, n_fd);
        dgemm_<true>('N', 'N', inv_lhs.rows(), n_fd, inv_lhs.columns(), 1.0,
                     inv_lhs.data(), inv_lhs.spacing(), rhs.data(),
                     rhs.spacing(), 0.0, full_recons.data(),
                     full_recons.spacing());
        // Drop the Lagrange multiplier row
        Matrix result(n_dg, n_fd);
        for (size_t i = 0; i < n_dg; ++i) {
          for (size_t j = 0; j < n_fd; ++j) {
            result(i, j) = full_recons(i, j);
          }
        }
        return result;
      });

  return cache(dg_mesh.extents(0), subcell_extents, parity);
}

namespace {
// The logical coordinates of the centers of the `ghost_zone_size`
// subcells nearest the `side` face of the element
DataVector ghost_zone_target_points(const size_t num_fd_points,
                                    const size_t ghost_zone_size,
                                    const Side side) {
  const DataVector& fd_points =
      Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::CellCentered>(
          num_fd_points);
  DataVector target_points(ghost_zone_size);
  for (size_t i = 0; i < ghost_zone_size; ++i) {
    target_points[i] =
        fd_points[side == Side::Lower ? i
                                      : (num_fd_points - ghost_zone_size + i)];
  }
  return target_points;
}
}  // namespace

const Matrix& projection_matrix(const Mesh<1>& dg_mesh,
                                const size_t subcell_extents,
                                const size_t ghost_zone_size, const Side side,
                                const Spectral::Parity parity) {
  static constexpr size_t max_ghost_zone_size = 5;
  ASSERT(dg_mesh.basis(0) == Spectral::Basis::Legendre or
             dg_mesh.basis(0) == Spectral::Basis::ZernikeB1,
         "FD Subcell ghost zone projection only supports Legendre or ZernikeB1 "
         "bases right now but got basis "
             << dg_mesh.basis(0));
  ASSERT(dg_mesh.basis(0) != Spectral::Basis::ZernikeB1 or
             dg_mesh.quadrature(0) == Spectral::Quadrature::GaussRadauUpper,
         "ZernikeB1 ghost zone projection only supports GaussRadauUpper "
         "quadrature, but got "
             << dg_mesh);
  ASSERT(dg_mesh.basis(0) != Spectral::Basis::ZernikeB1 or
             parity != Spectral::Parity::Uninitialized,
         "Parity must be set when using ZernikeB1");
  ASSERT(ghost_zone_size <= max_ghost_zone_size and ghost_zone_size >= 2,
         "ghost_zone_size must be in [2, " << max_ghost_zone_size
                                           << " ] but got " << ghost_zone_size);
  static const auto zernike_b1_cache = make_static_cache<
      CacheRange<Spectral::minimum_number_of_points<
                     Spectral::Basis::ZernikeB1,
                     Spectral::Quadrature::GaussRadauUpper>,
                 Spectral::maximum_number_of_points<
                     Spectral::Basis::ZernikeB1,
                     Spectral::Quadrature::GaussRadauUpper> +
                     1>,
      CacheRange<Spectral::minimum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered>,
                 Spectral::maximum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered> +
                     1>,
      CacheRange<2_st, max_ghost_zone_size + 1>,
      CacheEnumeration<Side, Side::Lower, Side::Upper>,
      CacheEnumeration<Spectral::Parity, Spectral::Parity::Even,
                       Spectral::Parity::Odd>>(
      [](const size_t local_num_dg_points, const size_t local_num_fd_points,
         const size_t local_ghost_zone_size, const Side local_side,
         const Spectral::Parity local_parity) {
        return Spectral::interpolation_matrix<
            Spectral::Basis::ZernikeB1, Spectral::Quadrature::GaussRadauUpper>(
            local_num_dg_points,
            ghost_zone_target_points(local_num_fd_points, local_ghost_zone_size,
                                     local_side),
            local_parity);
      });
  static const auto legendre_cache = make_static_cache<
      CacheRange<
          Spectral::minimum_number_of_points<
              Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>,
          Spectral::maximum_number_of_points<
              Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto> +
              1>,
      CacheRange<Spectral::minimum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered>,
                 Spectral::maximum_number_of_points<
                     Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered> +
                     1>,
      CacheRange<2_st, max_ghost_zone_size + 1>,
      CacheEnumeration<Spectral::Quadrature, Spectral::Quadrature::Gauss,
                       Spectral::Quadrature::GaussLobatto>,
      CacheEnumeration<Side, Side::Lower, Side::Upper>>(
      [](const size_t local_num_dg_points, const size_t local_num_fd_points,
         const size_t local_ghost_zone_size,
         const Spectral::Quadrature dg_quadrature, const Side local_side) {
        return Spectral::interpolation_matrix(
            Mesh<1>{local_num_dg_points, Spectral::Basis::Legendre,
                    dg_quadrature},
            ghost_zone_target_points(local_num_fd_points, local_ghost_zone_size,
                                     local_side));
      });
  if (dg_mesh.basis(0) == Spectral::Basis::ZernikeB1) {
    return zernike_b1_cache(dg_mesh.extents(0), subcell_extents,
                            ghost_zone_size, side, parity);
  }
  return legendre_cache(dg_mesh.extents(0), subcell_extents, ghost_zone_size,
                        dg_mesh.quadrature(0), side);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                             \
  template const Matrix& reconstruction_matrix(const Mesh<GET_DIM(data)>&, \
                                               const Index<GET_DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace evolution::dg::subcell::fd
