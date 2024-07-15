// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Interpolation/PolynomialInterpolation.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module supplied by Charm++, we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
void dump_equilibrium_eos(
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const size_t number_of_log10_number_density_points_for_dump,
    const std::string& output_file_name,
    const double lower_bound_rest_mass_density_cgs,
    const double upper_bound_rest_mass_density_cgs,
    const std::string& pressure_of_density_filename) {
  using std::log10;
  using std::pow;
  // Baryon mass, used to go from number density to rest mass
  // density. I.e. `rho_cgs = n_cgs * baryon_mass`, where `n_gcs` is the number
  // density in CGS units. This is the baryon mass that RotNS uses. This
  // might be different from the baryon mass that the EoS uses.
  //
  // https://github.com/sxs-collaboration/spectre/issues/4694
  const double baryon_mass_of_rotns_cgs =
      hydro::units::geometric::default_baryon_mass *
      hydro::units::cgs::mass_unit;
  const double log10_lower_bound_number_density_cgs =
      log10(lower_bound_rest_mass_density_cgs / baryon_mass_of_rotns_cgs);
  const double log10_upper_bound_number_density_cgs =
      log10(upper_bound_rest_mass_density_cgs / baryon_mass_of_rotns_cgs);
  const double delta_log_number_density_cgs =
      (log10_upper_bound_number_density_cgs -
       log10_lower_bound_number_density_cgs) /
      static_cast<double>(number_of_log10_number_density_points_for_dump - 1);

  if (file_system::check_if_file_exists(output_file_name)) {
    ERROR_NO_TRACE("File " << output_file_name
                           << " already exists. Refusing to overwrite.");
  }
  std::ofstream outfile(output_file_name.c_str());

  if (not file_system::check_if_file_exists(pressure_of_density_filename)) {
    ERROR("Cannot open file " << pressure_of_density_filename << ".\n");
  }
  std::ifstream p_of_rho_file(pressure_of_density_filename);

  size_t num_density_points = -1;
  p_of_rho_file >> num_density_points;

  DataVector log_rest_mass_density_interp = DataVector(num_density_points);
  DataVector log_pressure_interp = DataVector(num_density_points);
  for (size_t i = 0; i < num_density_points; i++) {
    p_of_rho_file >> log_rest_mass_density_interp[i] >> log_pressure_interp[i];
  }

  for (size_t log10_number_density_index = 0;
       log10_number_density_index <
       number_of_log10_number_density_points_for_dump;
       ++log10_number_density_index) {
    using std::pow;
    const double number_density_cgs =
        pow(10.0, log10_lower_bound_number_density_cgs +
                      static_cast<double>(log10_number_density_index) *
                          delta_log_number_density_cgs);

    // Note: we will want to add the baryon mass to our EOS interface.
    //
    // https://github.com/sxs-collaboration/spectre/issues/4694
    const Scalar<double> rest_mass_density_geometric{
        number_density_cgs * cube(hydro::units::cgs::length_unit) *
        eos.baryon_mass()};
    Scalar<double> pressure_geometric =
        make_with_value<Scalar<double>>(rest_mass_density_geometric, 0.0);

    constexpr size_t stencil_size = 4;

    for (size_t i = 0; i < number_of_log10_number_density_points_for_dump;
         ++i) {
      const double target_log_density =
          log10(get_element(get(rest_mass_density_geometric), i));

      size_t density_index = 0;
      for (size_t j = 0; j < num_density_points; ++j) {
        if (log_rest_mass_density_interp[j] > target_log_density) {
          density_index = j - 1;
          break;
        }
      }

      const size_t density_stencil_index = static_cast<size_t>(std::clamp(
          static_cast<int>(density_index) - static_cast<int>(stencil_size) / 2,
          0, static_cast<int>(num_density_points - stencil_size)));

      double target_log_pressure{std::numeric_limits<double>::signaling_NaN()};
      const gsl::not_null<double*> target_var =
          make_not_null(&target_log_pressure);
      const double max_pressure_ratio_for_linear_interpolation = 1.e2;
      const auto pressure_stencil = gsl::make_span(
          &log_pressure_interp[density_stencil_index], stencil_size);
      const auto density_stencil = gsl::make_span(
          &log_rest_mass_density_interp[density_stencil_index], stencil_size);

      double error_y = 0.0;
      if (const auto min_max_iters = std::minmax_element(
              pressure_stencil.begin(), pressure_stencil.end());
          *min_max_iters.second >
          max_pressure_ratio_for_linear_interpolation * *min_max_iters.first) {
        std::array<double, 2> density_linear{
            {std::numeric_limits<double>::signaling_NaN(),
             std::numeric_limits<double>::signaling_NaN()}};
        std::array<double, 2> pressure_linear{
            {std::numeric_limits<double>::signaling_NaN(),
             std::numeric_limits<double>::signaling_NaN()}};
        for (size_t k = 0; k < stencil_size - 1; ++k) {
          if (density_stencil[k] <= target_log_density and
              target_log_density <= density_stencil[k + 1]) {
            density_linear[0] = density_stencil[k];
            density_linear[1] = density_stencil[k + 1];
            pressure_linear[0] = gsl::at(pressure_stencil, k);
            pressure_linear[1] = gsl::at(pressure_stencil, k + 1);
            break;
          }
        }
        intrp::polynomial_interpolation<1>(
            target_var, make_not_null(&error_y), target_log_density,
            gsl::make_span(pressure_linear.data(), pressure_linear.size()),
            gsl::make_span(density_linear.data(), density_linear.size()));
      } else {
        intrp::polynomial_interpolation<stencil_size - 1>(
            target_var, make_not_null(&error_y), target_log_density,
            pressure_stencil, density_stencil);
      }

      get_element(get(pressure_geometric), i) = pow(10.0, target_log_pressure);
    }

    const Scalar<double> electron_fraction =
        eos.equilibrium_electron_fraction_from_density_temperature(
            rest_mass_density_geometric, pressure_geometric);

    const Scalar<double> specific_internal_energy_geometric =
        eos.specific_internal_energy_from_density_and_pressure(
            rest_mass_density_geometric, pressure_geometric);  //,
    // Scalar<double>{0.3});
    pressure_geometric = eos.pressure_from_density_and_energy(
        rest_mass_density_geometric, specific_internal_energy_geometric);
    const Scalar<double> energy_density_geometric{
        get(rest_mass_density_geometric) *
        (1. + get(specific_internal_energy_geometric))};

    // Note: the energy density is divided by c^2
    const double energy_density_cgs = get(energy_density_geometric) *
                                      hydro::units::cgs::rest_mass_density_unit;

    // should be dyne cm^(-3)
    const double pressure_cgs =
        get(pressure_geometric) * hydro::units::cgs::pressure_unit;

    outfile << std::scientific << std::setw(24) << std::setprecision(14)
            << log10(number_density_cgs) << std::setw(24)
            << std::setprecision(14) << log10(energy_density_cgs)
            << std::setw(24) << std::setprecision(14) << log10(pressure_cgs)
            << std::endl;
  }
  outfile.close();
}

namespace OptionTags {
struct NumberOfPoints {
  using type = size_t;
  static constexpr Options::String help = {
      "Number of points at which to dump the EoS"};
};

struct OutputFileName {
  using type = std::string;
  static constexpr Options::String help = {
      "Name of the output file to dump the EoS to, including file extension."};
};

struct LowerBoundRestMassDensityCgs {
  using type = double;
  static constexpr Options::String help = {
      "Lower bound of rest mass density in CGS units."};
};

struct UpperBoundRestMassDensityCgs {
  using type = double;
  static constexpr Options::String help = {
      "Upper bound of rest mass density in CGS units."};
};

struct PressureOfDensityFilename {
  using type = std::string;
  static constexpr Options::String help = {
      "File from which the T(rho) interpolation is constructed. Must be a two "
      "column file with density as the first column and pressure as the "
      "second. The file must also have a header which is the integer number "
      "of entries contained within."};
};
}  // namespace OptionTags
}  // namespace

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  bpo::positional_options_description pos_desc;

  const std::string help_string =
      "Dump a relativistic equilibrium equation of state to disk.\n"
      "All options controlling input and output are read from the input file.";

  bpo::options_description desc(help_string);
  desc.add_options()("help,h,", "show this help message")(
      "input-file", bpo::value<std::string>()->required(), "Input file name")(
      "check-options", "Check input file options");

  bpo::variables_map vars;

  bpo::store(bpo::command_line_parser(argc, argv)
                 .positional(pos_desc)
                 .options(desc)
                 .run(),
             vars);

  if (vars.count("help") != 0u or vars.count("input-file") == 0u) {
    Parallel::printf("%s\n", desc);
    return 1;
  }

  using option_list =
      tmpl::list<hydro::OptionTags::InitialDataEquationOfState<true, 3>,
                 OptionTags::NumberOfPoints, OptionTags::OutputFileName,
                 OptionTags::LowerBoundRestMassDensityCgs,
                 OptionTags::UpperBoundRestMassDensityCgs,
                 OptionTags::PressureOfDensityFilename>;

  Options::Parser<option_list> option_parser(help_string);
  option_parser.parse_file(vars["input-file"].as<std::string>());

  if (vars.count("check-options") != 0) {
    // Force all the options to be created.
    option_parser.template apply<option_list>([](auto... args) {
      (void)std::initializer_list<char>{((void)args, '0')...};
    });
    Parallel::printf("\n%s parsed successfully!\n",
                     vars["input-file"].as<std::string>());

    return 0;
  }

  const auto options =
      option_parser.template apply<option_list>([](auto... args) {
        return tuples::tagged_tuple_from_typelist<option_list>(
            std::move(args)...);
      });

  dump_equilibrium_eos(
      *get<hydro::OptionTags::InitialDataEquationOfState<true, 3>>(options),
      get<OptionTags::NumberOfPoints>(options),
      get<OptionTags::OutputFileName>(options),
      get<OptionTags::LowerBoundRestMassDensityCgs>(options),
      get<OptionTags::UpperBoundRestMassDensityCgs>(options),
      get<OptionTags::PressureOfDensityFilename>(options));

  return 0;
}
