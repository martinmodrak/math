#include <stan/math/rev.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/math/rev/fun/util.hpp>
#include <test/unit/math/expect_near_rel.hpp>
#include <fstream>
#include <iostream>
#include <limits>

// Set to true to write CSV file with recurrence test results for analysis
bool output_debug_csv = true;

using stan::math::LOG_TWO;
using stan::math::besselk_internal::ComputationType;
using stan::math::besselk_internal::choose_computation_type;

using stan::math::log_diff_exp;
using stan::math::log_modified_bessel_second_kind_frac;
using stan::math::log_sum_exp;
using stan::math::recover_memory;
using stan::math::var;

std::vector<double> v_to_test = {0,
                                    3.15e-7,
                                    2.62e-6,
                                    1.3e-5,
                                    9.2e-5,
                                    0.0026,
                                    0.0843,
                                    0.17345,
                                    1,
                                    1.63,
                                    7.42,
                                    42.42424,
                                    86.5,
                                    113.8,
                                    148.7565,
                                    180.6,
                                    246.3,
                                    300.5,
                                    513.6,
                                    712.456,
                                    714.456,
                                    1235.6,
                                    8656,
                                    15330.75,
                                    37634.2,
                                    85323 };

std::vector<double> z_to_test
    = {1.48e-7, 3.6e-6,   7.248e-5, 4.32e-4, 8.7e-3, 0.04523, 0.17532,
       1,       3,        11.32465, 105.6,   1038.4, 4236,    11457.6,
       62384,   105321.6, 158742.3, 196754,  1.98e6};

std::vector<double> hard_v_boundaries = {
  stan::math::besselk_internal::rothwell_max_v,
  stan::math::besselk_internal::gamma_max_v,
  stan::math::besselk_internal::gamma_low_v,
  stan::math::besselk_internal::trapezoid_min_v,
  stan::math::besselk_internal::asymp_z_log_min_v
};

std::vector<double> hard_z_boundaries = {
  stan::math::besselk_internal::rothwell_max_z,
  stan::math::besselk_internal::gamma_max_z,
  stan::math::besselk_internal::gamma_low_z
};


double allowed_recurrence_error = 1e-7;

const char* computation_type_to_string(ComputationType c) {
  switch (c) {
    case ComputationType::Rothwell:
      return "Rothwell";
    case ComputationType::Asymp_v:
      return "Asymp_v";
    case ComputationType::Asymp_z:
      return "Asymp_z";
    case ComputationType::Asymp_z_log:
      return "Asymp_z_log";
    case ComputationType::FirstKind:
      return "FirstKind";
    case ComputationType::IntegralGamma:
      return "z_Integral_Gamma";
    case ComputationType::TrapezoidCosh:
      return "z_Trapezoid_Cosh";
    case ComputationType::Recurrence_Small_V_Z:
      return "Recurrence_Small_V_Z";
    case ComputationType::SmallZ:
      return "SmallZ";
    default:
      return "Unknown";
  }
}

// TEST(AgradRev, log_modified_bessel_second_kind_frac_can_compute_double) {
//   std::for_each(v_to_test.begin(), v_to_test.end(), [](double v) {
//     std::for_each(z_to_test.begin(), z_to_test.end(), [v](double z) {
//       try {
//           log_modified_bessel_second_kind_frac(v, z);
//       } catch (...) {
//         std::cout << "\nAt v = " << v << ", z = " << z << ":\n";
//         throw;
//       }
//     });
//   });
// }

// TEST(AgradRev, log_modified_bessel_second_kind_frac_can_compute_var) {
//   std::for_each(v_to_test.begin(), v_to_test.end(), [](double v) {
//     std::for_each(z_to_test.begin(), z_to_test.end(), [v](double z) {
//       try {
//         AVAR v_var(v);
//         AVAR z_var(z);
//         log_modified_bessel_second_kind_frac(v_var, z_var);
//       } catch (...) {
//         std::cout << "\nAt v = " << v << ", z = " << z << ":\n";
//         throw;
//       }
//     });
//   });
// }

// Using the recurrence relation (adapted to log)
// http://functions.wolfram.com/Bessel-TypeFunctions/BesselK/17/01/01/
void test_single_pair(const double &v, const double &z, std::ostream* debug_output) {
  using stan::test::expect_near_rel;
  AVAR v_var(v);
  AVAR z_var(z);

  try {
    AVAR left_hand = log_modified_bessel_second_kind_frac(v_var, z_var);
    AVAR right_hand;

    AVAR log_K_vm1
        = log_modified_bessel_second_kind_frac(v_var - 1, z_var);
    AVAR log_K_vm2
        = log_modified_bessel_second_kind_frac(v_var - 2, z_var);

    AVAR log_K_vp1
        = log_modified_bessel_second_kind_frac(v_var + 1, z_var);
    AVAR log_K_vp2
        = log_modified_bessel_second_kind_frac(v_var + 2, z_var);

    // Trying to find the most numerically stable way to compute the
    // recursive formula
    if (v > 0) {
      if (v < 1 + 1e-4) {
        if (z > 1e-3) {
          right_hand = log_diff_exp(
              log_K_vp2, LOG_TWO + log(v_var + 1) - log(z_var) + log_K_vp1);
        } else {
          right_hand = log(z_var) - log(2) - log(v_var)
                         + log_diff_exp(log_K_vp1, log_K_vm1);
        }
      } else {
        right_hand = log_sum_exp(
            log_K_vm2, LOG_TWO + log(v_var - 1) - log(z_var) + log_K_vm1);
      }
    } else {
      if (v > -1 - 1e-4) {
        if (z > 1e-3) {
          right_hand
              = log_diff_exp(log_K_vm2, LOG_TWO + log(-v_var + 1)
                                            - log(z_var) + log_K_vm1);
        } else {
          if (v == 0) {
            AVAR right_hand_base
                = log(modified_bessel_second_kind(0, z_var));
            right_hand = var(new stan::math::precomp_vv_vari(
                value_of(right_hand_base), v_var.vi_, right_hand_base.vi_,
                0, 1));
          } else {
            right_hand = log(z_var) - log(2) - log(-v_var)
                          + log_diff_exp(log_K_vm1, log_K_vp1);
          }
        }
      } else {
        right_hand = log_sum_exp(
            log_K_vp2, LOG_TWO + log(-v_var - 1) - log(z_var) + log_K_vp1);
      }
    }

    expect_near_rel("value", left_hand.val(), right_hand.val(), allowed_recurrence_error);

    AVEC x = createAVEC(v_var, z_var);
    VEC g_left;
    left_hand.grad(x, g_left);
    stan::math::set_zero_all_adjoints();
    VEC g_right;
    right_hand.grad(x, g_right);
    expect_near_rel("dv", g_left[0], g_right[0], allowed_recurrence_error);
    expect_near_rel("dz", g_left[1], g_right[1], allowed_recurrence_error);
    if (debug_output != 0) {
      *debug_output << std::setprecision(24) << std::fixed
                    << v << "," << z << ","
                    << computation_type_to_string(
                            choose_computation_type(v, z))
                    << "," << left_hand.val() << "," << g_left[0] 
                    << "," << g_left[1]
                    << "," << right_hand.val() << "," << g_right[0] 
                    << "," << g_right[1]
                    << "," << log_K_vm2 << "," << log_K_vm1 << ","
                    << left_hand << "," << log_K_vp1 << "," << log_K_vp2
                    << std::endl;
      std::flush(*debug_output);
    }
    recover_memory();
  } catch (...) {
    std::cout << "\nAt v = " << v << ", z = " << z << " method = "
              << computation_type_to_string(choose_computation_type(v, z))
              << ":" << std::endl;
    throw;
  }
}

template<typename T>
auto concat(const std::vector<T>& ar1, const std::vector<T>& ar2)
{
    std::vector<T> result(ar1.size() + ar2.size());
    std::copy (ar1.cbegin(), ar1.cend(), result.begin());
    std::copy (ar2.cbegin(), ar2.cend(), result.begin() + ar1.size());
    return result;
}

template<typename T> 
auto operator+(const std::vector<T>& ar, const T& b) {
    std::vector<T> result(ar.size());
    for(int i = 0; i < ar.size(); i++) {
      result[i] = ar[i] + b;
    }
    return result;
}


TEST(AgradRev, log_modified_bessel_second_kind_frac_recurrence) {
  std::ostream* debug_output = 0;
  if (output_debug_csv) {
    debug_output = new std::ofstream("log_besselk_test.csv");
    *debug_output << "v,z,method,left_value,left_dv,left_dz,"
                     "right_value,right_dv,right_dz,value_m2,value_m1,"
                     "value,value_p1,value_p2"
                  << std::endl;
  }

  //For v, we test not only points just around the boundaries, but also
  //+/- 1 from boundaries as the check procedures involve those
  auto all_v = concat(v_to_test, 
    concat(hard_v_boundaries, 
    concat(hard_v_boundaries + 1.0, 
    concat(hard_v_boundaries + (-1.0),
    concat(hard_v_boundaries + 1e-8,
          hard_v_boundaries + (-1e-8))))));
  auto all_z = concat(z_to_test, 
    concat(hard_z_boundaries, 
    concat(hard_z_boundaries + 1e-8,
          hard_z_boundaries + (-1e-8))));

  double max_v = -std::numeric_limits<double>::infinity();
  double min_v = std::numeric_limits<double>::infinity();
  for (double v: all_v) {
    max_v = std::max(max_v, v);
    min_v = std::min(min_v, v);
  }

  double max_z = -std::numeric_limits<double>::infinity();
  for (double z: all_z) {
    max_z = std::max(max_z, z);
  }

  for (double v: all_v) {
    for (double z: all_z) {
      if(z < 0) { //May arise from the boundaries - 1e-8 test value
        continue;
      }
      //TODO(martinmodrak) resolve
      //for (int sign = -1; sign <= 1; sign += 2) {
      int sign = 1; {
        test_single_pair(sign * v, z, debug_output);
      }
    }
  }

  //Test the linear boundaries
  for (double z: all_z) {
      //TODO(martinmodrak) resolve
      //for (int sign = -1; sign <= 1; sign += 2) {
      int sign = 1; {
        if(z < 0) {
          continue;
        }
        double v1 = exp(stan::math::besselk_internal::asymp_v_slope * log(z)
            + stan::math::besselk_internal::asymp_v_intercept);
        double v2 = exp(stan::math::besselk_internal::asymp_z_slope * log(z)
            + stan::math::besselk_internal::asymp_z_intercept);        
        double v3 = z;

        if(v1 + 1 < max_v && v1 - 1 > min_v) {
          test_single_pair(v1 * sign, z, debug_output);
          test_single_pair(v1 * sign + 1, z, debug_output);
          test_single_pair(v1 * sign - 1, z, debug_output);
          test_single_pair(v1 * sign + 1e-8, z, debug_output);
          test_single_pair(v1 * sign - 1e-8, z, debug_output);
        }


        if(v2 + 1 < max_v && v2 - 1 > min_v) {
          test_single_pair(v2 * sign, z, debug_output);
          test_single_pair(v2 * sign + 1, z, debug_output);
          test_single_pair(v2 * sign - 1, z, debug_output);
          test_single_pair(v2 * sign + 1e-8, z, debug_output);
          test_single_pair(v2 * sign - 1e-8, z, debug_output);
        }

        if(v3 + 1 < max_v && v3 - 1 > min_v) {
          test_single_pair(v3 * sign, z, debug_output);
          test_single_pair(v3 * sign + 1, z, debug_output);
          test_single_pair(v3 * sign - 1, z, debug_output);
          test_single_pair(v3 * sign + 1e-8, z, debug_output);
          test_single_pair(v3 * sign - 1e-8, z, debug_output);
        }

      }
  }

  //The final non-linear boundary
  for (double v : all_v) {
    if(v < 0.5) {
      continue;
    }
    double z = exp(
      stan::math::besselk_internal::get_rothwell_log_z_boundary(v));
    if( z < max_z) {
      //for (int sign = -1; sign <= 1; sign += 2) {
      int sign = 1; {
        test_single_pair(v * sign, z, debug_output);
        test_single_pair(v * sign, z + 1e-8, debug_output);
        if(z > 1e-8) {
          test_single_pair(v * sign, z - 1e-8, debug_output);
        }
      }
    }
  }
}

struct fun {
  template <typename T_v, typename T_z>
  inline typename boost::math::tools::promote_args<T_v, T_z>::type operator()(
      const T_v& arg1, const T_z& arg2) const {
    return log_modified_bessel_second_kind_frac(arg1, arg2);
  }
};

TEST(AgradRev, log_modified_bessel_second_kind_frac_input) {
  fun f;
  //TODO(martinmodrak) Test for NaNs
  EXPECT_THROW(log_modified_bessel_second_kind_frac(1.0, -1.0),
               std::domain_error);
  EXPECT_THROW(log_modified_bessel_second_kind_frac(
                   std::numeric_limits<double>::infinity(), 1),
               std::domain_error);
  EXPECT_THROW(log_modified_bessel_second_kind_frac(
                   1.0, std::numeric_limits<double>::infinity()),
               std::domain_error);

  std::for_each(v_to_test.begin(), v_to_test.end(), [](double v) {
    EXPECT_TRUE(std::isinf(log_modified_bessel_second_kind_frac(v, 0)))
        << "Infinity for z = 0";
  });
}

namespace log_modified_bessel_second_kind_internal {
struct TestValue {
  double v;
  double z;
  double value;
  double grad_v;
  double grad_z;
};

const double NaN = std::numeric_limits<double>::quiet_NaN();

// Test values generated by Mathematica with the following code:
// TODO: update code
std::vector<TestValue> testValues = {
  {3.e-8, 4.e-8, 2.84201670979535185474305, 2.96510217769719201562657e-6, -1.45769892709695339800853e6},
  {3.e-8, 2.56000000000000005356176e-6, 2.56429027994492526029667, 1.73343427074306758686969e-6, -30067.887461290865188866},
  {3.e-8, 1., -0.865064398906787767803676, 2.19330054363531404298969e-8, -1.42962539826040201520839},
  {3.e-8, 1.35000000000000008881784, -1.34589936901498513297139, 1.7261923946244755490914e-8, -1.3274527125104544953011},
  {3.e-8, 5.41999999999999992894573, -6.06051393330273661621426, 5.10426034300837985369552e-9, -1.08861450772611768069924},
  {3.e-8, 15.6099999999999994315658, -16.7659313101551834018895, 1.8641684878232514215386e-9, -1.03154771574265619755468},
  {3.e-8, 482., -484.863439772038824074741, NaN, -1.00103680746930587460687},
  {3.e-8, 1823., -1826.52839658419291801406, NaN, -1.000274235583810567911},
  {3.e-8, 2023., -2026.58043884009692333897, NaN, -1.0002471271582294826744},
  {8.41999999999999900488021e-6, 4.e-8, 2.8420167133274605116955, 0.000838967151451003108949667, -1.45769893723170418803865e6},
  {8.41999999999999900488021e-6, 2.56000000000000005356176e-6, 2.56429028199530733153848, 0.000487030965398991002395328, -30067.8875812829477671929},
  {8.41999999999999900488021e-6, 1., -0.865064398880871911355192, 6.15586352577742950777774e-6, -1.4296253982806608200594},
  {8.41999999999999900488021e-6, 1.35000000000000008881784, -1.3458993689945885874859, 4.84484665423425587369002e-6, -1.32745271252275297809618},
  {8.41999999999999900488021e-6, 5.41999999999999992894573, -6.06051393329670546472847, 1.43259573627062151989288e-6, -1.08861450772715032259574},
  {8.41999999999999900488021e-6, 15.6099999999999994315658, -16.765931310152980715939, 5.23209955582370601119238e-7, -1.03154771574279320675799},
  {8.41999999999999900488021e-6, 482., -484.86343977203875060781, NaN, -1.00103680746930602687032},
  {8.41999999999999900488021e-6, 1823., -1826.52839658419289857465, NaN, -1.0002742355838105785715},
  {8.41999999999999900488021e-6, 2023., -2026.58043884009690582093, NaN, -1.0002471271582294913317},
  {-6.09999999999999964472863, 4.e-8, 112.403812264339315911945, -19.4516215238209581871388, -1.52499999999999995039784e8},
  {-6.09999999999999964472863, 2.56000000000000005356176e-6, 87.0346254558449977607083, -15.2927384404613492854914, -2.38281250000025079175974e6},
  {-6.09999999999999964472863, 1., 8.44532282091818906299108, -2.42666715423891555469876, -6.19690225027373298338376},
  {-6.09999999999999964472863, 1.35000000000000008881784, 6.57501751331719058043773, -2.13406764137734523966831, -4.64813931010659429494834},
  {-6.09999999999999964472863, 5.41999999999999992894573, -3.08468943473876445962734, -0.923785030284870535488501, -1.54830908785355380950653},
  {-6.09999999999999964472863, 15.6099999999999994315658, -15.6220878490085913846179, -0.371153595547871386793071, -1.10128219478065585429852},
  {-6.09999999999999964472863, 482., -484.82488065032223745014, NaN, -1.00111672082824118402501},
  {-6.09999999999999964472863, 1823., -1826.51819368630675942101, NaN, -1.00027983080216474078176},
  {-6.09999999999999964472863, 2023., -2026.57124438136435241702, NaN, -1.00025167099125685812975},
  {2.29999999999999982236432, 4.e-8, 40.2343694702022480826207, 18.3275734437563898783141, -5.75000000000000109437233e7},
  {2.29999999999999982236432, 2.56000000000000005356176e-6, 30.6689383784737435033071, 14.1686903603976872316574, -898437.500000984527198059},
  {2.29999999999999982236432, 1., 0.883998066042126683883583, 1.3959577532624389819493, -2.61548383034204942694716},
  {2.29999999999999982236432, 1.35000000000000008881784, 0.0698724824026178363591914, 1.15466073985974779197135, -2.09359608893796148201146},
  {2.29999999999999982236432, 5.41999999999999992894573, -5.6149949587931698486133, 0.383609793640167362775495, -1.16353051125528502748072},
  {2.29999999999999982236432, 15.6099999999999994315658, -16.6018291360965033145665, 0.142476664450821565138028, -1.04172468722824522255987},
  {2.29999999999999982236432, 482., -484.857957910249282063819, NaN, -1.00104816882643669124414},
  {2.29999999999999982236432, 1823., -1826.52694607699165830967, NaN, -1.00027503103600216099106},
  {2.29999999999999982236432, 2023., -2026.57913169905232114098, NaN, -1.00024777313843681031481},
  {5.67999999999999971578291, 4.e-8, 104.250245712652775416954, 19.3738815325878530202873, -1.41999999999999997168077e8},
  {5.67999999999999971578291, 2.56000000000000005356176e-6, 80.6277897991694903352007, 15.2149984492282559291861, -2.21875000000027334682927e6},
  {5.67999999999999971578291, 1., 7.44189018457783185936192, 2.35065060247970231216087, -5.78534450816101479971245},
  {5.67999999999999971578291, 1.35000000000000008881784, 5.69421198082100658426491, 2.05936272033682622078359, -4.34806908333593262305466},
  {5.67999999999999971578291, 5.41999999999999992894573, -3.46173293119732396502802, 0.871361663968736964057826, -1.49454264084043611367054},
  {5.67999999999999971578291, 15.6099999999999994315658, -15.7728086430973974123194, 0.346532150696723918280165, -1.09224849820389573541921},
  {5.67999999999999971578291, 482., -484.830007576268014426277, NaN, -1.00110609557442196797999},
  {5.67999999999999971578291, 1823., -1826.51955030594664373003, NaN, -1.00027908683991656622279},
  {5.67999999999999971578291, 2023., -2026.57246691493537867578, NaN, -1.00025106682506373303291},
  {-5.41999999999999992894573, 4.e-8, 99.2196525513961465520481, -19.3225512173870557836925, -1.3550000000000000274853e8},
  {-5.41999999999999992894573, 2.56000000000000005356176e-6, 76.6785062395863546836706, -15.1636681340274677497587, -2.11718750000028952070762e6},
  {-5.41999999999999992894573, 1., 6.8371712948858468243496, -2.30063135530162348727447, -5.53133187983039472405912},
  {-5.41999999999999992894573, 1.35000000000000008881784, 5.16510315743349232786056, -2.01032848561550191036855, -4.16326788099159662801459},
  {-5.41999999999999992894573, 5.41999999999999992894573, -3.68396561124710998912674, -0.838004123015301362105218, -1.46234669196492661085695},
  {-5.41999999999999992894573, 15.6099999999999994315658, -15.8609147084550996310136, -0.331194933441533874991904, -1.0869467502625218469978},
  {-5.41999999999999992894573, 482., -484.832998187804225842049, NaN, -1.00109989767253446876777},
  {-5.41999999999999992894573, 1823., -1826.52034164013871853704, NaN, -1.000278652876719988467},
  {-5.41999999999999992894573, 2023., -2026.57318003492245783141, NaN, -1.00025071440682731653279},
  {14.0999999999999996447286, 4.e-8, 272.077901016881713170409, 20.3378284175285263730341, -3.52499999999999992644933e8},
  {14.0999999999999996447286, 2.56000000000000005356176e-6, 213.437649541510216138068, 16.1789453341688640405065, -5.50781250000009745590824e6},
  {14.0999999999999996447286, 1., 31.8739840645984492478061, 3.30489529865540415432164, -14.1381079469706087658527},
  {14.0999999999999996447286, 1.35000000000000008881784, 27.6268475105274859184837, 3.00598076524988572313932, -10.4958239727403103838855},
  {14.0999999999999996447286, 5.41999999999999992894573, 7.51427571886373470238786, 1.65339269968628988315073, -2.79961592354477527014457},
  {14.0999999999999996447286, 15.6099999999999994315658, -10.8949146098749955770031, 0.795753495685882691046597, -1.36543716043966008897973},
  {14.0999999999999996447286, 482., -484.657433401394295394366, NaN, -1.0014637042839450891328},
  {14.0999999999999996447286, 1823., -1826.47388355219587910316, NaN, -1.00030413001705301218422},
  {14.0999999999999996447286, 2023., -2026.53131375723788293141, NaN, -1.00027140424879563549114},
  {31.3999999999999985789145, 4.e-8, 631.979362972330866872584, 21.1583333780018165874988, -7.84999999999999965130758e8},
  {31.3999999999999985789145, 2.56000000000000005356176e-6, 501.390434154837121949958, 16.9994502946421464824932, -1.22656250000000412935234e7},
  {31.3999999999999985789145, 1., 97.0914080168895967102955, 4.12421739638303619699692, -31.4164427704724115504232},
  {31.3999999999999985789145, 1.35000000000000008881784, 87.6613625015310809815529, 3.82433503717677952740899, -23.2814518993302365887109},
  {31.3999999999999985789145, 5.41999999999999992894573, 43.7900211838017866523188, 2.44170075172794246214928, -5.88178235665592602070205},
  {31.3999999999999985789145, 15.6099999999999994315658, 8.87393844520969827557005, 1.43604451690894996786073, -2.25287246145100620467963},
  {31.3999999999999985789145, 482., -483.842077310969788090525, NaN, -1.00315214213334820999419},
  {31.3999999999999985789145, 1823., -1826.25805500319940680378, NaN, -1.00042248251925511048073},
  {31.3999999999999985789145, 2023., -2026.33681632922427412565, NaN, -1.00036751894489471114742},
  {105.599999999999994315658, 4.e-8, 2256.38421960664362324849, NaN, -2.63999999999999985808266e9},
  {105.599999999999994315658, 2.56000000000000005356176e-6, 1817.20616600386228097317, NaN, -4.1250000000000009153592e7},
  {105.599999999999994315658, 1., 457.55062754974151614245, NaN, -105.604780004450391139777},
  {105.599999999999994315658, 1.35000000000000008881784, 425.857616828804223271191, NaN, -78.2286751057969761505899},
  {105.599999999999994315658, 5.41999999999999992894573, 279.008711956720925842677, NaN, -19.5092855213431364529549},
  {105.599999999999994315658, 15.6099999999999994315658, 166.792772820735260177488, NaN, -6.8390971194778190334134},
  {105.599999999999994315658, 482., -473.352952414217970790849, NaN, -1.02470775477656278330605},
  {105.599999999999994315658, 1823., -1823.47156900224482325247, NaN, -1.00194965350794486016286},
  {105.599999999999994315658, 2023., -2023.82559973231483522299, NaN, -1.00160793377524982043157},
  {1823., 4.e-8, 44178.2846899810856259221, NaN, -4.5575000000000000000011e10},
  {1823., 2.56000000000000005356176e-6, 36596.6408290164037924755, NaN, -7.12109374999999985803373e8},
  {1823., 1., 13124.5981768656340334956, NaN, -1823.00027442368953081315},
  {1823., 1.35000000000000008881784, 12577.5073919719287699592, NaN, -1350.37074084232818807686},
  {1823., 5.41999999999999992894573, 10043.5496117234203024291, NaN, -336.348350841851722254916},
  {1823., 15.6099999999999994315658, 8115.12178764019730021986, NaN, -116.788396423705148952061},
  {1823., 482., 1830.60129464491256178038, NaN, -3.91219213160082435507174},
  {1823., 1823., -975.068897987261073952656, NaN, -1.4143507189019339860151},
  {1823., 2023., -1250.83175246120609348756, NaN, -1.34625964535750633339147},
};

} //namespace

#define EXPECT_REL_ERROR(a, b) EXPECT_TRUE(check_relative_error(a, b))

TEST(AgradRev, log_modified_bessel_second_kind_frac_double_double) {
  using log_modified_bessel_second_kind_internal::TestValue;
  using log_modified_bessel_second_kind_internal::testValues;
  for(TestValue test : testValues) {
    double f1 = log_modified_bessel_second_kind_frac(test.v, test.z);
    std::stringstream msg;
    msg << std::setprecision(22) << "value at v = " << test.v 
      << ", z = " << test.z;
    stan::test::expect_near_rel(msg.str(), test.value, f1);        
  }
}

TEST(AgradRev, log_modified_bessel_second_kind_frac_double_var) {
  using log_modified_bessel_second_kind_internal::TestValue;
  using log_modified_bessel_second_kind_internal::testValues;
  using stan::test::expect_near_rel;
  for(TestValue test : testValues) {
    AVAR z(test.z);
    try {
      std::stringstream msg_base;
      msg_base << std::setprecision(22) << " at v = " << test.v 
        << ", z = " << test.z;

      AVAR f = log_modified_bessel_second_kind_frac(test.v, z);
      std::stringstream msg_ratio;
      msg_ratio << "value" << msg_base.str();
      expect_near_rel(msg_ratio.str(), test.value, f.val());

      AVEC x = createAVEC(z);
      VEC g;
      f.grad(x, g);

      std::stringstream msg_grad_z;
      msg_grad_z << "grad_z" << msg_base.str();

      expect_near_rel(msg_grad_z.str(), test.grad_z, g[0]);
    } catch (const std::domain_error& err) {
      std::cout << "\nAt v = " << test.v << ", z = " << test.z << ":\n";
      throw;
    }
  }
}

TEST(AgradRev, log_modified_bessel_second_kind_frac_var_double) {
  using log_modified_bessel_second_kind_internal::TestValue;
  using log_modified_bessel_second_kind_internal::testValues;
  using stan::math::is_nan;
  using stan::test::expect_near_rel;
  for(TestValue test : testValues) {
    try {
      std::stringstream msg_base;
      msg_base << std::setprecision(22) << " at v = " << test.v 
        << ", z = " << test.z;


      AVAR v(test.v);
      AVAR f = log_modified_bessel_second_kind_frac(v, test.z);
      std::stringstream msg_ratio;
      msg_ratio << "value" << msg_base.str();
      expect_near_rel(msg_ratio.str(), test.value, f.val());

      AVEC x = createAVEC(v);
      VEC g;
      f.grad(x, g);
      std::stringstream msg_grad_v;
      msg_grad_v << "grad_v" << msg_base.str();

      if(!is_nan(test.grad_v)) {
        expect_near_rel(msg_grad_v.str(), test.grad_v, g[0]);
      }
    } catch (const std::domain_error& err) {
      std::cout << "\nAt v = " << test.v << ", z = " << test.z << ":\n";
      throw;
    }
  }
}

TEST(AgradRev, log_modified_bessel_second_kind_frac_var_var) {
  using log_modified_bessel_second_kind_internal::TestValue;
  using log_modified_bessel_second_kind_internal::testValues;
  using stan::math::is_nan;
  using stan::test::expect_near_rel;
  for(TestValue test : testValues) {
    try {
      std::stringstream msg_base;
      msg_base << std::setprecision(22) << " at v = " << test.v 
        << ", z = " << test.z << ", method = " 
        << computation_type_to_string(stan::math::besselk_internal::choose_computation_type(test.v, test.z));

      AVAR v(test.v);
      AVAR z(test.z);
      AVAR f = log_modified_bessel_second_kind_frac(v, z);
      std::stringstream msg_ratio;
      msg_ratio << "value" << msg_base.str();
      expect_near_rel(msg_ratio.str(), test.value, f.val());

      AVEC x = createAVEC(v, z);
      VEC g;
      f.grad(x, g);
      std::stringstream msg_grad_v;
      msg_grad_v << "grad_v" << msg_base.str();

      if(!is_nan(test.grad_v)) {
        expect_near_rel(msg_grad_v.str(), test.grad_v, g[0]);
      }

      std::stringstream msg_grad_z;
      msg_grad_z << "grad_z" << msg_base.str();

      expect_near_rel(msg_grad_z.str(), test.grad_z, g[1]);
    } catch (const std::domain_error& err) {
      std::cout << "\nAt v = " << test.v << ", z = " << test.z << ":\n";
      throw;
    }
  }
}
