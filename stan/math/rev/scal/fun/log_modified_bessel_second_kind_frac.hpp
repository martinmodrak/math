#ifndef STAN_MATH_REV_SCAL_FUN_LOG_MODIFIED_BESSEL_SECOND_KIND_FRAC_HPP
#define STAN_MATH_REV_SCAL_FUN_LOG_MODIFIED_BESSEL_SECOND_KIND_FRAC_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/to_var.hpp>

#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>

#include <stan/math/rev/scal/fun/cos.hpp>

#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/prim/scal/fun/exp.hpp>
#include <stan/math/prim/scal/fun/log_diff_exp.hpp>
#include <stan/math/rev/arr/fun/log_sum_exp.hpp>

#include <stan/math/rev/scal/meta/is_var.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

#include <stan/math/prim/scal/err/domain_error.hpp>

#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/tools/numerical_differentiation.hpp>
#include <limits>
#include <algorithm>

// Combining multiple approaches, documented at relevant sections
// Due to symmetry around v, the code assumes all v's positive except for
// top-level calls which pass the absolute value.

namespace stan {
namespace math {

template <typename _Tp>
inline int is_inf(const std::complex<_Tp> &c) {
  return is_inf(c.imag()) || is_inf(c.real());
}

template <typename _Tp>
inline bool non_zero(const std::complex<_Tp> &c) {
  return abs(c) > 0;
}

inline bool non_zero(const double &c) { return c > 0; }

namespace besselk_internal {

////////////////////////////////////////////////////////////////
//                    FORMULAE                                //
////////////////////////////////////////////////////////////////

// The formulas that contain integrals are split into a function representing
// the integral body and "lead" - the logarithm of the term before the integral
// The function object also references the integration method, as some integrate
// From 0 to 1 and others from 0 to infinity.

// The formulas for Rothwell approach and code for small z are based on
// https://github.com/stan-dev/stan/wiki/Stan-Development-Meeting-Agenda/0ca4e1be9f7fc800658bfbd97331e800a4f50011
// Which is in turn based on Equation 26 of Rothwell: Computation of the
// logarithm of Bessel functions of complex argument and fractional order
// https://scholar.google.com/scholar?cluster=2908870453394922596&hl=en&as_sdt=5,33&sciodt=0,33

template <typename T_v, typename T_z, typename T_u>
class inner_integral_rothwell {
 private:
  T_v v;
  T_z z;

 public:
  typedef typename boost::math::tools::promote_args<T_v, T_z, T_u>::type T_Ret;

  inner_integral_rothwell(const T_v &v, const T_z &z) : v(v), z(z) {}

  inline T_Ret operator()(const T_u &u) const {
    using std::exp;
    using std::pow;

    auto v_mhalf = v - 0.5;
    auto neg2v_m1 = -2.0 * v - 1.0;
    auto beta = 16.0 / (2.0 * v + 1.0);

    T_Ret value;
    T_Ret uB = pow(u, beta);
    T_Ret first
        = beta * exp(-uB) * pow(2.0 * z + uB, v_mhalf) * boost::math::pow<7>(u);
    T_Ret second = exp(-1.0 / u);
    if (non_zero(second)) {
      //    if (abs(second) > 0) {
      second = second * pow(u, neg2v_m1);
      if (is_inf(second)) {
        second = exp(-1.0 / u + neg2v_m1 * log(u));
      }
      second = second * pow(2.0 * z * u + 1.0, v_mhalf);
    }
    value = first + second;

    return value;
  }

  T_Ret integrate() {
    typedef T_u value_type;
    value_type tolerance
        = std::sqrt(std::numeric_limits<value_type>::epsilon());

    value_type error;
    value_type L1;
    size_t levels;

    boost::math::quadrature::tanh_sinh<value_type> integrator;
    T_Ret value = integrator.integrate(*this, 0.0, 1.0, tolerance, &error, &L1,
                                       &levels);
    if (error > 1e-6 * L1) {
      domain_error("inner_integral_rothwell",
                   "error estimate of integral / L1 ", error / L1, "",
                   "is larger than 1e-6");
    }

    return value;
  }
};

template <typename T_v, typename T_z>
typename boost::math::tools::promote_args<T_v, T_z>::type compute_lead_rothwell(
    const T_v &v, const T_z &z) {
  typedef typename boost::math::tools::promote_args<T_v, T_z>::type T_Ret;

  using std::exp;
  using std::lgamma;
  using std::log;
  using std::pow;

  const T_Ret lead = 0.5 * log(pi()) - lgamma(v + 0.5) - v * log(2 * z) - z;
  if (is_inf(lead))
    return -z + 0.5 * log(0.5 * pi() / z);

  return lead;
}

template <typename T_v, typename T_z>
typename boost::math::tools::promote_args<T_v, T_z>::type
compute_log_integral_rothwell(const T_v &v, const T_z &z) {
  typedef typename boost::math::tools::promote_args<T_v, T_z>::type T_Ret;

  inner_integral_rothwell<T_v, T_z, double> f(v, z);
  return log(f.integrate());
}

template <>
var compute_log_integral_rothwell(const var &v, const double &z) {
  double value = compute_log_integral_rothwell(value_of(v), z);
  typedef std::complex<double> Complex;
  auto complex_func
      = [z](const Complex &v) { return compute_log_integral_rothwell(v, z); };

  double d_dv = boost::math::tools::complex_step_derivative(
      complex_func, stan::math::value_of(v));

  return var(new precomp_v_vari(value, v.vi_, d_dv));
}

template <typename T_v, typename T_z>
typename boost::math::tools::promote_args<T_v, T_z>::type compute_rothwell(
    const T_v &v, const T_z &z) {
  typedef typename boost::math::tools::promote_args<T_v, T_z>::type T_Ret;

  T_v lead = compute_lead_rothwell(v, z);
  T_v log_integral = compute_log_integral_rothwell(v, z);
  return lead + log_integral;
}

// Formula 1.10 of
// Temme, Journal of Computational Physics, vol 19, 324 (1975)
// https://doi.org/10.1016/0021-9991(75)90082-0
// Also found on wiki at
// https://en.wikipedia.org/w/index.php?title=Bessel_function&oldid=888330504#Asymptotic_forms
template <typename T_v>
T_v asymptotic_large_v(const T_v &v, const double &z) {
  using std::lgamma;
  using std::log;

  // return 0.5 * (log(stan::math::pi()) - log(2) - log(v)) - v * (log(z) -
  // log(2) - log(v));
  return lgamma(v) - stan::math::LOG_2 + v * (stan::math::LOG_2 - log(z));
}

// Formula 10.40.2 from https://dlmf.nist.gov/10.40
template <typename T_v>
T_v asymptotic_large_z(const T_v &v, const double &z) {
  using std::log;
  using std::pow;

  //const int max_terms = 1000;
  const int max_terms = std::min(1000, static_cast<int>(floor(value_of(v) + 0.5)));

  double log_z = log(z);
  T_v base = 0.5 * (log(pi()) - log(2) - log_z) - z; 

  std::vector<T_v> log_sum_terms;
  log_sum_terms.reserve(max_terms);
  log_sum_terms.push_back(0);
  
  T_v log_v_sq_4 = 2 * (log(v) + log(2));
  T_v log_a_k_z_k = 0;  
  for (int k = 1; k < max_terms; k++) {
    log_a_k_z_k += log_diff_exp(log_v_sq_4, 2 * log(2 * k - 1)) - log(k) - log(z) - log(8);
    log_sum_terms.push_back(log_a_k_z_k);
    //TODO figure out a good stopping criterion
    // if(log_a_k_z_k < 30) {
    //   break;
    // }
  }
  return base + log_sum_exp(log_sum_terms);


  // T_v series_sum(1);
  // T_v a_k_z_k(1);
  // const T_v v_squared_4 = v * v * 4;
  
  // for (int k = 1; k < max_terms; k++) {
  //   a_k_z_k *= (v_squared_4 - boost::math::pow<2>(2 * k - 1)) / (k * z * 8);
  //   series_sum += a_k_z_k;
  //   if(fabs(a_k_z_k) < 1e-8) {
  //     break;
  //   }
  // }
  // return base + log(series_sum);
}

template <typename T_v>
inline T_v log_integral_gamma_func(const T_v &v, const double &z, const double &t, const double &log_offset = 0) {
    return (v - 1.0) * std::log(t) - 0.5 * z * (t + 1.0 / t) - log_offset;
}

template <typename T_v>
inline T_v log_integral_gamma_func_max_t(const T_v &v_, const double &z) {
    return (std::sqrt(v_ * v_ - 2.0*v_ + z*z + 1.0) + v_ - 1.0) / z;
}

template <typename T_v>
T_v integral_gamma(const T_v &v, const double &z) {
  using std::log;
  using std::pow;

  //auto integrand = [&](T_v t) { return std::pow(t, v - 1) * std::exp(-0.5 * z * (t + 1 / t)); };
  using std::cosh;  using std::exp;
  double value_at_max = std::abs(log_integral_gamma_func(v, z,
  //The abs is unnecessary, just to work around issues with using complex
    std::abs(log_integral_gamma_func_max_t(v, z))));

  auto f = [&, v, z](double t)
  {
    T_v log_value = log_integral_gamma_func(v, z, t, value_at_max);
    T_v value = 0.5 * std::exp(log_value);
    if(!std::isfinite(std::abs(value))) {
      std::cout << v << " " << z << " " << t << std::endl;
    }
    return value;
  };
  boost::math::quadrature::exp_sinh<double> integrator;
  return std::log(integrator.integrate(f)) + value_at_max;  
}

template <>
var integral_gamma(const var &v, const double &z) {
  double value = integral_gamma(value_of(v), z);
  typedef std::complex<double> Complex;
  auto complex_func
      = [z](const Complex &complex_v) { return integral_gamma(complex_v, z); };

  double d_dv = boost::math::tools::complex_step_derivative(
      complex_func, stan::math::value_of(v));

  return var(new precomp_v_vari(value, v.vi_, d_dv)); 
}

////////////////////////////////////////////////////////////////
//                    CHOOSING AMONG FORMULAE                 //
////////////////////////////////////////////////////////////////

// The code to choose computation method is separate, because it is
// referenced from the test code.
enum class ComputationType { Rothwell, Asymp_v, Asymp_z, IntegralGamma };

const double rothwell_max_v = 50;
const double rothwell_max_log_z_over_v = 300;

const double gamma_max_z = 800;
const double gamma_max_v = 5;
const double gamma_low_z = 0.01;
const double gamma_low_v = 0.001;

const double small_z_factor = 10;
const double small_z_min_v = 15;

const double asymp_v_slope = 0.8;
const double asymp_v_intercept = 4.8;

const double asymp_z_slope = 1;
const double asymp_z_intercept = 0.5;

inline ComputationType choose_computation_type(const double &v,
                                               const double &z) {
  using std::fabs;
  using std::pow;
  const double v_ = fabs(v);
  const double rothwell_log_z_boundary
      = rothwell_max_log_z_over_v / (v_ - 0.5) - log(2);

  const double log_z = log(z);
  const double log_v = log(v_);

  if (v_ >= rothwell_max_v &&  
      (log_v > asymp_v_slope * log_z + asymp_v_intercept)) {
    return ComputationType::Asymp_v;
  } else if(z >= gamma_max_z && 
    (log_v < asymp_z_slope * log_z + asymp_z_intercept)) {
    return ComputationType::Asymp_z;
  } else if ((v_ > gamma_low_v || z > gamma_low_z) && 
      (log_v < asymp_v_slope * log_z + asymp_v_intercept || v < gamma_max_v))  {
      //  ( (z < gamma_max_z && v < gamma_max_v) ||
      //    (log(v_) < 0.8 * log(z) + 4.8 && log(v_) > 0.9 * log(z) + 1.5) ||
      //    (z < gamma_max_z && log_integral_gamma_func(v_, z, log_integral_gamma_func_max_t(v_, z)) < gamma_max_log_max )
      //  ) ) {
    return ComputationType::IntegralGamma;
  } else if (v_ < rothwell_max_v
             && (v_ <= 0.5 || log(z) < rothwell_log_z_boundary)) {
    return ComputationType::Rothwell;
  } else {
    return ComputationType::IntegralGamma;
  } 

  //return ComputationType::IntegralGamma;
}

////////////////////////////////////////////////////////////////
//                    UTILITY FUNCTIONS                       //
////////////////////////////////////////////////////////////////

void check_params(const double &v, const double &z) {
  const char *function = "log_modified_bessel_second_kind_frac";
  if (!std::isfinite(v)) {
    stan::math::domain_error(function, "v must be finite", v, "");
  }
  if (!std::isfinite(z)) {
    stan::math::domain_error(function, "z must be finite", z, "");
  }
  if (z < 0) {
    stan::math::domain_error(function, "z is negative", z, "");
  }
}

}  // namespace besselk_internal

////////////////////////////////////////////////////////////////
//                    TOP LEVEL FUNCTIONS                     //
////////////////////////////////////////////////////////////////

template <typename T_v>
T_v log_modified_bessel_second_kind_frac(const T_v &v, const double &z) {
  using besselk_internal::ComputationType;
  using besselk_internal::asymptotic_large_v;
  using besselk_internal::asymptotic_large_z;
  using besselk_internal::integral_gamma;
  using besselk_internal::check_params;
  using besselk_internal::choose_computation_type;
  using besselk_internal::compute_rothwell;
  using std::fabs;
  using std::pow;

  check_params(value_of(v), value_of(z));

  if (z == 0) {
    return std::numeric_limits<double>::infinity();
  }

  T_v v_ = fabs(v);
  switch (choose_computation_type(value_of(v_), value_of(z))) {
    case ComputationType::Rothwell: {
      return compute_rothwell(v_, z);
    }
    case ComputationType::Asymp_v: {
      return asymptotic_large_v(v_, z);
    }
    case ComputationType::Asymp_z: {
      return asymptotic_large_z(v_, z);
    }
    case ComputationType::IntegralGamma: {
      return integral_gamma(v_, z);
    }
    default: {
      stan::math::domain_error("log_modified_bessel_second_kind_frac",
                               "Invalid computation type ", 0, "");
      return asymptotic_large_v(v_, z);
    }
  }
}

template <typename T_v>
var log_modified_bessel_second_kind_frac(const T_v &v, const var &z) {
  using stan::is_var;

  T_v value = log_modified_bessel_second_kind_frac(v, z.val());

  double value_vm1
      = log_modified_bessel_second_kind_frac(value_of(v) - 1, z.val());
  double gradient_dz
      = -std::exp(value_vm1 - value_of(value)) - value_of(v) / z.val();

  // Compute gradient using boost, avoiding log sacale, seems to be less stable
  // double gradient_dz =
  //   -boost::math::cyl_bessel_k(value_of(v) - 1, value_of(z)) /
  //    boost::math::cyl_bessel_k(value_of(v), value_of(z))
  //    - value_of(v) / value_of(z);

  if (is_var<T_v>::value) {
    // A trick to combine the autodiff gradient with precomputed_gradients
    return var(new precomp_vv_vari(value_of(value), to_var(value).vi_, z.vi_, 1,
                                   gradient_dz));
  } else {
    return var(new precomp_v_vari(value_of(value), z.vi_, gradient_dz));
  }
}

}  // namespace math
}  // namespace stan
#endif
