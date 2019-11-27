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
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>

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

  const int max_terms = 10;

  T_v series_sum(1);
  T_v a_k_z_k(1);

  double log_z = log(z);
  T_v v_squared_4 = v * v * 4;

  for (int k = 1; k < max_terms; k++) {
    a_k_z_k *= (v_squared_4 - boost::math::pow<2>(2 * k - 1)) / (k * z * 8);
    series_sum += a_k_z_k;
    if (fabs(a_k_z_k) < 1e-8) {
      break;
    }
  }

  return 0.5 * (log(pi()) - log(2) - log(z)) - z + log(series_sum);
}

template <typename T_v>
inline T_v log_integral_gamma_func(const T_v &v, const double &z, const double &t) {
    return (v - 1.0) * std::log(t) - 0.5 * z * (t + 1.0 / t);
}

template <typename T_v>
T_v integral_gamma(const T_v &v, const double &z) {
  using std::log;
  using std::pow;

  //auto integrand = [&](T_v t) { return std::pow(t, v - 1) * std::exp(-0.5 * z * (t + 1 / t)); };
  using std::cosh;  using std::exp;
  auto f = [&, v, z](double t)
  {
    T_v log_value = log_integral_gamma_func(v, z, t);
    T_v value = 0.5 * std::exp(log_value);
    if(!std::isfinite(std::abs(value))) {
      std::cout << v << " " << z << " " << t << std::endl;
    }
    return value;
  };
  boost::math::quadrature::exp_sinh<double> integrator;
  return std::log(integrator.integrate(f));
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

//Formula 24 of Rothwell
template <typename T_v>
inline T_v log_integral_rothwell_24_func(const T_v &v_mhalf, const double &z, const long double &t) {
  return -t + std::log(t) * v_mhalf + std::log1p(t / (2 * z)) * v_mhalf ;
}

template <typename T_v>
T_v integral_rothwell_24(const T_v &v, const double &z) {
  using std::log;

  using std::exp;
  T_v v_mhalf = v - 0.5;
  auto f = [&, v_mhalf, z](long double t)
  {
    T_v log_value = log_integral_rothwell_24_func(v_mhalf, z, t);
    T_v value = std::exp(log_value);
    if(!std::isfinite(std::abs(value))) {
      std::cout << v << " " << z << " " << t << std::endl;
    }
    return value;
  };
  boost::math::quadrature::exp_sinh<long double> integrator;
  T_v inner_integral = std::log(integrator.integrate(f));
  return 0.5 * (log(pi()) - log(2) - log(z)) -z -std::lgamma(v_mhalf) + inner_integral;
}

//Ugly hach to have complex step working. Should be removed and complex
//step replaced with autodiff
template <>
var integral_rothwell_24(const var &v, const double &z) {
  double value = integral_rothwell_24(value_of(v), z);
  typedef std::complex<long double> Complex;
  auto complex_func
    = [&z](const Complex &complex_v) {
        using std::log;

        using std::exp;
        const Complex v_mhalf = complex_v - Complex(0.5, 0);
        auto f = [&](long double t)
        {
          Complex log_value = log_integral_rothwell_24_func(v_mhalf, z, t);
          Complex value = std::exp(log_value);
          if(!std::isfinite(std::abs(value))) {
            std::cout << complex_v << " " << z << " " << t << std::endl;
          }
          return value;
        };
        boost::math::quadrature::exp_sinh<long double> integrator;
        Complex inner_integral = std::log(integrator.integrate(f));
        return Complex(0.5 * (log(pi()) - log(2) - log(z)) -z -std::lgamma(complex_v.real() + 0.5) + inner_integral.real(),
          - complex_v.imag() * boost::math::digamma(complex_v.real() + 0.5)
          + inner_integral.imag());
      };

    double d_dv = boost::math::tools::complex_step_derivative(
      complex_func, static_cast<long double>(stan::math::value_of(v)));

    return var(new precomp_v_vari(value, v.vi_, d_dv));
}

////////////////////////////////////////////////////////////////
//                    CHOOSING AMONG FORMULAE                 //
////////////////////////////////////////////////////////////////

// The code to choose computation method is separate, because it is
// referenced from the test code.
enum class ComputationType { Rothwell, Asymp_v, Asymp_z, IntegralGamma, Rothwell_24 };

const double rothwell_max_v = 50;
const double rothwell_max_log_z_over_v = 300;

const double gamma_max_z = 800;
const double gamma_max_log_max = 50;
const double gamma_low_z = 0.01;
const double gamma_low_v = 0.001;

const double rothwell_24_max_log_max = 300;

const double small_z_factor = 10;
const double small_z_min_v = 15;

inline ComputationType choose_computation_type(const double &v,
                                               const double &z) {
  using std::fabs;
  using std::pow;
  const double v_ = fabs(v);
  const double rothwell_log_z_boundary
      = rothwell_max_log_z_over_v / (v_ - 0.5) - log(2);

  const double gamma_maximum_t = (std::sqrt(v_ * v_ - 2*v_ + z*z + 1) + v_ - 1) / z;
  const double rothwell_24_maximum_t = 0.5 * (std::sqrt(4 * v_ * v_ - 4 * v_ + 4 * z * z + 1) + 2 * v_ - 2 * z - 1);

  if (v_ >= small_z_min_v && z * small_z_factor < sqrt(v_ + 1)) {
    return ComputationType::Asymp_v;
  } else if (z < gamma_max_z &&
      (v_ > gamma_low_v || z > gamma_low_z) &&
      log_integral_gamma_func(v_, z, gamma_maximum_t) < gamma_max_log_max) {
           return ComputationType::IntegralGamma;
  } else if (z < gamma_max_z &&
    (v_ > 0.5 || z > gamma_low_z) &&
    log_integral_rothwell_24_func(v_, z, rothwell_24_maximum_t) < rothwell_24_max_log_max) {
          return ComputationType::Rothwell_24;
  } else if (v_ < rothwell_max_v
             && (v_ <= 0.5 || log(z) < rothwell_log_z_boundary)) {
    return ComputationType::Rothwell;
  } else if (v_ > z) {
    return ComputationType::Asymp_v;
  } else {
    return ComputationType::Asymp_z;
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
  using besselk_internal::integral_rothwell_24;
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
    case ComputationType::Rothwell_24: {
      return integral_rothwell_24(v_, z);
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
