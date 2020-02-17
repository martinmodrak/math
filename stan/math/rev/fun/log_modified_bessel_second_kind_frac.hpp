#ifndef STAN_MATH_REV_FUN_LOG_MODIFIED_BESSEL_SECOND_KIND_FRAC_HPP
#define STAN_MATH_REV_FUN_LOG_MODIFIED_BESSEL_SECOND_KIND_FRAC_HPP

#include <stan/math/prim/err/domain_error.hpp>
#include <stan/math/prim/err/is_scal_infinite.hpp>
#include <stan/math/prim/meta/return_type.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/log_diff_exp.hpp>
#include <stan/math/prim/fun/log1p_exp.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/fun/log_modified_bessel_first_kind.hpp>
#include <test/unit/math/relative_tolerance.hpp>


#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/cos.hpp>
#include <stan/math/rev/fun/digamma.hpp>
#include <stan/math/rev/fun/exp.hpp>
#include <stan/math/rev/fun/fabs.hpp>
#include <stan/math/rev/fun/log_sum_exp.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/rev/fun/to_var.hpp>
#include <stan/math/rev/fun/pow.hpp>
#include <stan/math/rev/meta/is_var.hpp>

#include <boost/math/differentiation/finite_difference.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/tools/roots.hpp>
#include <tuple> // for std::tuple and std::make_tuple.
#include <limits>
#include <algorithm>

// Combining multiple approaches, documented at relevant sections
// Due to symmetry around v, the code assumes all v's positive except for
// top-level calls which pass the absolute value.

namespace stan {
namespace math {

template <typename T_v>
T_v log_modified_bessel_second_kind_frac(const T_v &v, const double &z);

namespace besselk_internal {

////////////////////////////////////////////////////////////////
//                    FORMULAE                                //
////////////////////////////////////////////////////////////////

/*
The formulas that contain integrals are split into a function representing
the integral body and "lead" - the logarithm of the term before the integral
The function object also references the integration method, as some integrate
From 0 to 1 and others from 0 to infinity.

The formulas for Rothwell approach and code for small z are based on
https://github.com/stan-dev/stan/wiki/Stan-Development-Meeting-Agenda/0ca4e1be9f7fc800658bfbd97331e800a4f50011
Which is in turn based on Equation 26 of Rothwell: Computation of the
logarithm of Bessel functions of complex argument and fractional order
https://scholar.google.com/scholar?cluster=2908870453394922596&hl=en&as_sdt=5,33&sciodt=0,33
*/

//TODO(martinmodrak) Docs for the classes and internal functions
template <typename T_v, typename T_z, typename T_u>
class inner_integral_rothwell {
 private:
  T_v v;
  T_z z;

 public:
  typedef typename stan::return_type<T_v, T_z, T_u>::type T_Ret;

  inner_integral_rothwell(const T_v &v, const T_z &z) : v(v), z(z) {}

  inline T_Ret operator()(const T_u &u) const {
    using std::exp;
    using std::pow;

    constexpr unsigned int n = 8;

    const auto v_mhalf = v - 0.5;
    const auto neg2v_m1 = -2.0 * v - 1.0;
    const auto beta = static_cast<double>(2 * n) / (2.0 * v + 1.0);

    T_Ret uB = pow(u, beta);
    T_Ret first
        = beta * exp(-uB) * pow(2.0 * z + uB, v_mhalf) * boost::math::pow<n - 1>(u);
    T_u inv_u = -1.0 / u; //Not using inv because T_u is either double or complex
    T_Ret second = exp(inv_u);
    if (abs(second) > 0)  {
      second = second * pow(u, neg2v_m1);
      if (is_scal_infinite(second)) {
        second = exp(inv_u + neg2v_m1 * log(u));
      }
      second = second * pow(2.0 * z * u + 1.0, v_mhalf);
    }

    return first + second;
  }

  T_Ret integrate() {
    //TODO(martinmodrak) use T_u directly once the code settles
    typedef T_u value_type;
    value_type tolerance
        = std::sqrt(std::numeric_limits<value_type>::epsilon());

    value_type error;
    value_type L1;
    size_t levels;

    //TODO(martinmodrak) check if new version of integrate_1d is not enough
    boost::math::quadrature::tanh_sinh<value_type> integrator;
    T_Ret value = integrator.integrate(*this, 0.0, 1.0, tolerance, &error, &L1,
                                       &levels);
    if (error > 1e-7 * L1) {
      domain_error("inner_integral_rothwell",
                   "error estimate of integral / L1 ", error / L1, "",
                   "is larger than 1e-8");
    }

    return value;
  }
};

template <typename T_v, typename T_z>
typename stan::return_type<T_v, T_z>::type compute_lead_rothwell(
    const T_v &v, const T_z &z) {
  typedef typename stan::return_type<T_v, T_z>::type T_Ret;

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
typename stan::return_type<T_v, T_z>::type
compute_log_integral_rothwell(const T_v &v, const T_z &z) {
  typedef typename stan::return_type<T_v, T_z>::type T_Ret;

  inner_integral_rothwell<T_v, T_z, double> f(v, z);
  return log(f.integrate());
}

template <>
var compute_log_integral_rothwell(const var &v, const double &z) {
  double value = compute_log_integral_rothwell(value_of(v), z);
  typedef std::complex<double> Complex;
  auto complex_func
      = [z](const Complex &v) { return compute_log_integral_rothwell(v, z); };

  double d_dv = boost::math::differentiation::complex_step_derivative(
      complex_func, stan::math::value_of(v));

  return var(new precomp_v_vari(value, v.vi_, d_dv));
}

template <typename T_v, typename T_z>
typename stan::return_type<T_v, T_z>::type compute_rothwell(
    const T_v &v, const T_z &z) {
  typedef typename stan::return_type<T_v, T_z>::type T_Ret;

  T_v lead = compute_lead_rothwell(v, z);
  T_v log_integral = compute_log_integral_rothwell(v, z);
  return lead + log_integral;
}

/* 
 Formula 1.10 of
 Temme, Journal of Computational Physics, vol 19, 324 (1975)
 https://doi.org/10.1016/0021-9991(75)90082-0
 Also found on wiki at
 https://en.wikipedia.org/w/index.php?title=Bessel_function&oldid=888330504#Asymptotic_forms
 */
template <typename T_v>
T_v asymptotic_large_v(const T_v &v, const double &z) {
  using stan::math::lgamma;
  using stan::math::log;

  // return 0.5 * (log(stan::math::pi()) - log(2) - log(v)) - v * (log(z) -
  // log(2) - log(v));
  return lgamma(v) - stan::math::LOG_TWO + v * (stan::math::LOG_TWO - log(z));
}

// Formula 10.40.2 from https://dlmf.nist.gov/10.40
template <typename T_v>
T_v asymptotic_large_z(const T_v &v, const double &z) {
  using std::log;
  using std::pow;

  const double log_z = log(z);
  const double base = 0.5 * (log(pi()) - log(2) - log_z) - z; 

  const int max_terms = 50;
  T_v series_sum(1);
  T_v a_k_z_k(1);
  const T_v v_squared_4 = v * v * 4;
  
  for (int k = 1; k < max_terms; k++) {
    a_k_z_k *= (v_squared_4 - boost::math::pow<2>(2 * k - 1)) / (k * z * 8);
    series_sum += a_k_z_k;
    if(fabs(a_k_z_k) < 1e-8) {
      break;
    }
  }
  return base + log(series_sum);
}

// Formula 10.40.2 from https://dlmf.nist.gov/10.40, log formulation
template <typename T_v>
T_v asymptotic_large_z_log(const T_v &v, const double &z) {
  using std::log;
  using std::pow;

  const double log_z = log(z);
  const double base = 0.5 * (log(pi()) - log(2) - log_z) - z; 

  //Choosing max terms to avoid negative values
   const int max_terms = std::min(1000, static_cast<int>(floor(value_of(v) + 0.5)));


  std::vector<T_v> log_sum_terms;
  log_sum_terms.reserve(max_terms);
  log_sum_terms.push_back(0);
  
  T_v log_v_sq_4 = 2 * (log(v) + log(2));
  T_v log_a_k_z_k = 0;  
  for (int k = 1; k < max_terms; k++) {
    log_a_k_z_k += log_diff_exp(log_v_sq_4, 2 * log(2 * k - 1)) - log(k) - log(z) - log(8);
    log_sum_terms.push_back(log_a_k_z_k);
    //TODO(martinmodrak) figure out a good stopping criterion
    // if(log_a_k_z_k < 30) {
    //   break;
    // }
  }
  return base + log_sum_exp(log_sum_terms);
}

// Connection formula 10.27.4 https://dlmf.nist.gov/10.27
template <typename T_v>
T_v first_kind(const T_v &v, const double &z) {
  return -stan::math::LOG_TWO + stan::math::LOG_PI - log(sin(v * stan::math::pi())) +
    log_diff_exp(log_modified_bessel_first_kind(-v, z), log_modified_bessel_first_kind(v, z));
}

template <typename T_v>
T_v recurrence_formula_small_v_small_z(const T_v &v, const double &z) {
    T_v log_K_vm1
        = log_modified_bessel_second_kind_frac(v - 1, z);

    T_v log_K_vp1
        = log_modified_bessel_second_kind_frac(v + 1, z);

  return log(z) - stan::math::LOG_TWO - log(v)
                         + log_diff_exp(log_K_vp1, log_K_vm1);
}

// Formula 10.30.2 https://dlmf.nist.gov/10.30
template <typename T_v>
T_v asymp_small_z(const T_v &v, const double &z) {
  return -stan::math::LOG_TWO + lgamma(v) - v * (-stan::math::LOG_TWO + log(z));
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

  double d_dv = boost::math::differentiation::complex_step_derivative(
      complex_func, stan::math::value_of(v));

  return var(new precomp_v_vari(value, v.vi_, d_dv)); 
}

//Using trapezoidal rule from https://arxiv.org/pdf/1209.1547.pdf

template <typename T>
inline T logcosh(const T& x) {
  //By bgoodri
  return x + log1p_exp(-2 * x) - log(2);
}

template <typename T_v>
inline T_v cosh_integral(const T_v &v, const double &z, const double &x) {
  return logcosh(v * x) - z * cosh(x);
}

template <typename T_v>
T_v trapezoid_cosh(const T_v &v, const double &z) {
  const double& approximate_maximum = std::asinh(value_of(v) / z);

   boost::uintmax_t maxit = 20;
   using boost::math::tools::newton_raphson_iterate;

  double v_ = value_of(v);

  // double ub = 1.0;
  
  // while ( -z * sinh(ub) + v_ * tanh(ub * v_) > 0 ) {
  //    ub *= 2.0;
  // }

  // auto arg_max = newton_raphson_iterate(
  //   [&](double t){
  //     auto sinh_t = sinh(t);
  //     auto v_t = v_ * t;
  //     auto tanh_vt = tanh(v_t);
  //     auto cosh_vt = cosh(v_t);
  //     auto z_sinh_t = z * sinh_t;
  //     auto value = v_ * tanh_vt - z_sinh_t;
  //     auto ratio = v_ / cosh_vt;
  //     auto ratio2 = ratio * ratio;
  //     auto D1 = ratio2 - z * cosh(t);
  //     return std::make_tuple(value, D1);
  //   }, approximate_maximum, ub > 1.0 ? 0.5 * ub : 0, ub, 8, maxit);
  
  // const double log2 = log(2.0);
  // // log(cosh(x)) == x + log1p(exp(-2 * x)) - log(2)
  // auto value_at_maximum = logcosh(v * arg_max)
  //            - z * cosh(arg_max);

  constexpr int max_doublings = 13;
  const stan::test::relative_tolerance tolerance = 1e-10;
  const int max_terms = boost::math::pow<max_doublings>(2);

  std::vector<T_v> terms;
  terms.reserve(max_terms);

  double h = approximate_maximum;
  double trapezoid_ub = approximate_maximum;
  terms.push_back(cosh_integral(v, z, 0));
  double last_value = value_of(terms[0]) + std::log(h);

  bool doubling_ub_helps = true;
  bool halving_h_helps = true;
  for(int doubling = 0; doubling < max_doublings; doubling++) {
    bool should_double_ub = !halving_h_helps || (doubling % 2) == 0;
    if(doubling_ub_helps && should_double_ub) {
      size_t n_previous_terms = terms.size();
      for(size_t n = 0; n < n_previous_terms; ++n) {
        const double x = trapezoid_ub + n * h;
        const T_v last_term = cosh_integral(v, z, x);
        terms.push_back(last_term);
      }
      trapezoid_ub *= 2;
      double new_value = value_of(log_sum_exp(terms)) + std::log(h);
      if(fabs(new_value - last_value) < tolerance.inexact(new_value, last_value)) {
        doubling_ub_helps = false;
      }      
      last_value = new_value;
    } else if (halving_h_helps) {
      size_t n_previous_terms = terms.size();
      for(size_t n = 0; n < n_previous_terms; ++n) {
        const double x = ((n * 2) + 1) * h * 0.5;
        const T_v last_term = cosh_integral(v, z, x);
        terms.push_back(last_term);
      }
      h *= 0.5;
      double new_value = value_of(log_sum_exp(terms)) + std::log(h);
      if(fabs(new_value - last_value) < tolerance.inexact(new_value, last_value)) {
        halving_h_helps = false;
      }
      last_value = new_value;
    }
  }
  return log_sum_exp(terms) + std::log(h);
}


// template <>
// var trapezoid_cosh(const var &v, const double &z) {
//   //The rule is terrible at calculating d/dv
//   //But where we need this rule, the derivative of the asymptotic formulae for
//   //Large v / large z are good
//   double v_ = value_of(v);
//   double value = trapezoid_cosh(v_, z);
//   double d_dv;
//   if(v > z) {
//     d_dv = stan::math::digamma(v_) - std::log(z) + std::log(2);
//   } else {
//     int max_terms = 100;
//     double a_k_z_k(1);    
//     double d_a_k_z_k_dv(0);
//     const double v_squared_4 = v_ * v_ * 4;
    
//     d_dv = 0;

//     for (int k = 1; k < max_terms; k++) {
//       const double next_term = (v_squared_4 - boost::math::pow<2>(2 * k - 1)) / (k * z * 8);
//       d_a_k_z_k_dv = d_a_k_z_k_dv * next_term + a_k_z_k * 8 * v_ / (k * z);
//       a_k_z_k *= next_term;
//       d_dv += d_a_k_z_k_dv;
//       if(fabs(d_a_k_z_k_dv) < 1e-8) {
//         break;
//       }
//     }
//   }

//   const double base = 0.5 * (log(pi()) - log(2) - log(z)) - z; 
//   return var(new precomp_v_vari(value, v.vi_, base * d_dv));
// }

// template <typename T_v>
// T_v trapezoid_cosh(const T_v &nu, const double &x) {
//   using std::exp;
//   using std::log;
//   using std::log1p;
//   using std::cosh;
//   using std::sinh;
//   using std::tanh;
//   using std::sqrt;
  
//   double ub = 1.0;
  
//   double nu_real = std::abs(nu);
//   while ( -x * sinh(ub) +nu_real * tanh(ub * nu_real) > 0 ) {
//     ub *= 2.0;
//   }
//   double guess = 0.75 * ub;
  
//   boost::uintmax_t maxit = 20;
//   using boost::math::tools::newton_raphson_iterate;
  
//   auto arg_max = newton_raphson_iterate(
//     [&](double t){
//       auto sinh_t = sinh(t);
//       auto v_t = nu_real * t;
//       auto tanh_vt = tanh(v_t);
//       auto cosh_vt = cosh(v_t);
//       auto x_sinh_t = x * sinh_t;
//       auto value = nu_real * tanh_vt - x_sinh_t;
//       auto ratio = nu_real / cosh_vt;
//       auto ratio2 = ratio * ratio;
//       auto D1 = ratio2 - x * cosh(t);
//       return std::make_tuple(value, D1);
//     }, guess, ub > 1.0 ? 0.5 * ub : 0, ub, 8, maxit);
  
//   const double log2 = log(2.0);
//   // log(cosh(x)) == x + log1p(exp(-2 * x)) - log(2)
//   auto pivot = logcosh(nu * arg_max)
//              - x * cosh(arg_max);
  
//   double error_estimate;
//   double L1;

//   boost::math::quadrature::exp_sinh<double> integrator;
//   double termination = sqrt(std::numeric_limits<double>::epsilon());
  
//   auto log_K = pivot + log(integrator.integrate(
//     [&](double t) {
//       auto nut = nu * t;
//       return exp(-pivot + logcosh(nut)
//                  - x * cosh(t));
//     }, termination, &error_estimate, &L1));

//   // std::cout << "error_estimate = " << error_estimate << std::endl 
//   //           << "L1 = " << L1 << std::endl
//   //           << "arg_max = " << arg_max << std::endl
//   //           << "pivot = " << pivot << std::endl;
//   return log_K;
// }

// template <>
// var trapezoid_cosh(const var &v, const double &z) {
//   double value = trapezoid_cosh(value_of(v), z);
//   typedef std::complex<double> Complex;
//   auto complex_func
//       = [z](const Complex &complex_v) { return trapezoid_cosh(complex_v, z); };

//   double d_dv = boost::math::tools::complex_step_derivative(
//       complex_func, stan::math::value_of(v));

//   return var(new precomp_v_vari(value, v.vi_, d_dv)); 
// }




////////////////////////////////////////////////////////////////
//                    CHOOSING AMONG FORMULAE                 //
////////////////////////////////////////////////////////////////

// The code to choose computation method is separate, because it is
// referenced from the test code.
enum class ComputationType { Rothwell, Asymp_v, Asymp_z, Asymp_z_log, IntegralGamma, TrapezoidCosh, FirstKind, Recurrence_Small_V_Z , SmallZ};


const double gamma_max_z = 200;
const double gamma_max_v = 3;
const double gamma_low_z = 0.01;
const double gamma_low_v = 0.001;

const double asymp_v_slope = 1;
const double asymp_v_intercept = 8;

const double asymp_z_slope = 1;
const double asymp_z_intercept = -3;
const double asymp_z_log_min_v = 10;

const double rothwell_max_v = 50;
const double rothwell_max_z = 100000;
const double rothwell_max_log_z_over_v = 300;

const double trapezoid_min_v = 100;

inline double get_rothwell_log_z_boundary(const double& v) {
  return rothwell_max_log_z_over_v / (std::fabs(v) - 0.5) - std::log(2);
}

inline ComputationType choose_computation_type(const double &v,
                                               const double &z) {
  using std::fabs;
  using std::pow;
  const double v_ = fabs(v);

  const double log_z = log(z);
  const double log_v = log(v_);

  if(v_ < gamma_low_v && z < gamma_low_z) {
    return ComputationType::Rothwell;
  } 
  if(v_ < gamma_max_v && z < gamma_max_z) {
    return ComputationType::IntegralGamma;
  }
  if(v_ < rothwell_max_v && z < rothwell_max_z &&
      (v_ <= 0.5 || log(z) < get_rothwell_log_z_boundary(v_))) {
    return ComputationType::Rothwell;
  }
  if (log_v < asymp_v_slope * log_z + asymp_v_intercept &&
             log_v > asymp_z_slope * log_z + asymp_z_intercept &&
             (v_ > trapezoid_min_v || v > z)) {
    return ComputationType::TrapezoidCosh;
  } 
  if (v_ > z) {
    return ComputationType::Asymp_v;
  } 
  if (v_ > asymp_z_log_min_v) {
    return ComputationType::Asymp_z_log;
  }
  return ComputationType::Asymp_z;

  
  // if (v_ >= rothwell_max_v &&  
  //     (log_v > asymp_v_slope * log_z + asymp_v_intercept)) {
  //   return ComputationType::Asymp_v;
  // } else if(((z >= gamma_max_z && v >= rothwell_max_v) || z >= rothwell_max_z) && 
  //   (log_v < asymp_z_slope * log_z + asymp_z_intercept)) {
  //   return ComputationType::Asymp_z;
  // } else if ((v_ > gamma_low_v || z > gamma_low_z) && (z < gamma_max_z) &&
  //     (log_v < asymp_v_slope * log_z + asymp_v_intercept || v_ < gamma_max_v))  {
  //   return ComputationType::IntegralGamma;
  // } else if (v_ < rothwell_max_v
  //            && (v_ <= 0.5 || log(z) < get_rothwell_log_z_boundary(v_))) {
  //   return ComputationType::Rothwell;
  // } else {
  //   return ComputationType::IntegralGamma;
  // } 

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
  using besselk_internal::asymptotic_large_z_log;
  using besselk_internal::first_kind;
  using besselk_internal::recurrence_formula_small_v_small_z;
  using besselk_internal::integral_gamma;
  using besselk_internal::check_params;
  using besselk_internal::choose_computation_type;
  using besselk_internal::compute_rothwell;
  using besselk_internal::trapezoid_cosh;
  using besselk_internal::asymp_small_z;
  using std::fabs;
  using std::pow;

  check_params(value_of(v), value_of(z));

  if (z == 0) {
    return std::numeric_limits<double>::infinity();
  }

  T_v v_ = fabs(v);
  ComputationType ctype = choose_computation_type(value_of(v_), value_of(z));
  if(ctype == ComputationType::Rothwell) {
    return compute_rothwell(v_, z);
  } 
  if(ctype == ComputationType::Asymp_v) {
    return asymptotic_large_v(v_, z);
  } 
  if(ctype == ComputationType::Asymp_z) {
    return asymptotic_large_z(v_, z);
  } 
  if(ctype == ComputationType::Asymp_z_log) {
    return asymptotic_large_z_log(v_, z);
  } 
  if(ctype == ComputationType::IntegralGamma) {
    return integral_gamma(v_, z);
  } 
  if(ctype == ComputationType::TrapezoidCosh) {
      return trapezoid_cosh(v_, z);
  }
  if(ctype == ComputationType::FirstKind) {
      return first_kind(v_, z);
  }
  if(ctype == ComputationType::Recurrence_Small_V_Z) {
      return recurrence_formula_small_v_small_z(v_, z);
  }
  if(ctype == ComputationType::SmallZ) {
    return asymp_small_z(v_,z);
  }
  

  stan::math::domain_error("log_modified_bessel_second_kind_frac",
                            "Invalid computation type ", 0, "");
  return asymptotic_large_v(v_, z);
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
