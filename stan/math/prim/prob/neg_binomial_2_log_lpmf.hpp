#ifndef STAN_MATH_PRIM_PROB_NEG_BINOMIAL_2_LOG_LPMF_HPP
#define STAN_MATH_PRIM_PROB_NEG_BINOMIAL_2_LOG_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/scal/fun/size_zero.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <cmath>

namespace stan {
namespace math {

namespace internal {
  //TODO[martinmodrak] merge with neg_binomial_2_phi_cutoff once #1497 is 
  //processed
  constexpr double neg_binomial_2_log_phi_cutoff = 1e10;
}

// NegBinomial(n|eta, phi)  [phi > 0;  n >= 0]
template <bool propto, typename T_n, typename T_log_location,
          typename T_precision>
return_type_t<T_log_location, T_precision> neg_binomial_2_log_lpmf(
    const T_n& n, const T_log_location& eta, const T_precision& phi) {
  typedef
      typename stan::partials_return_type<T_n, T_log_location,
                                          T_precision>::type T_partials_return;

  static const char* function = "neg_binomial_2_log_lpmf";

  if (size_zero(n, eta, phi)) {
    return 0.0;
  }

  T_partials_return logp(0.0);
  check_nonnegative(function, "Failures variable", n);
  check_finite(function, "Log location parameter", eta);
  check_positive_finite(function, "Precision parameter", phi);
  check_consistent_sizes(function, "Failures variable", n,
                         "Log location parameter", eta, "Precision parameter",
                         phi);

  if (!include_summand<propto, T_log_location, T_precision>::value) {
    return 0.0;
  }

  using std::exp;
  using std::log;

  scalar_seq_view<T_n> n_vec(n);
  scalar_seq_view<T_log_location> eta_vec(eta);
  scalar_seq_view<T_precision> phi_vec(phi);
  size_t max_size_seq_view = max_size(n, eta, phi);

  operands_and_partials<T_log_location, T_precision> ops_partials(eta, phi);

  size_t len_ep = max_size(eta, phi);
  size_t len_np = max_size(n, phi);

  VectorBuilder<true, T_partials_return, T_log_location> eta_val(size(eta));
  for (size_t i = 0, max_size_seq_view = size(eta); i < max_size_seq_view;
       ++i) {
    eta_val[i] = value_of(eta_vec[i]);
  }

  VectorBuilder<true, T_partials_return, T_precision> phi_val(size(phi));
  for (size_t i = 0, max_size_seq_view = size(phi); i < max_size_seq_view;
       ++i) {
    phi_val[i] = value_of(phi_vec[i]);
  }

  VectorBuilder<true, T_partials_return, T_precision> log_phi(size(phi));
  for (size_t i = 0, max_size_seq_view = size(phi); i < max_size_seq_view;
       ++i) {
    log_phi[i] = log(phi_val[i]);
  }

  VectorBuilder<true, T_partials_return, T_log_location, T_precision>
      logsumexp_eta_logphi(len_ep);
  for (size_t i = 0; i < len_ep; ++i) {
    logsumexp_eta_logphi[i] = log_sum_exp(eta_val[i], log_phi[i]);
  }

  VectorBuilder<true, T_partials_return, T_n, T_precision> n_plus_phi(len_np);
  for (size_t i = 0; i < len_np; ++i) {
    n_plus_phi[i] = n_vec[i] + phi_val[i];
  }

  for (size_t i = 0; i < max_size_seq_view; i++) {
   if (phi_val[i] > internal::neg_binomial_2_log_phi_cutoff) {
      // Phi is large, delegate to Poisson.
      // Copying the code here as just calling
      // poisson_lpmf does not preserve propto logic correctly.
      // Note that Poisson can be seen as first term of Taylor series for
      // phi -> Inf. Similarly, the derivative wrt mu and phi can be obtained
      // via the Same Taylor expansions, in Mathematica using the code:
      //
      // nb2log[n_,eta_,phi_]:= LogGamma[n + phi] - LogGamma[n + 1] - 
      //     LogGamma[phi ] + n * (eta - Log[Exp[eta] + phi]) + 
      //     phi * (Log[phi] - Log[Exp[eta] + phi]);
      // nb2logdeta[n_,eta_,phi_]= D[nb2log[n, eta, phi],eta];
      // nb2logdphi[n_,eta_,phi_]= D[nb2log[n, eta, phi],phi];
      // Series[nb2logdeta[n, eta, phi],{phi, Infinity, 0}]
      // Series[nb2logdphi[n, eta, phi],{phi, Infinity, 2}]      
      //
      // The derivative wrt phi = 0 + O(1/phi^2),
      // But the quadratic term is big enough to warrant inclusion here
      // (can be around 1e-6 at cutoff).
      if (include_summand<propto>::value) {
        logp -= lgamma(n_vec[i] + 1.0);
      }
      double exp_eta = exp(eta_val[i]);
      if (include_summand<propto, T_log_location>::value) {
        logp += eta_val[i] * n_vec[i] - exp_eta;
      }

      if (!is_constant_all<T_log_location>::value) {
        ops_partials.edge1_.partials_[i] += n_vec[i] - exp_eta;
      }
      if (!is_constant_all<T_precision>::value) {
        ops_partials.edge2_.partials_[i]
            += (exp_eta * (-exp_eta + 2 * n_vec[i])
                + n_vec[i] * (1 - n_vec[i]))
               / (2 * square(phi_val[i]));
      }
    } else {
      if (include_summand<propto, T_precision>::value) {
        logp += binomial_coefficient_log(n_plus_phi[i] - 1, n_vec[i]);
      }
      if (include_summand<propto, T_log_location>::value) {
        logp += n_vec[i] * eta_val[i];
      }
      logp += phi_val[i] * (log_phi[i] - logsumexp_eta_logphi[i])
              - n_vec[i] * logsumexp_eta_logphi[i];

      if (!is_constant_all<T_log_location>::value) {
        ops_partials.edge1_.partials_[i]
            += n_vec[i] - n_plus_phi[i] / (phi_val[i] / exp(eta_val[i]) + 1.0);
      }
      if (!is_constant_all<T_precision>::value) {
        ops_partials.edge2_.partials_[i]
            += 1.0 - n_plus_phi[i] / (exp(eta_val[i]) + phi_val[i]) + log_phi[i]
              - logsumexp_eta_logphi[i] - digamma(phi_val[i])
              + digamma(n_plus_phi[i]);
      }
    }
  }
  return ops_partials.build(logp);
}

template <typename T_n, typename T_log_location, typename T_precision>
inline return_type_t<T_log_location, T_precision> neg_binomial_2_log_lpmf(
    const T_n& n, const T_log_location& eta, const T_precision& phi) {
  return neg_binomial_2_log_lpmf<false>(n, eta, phi);
}
}  // namespace math
}  // namespace stan
#endif
