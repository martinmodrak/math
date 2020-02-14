#ifndef STAN_MATH_PRIM_ERR_IS_SCAL_INFINITE_HPP
#define STAN_MATH_PRIM_ERR_IS_SCAL_INFINITE_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/get.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Return <code>true</code> if <code>y</code> is infinite.
 * This function is vectorized and will check each element of
 * <code>y</code>.
 * @tparam T_y Type of y
 * @param y Variable to check
 * @throw <code>true</code> if y is infinity
 */
template <typename T_y>
inline bool is_scal_infinite(const T_y& y) {
  for (size_t n = 0; n < size(y); ++n) {
    if (!std::isinf(value_of_rec(stan::get(y, n)))) {
      return false;
    }
  }
  return true;
}

template <typename T_y>
inline bool is_scal_infinite(const std::complex<T_y>& y) {
  return std::isinf(y.real()) || std::isinf(y.imag());
}


}  // namespace math
}  // namespace stan
#endif
