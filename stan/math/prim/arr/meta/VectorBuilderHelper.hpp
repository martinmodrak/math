#ifndef STAN_MATH_PRIM_ARR_META_VECTORBUILDER_HELPER_HPP
#define STAN_MATH_PRIM_ARR_META_VECTORBUILDER_HELPER_HPP

#include <stan/math/prim/scal/meta/VectorBuilderHelper.hpp>
#include <stdexcept>
#include <vector>

namespace stan {

/** \ingroup type_trait
 * Template specialization for using a vector
 */
template <typename T1>
class VectorBuilderHelper<T1, true, true> {
 private:
  std::vector<T1> x_;

 public:
  explicit VectorBuilderHelper(size_t n) : x_(n) {}

  using type = std::vector<T1>;

  T1& operator[](size_t i) { return x_[i]; }

  inline type& data() { return x_; }
};
}  // namespace stan
#endif
