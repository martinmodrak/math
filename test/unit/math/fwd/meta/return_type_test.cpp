#include <stan/math/fwd/scal.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

using stan::math::fvar;
using stan::return_type;

TEST(MetaTraitsFwdScal, ReturnTypeFvarDouble) {
  test::expect_same_type<fvar<double>, return_type<fvar<double> >::type>();
}

TEST(MetaTraitsFwdScal, ReturnTypeFvarFvarDouble) {
  test::expect_same_type<fvar<fvar<double> >,
                         return_type<fvar<fvar<double> > >::type>();
}

TEST(MetaTraitsFwdScal, ReturnTypeFvarDoubleTenParams) {
  test::expect_same_type<
      fvar<double>,
      return_type<double, fvar<double>, double, int, double, float, float,
                  float, fvar<double>, int>::type>();
}

TEST(MetaTraitsFwdScal, ReturnTypeFvarFvarDoubleTenParams) {
  test::expect_same_type<
      fvar<fvar<double> >,
      return_type<double, fvar<fvar<double> >, double, int, double, float,
                  float, float, fvar<fvar<double> >, int>::type>();
}
