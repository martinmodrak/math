#include <stan/math/prim/meta.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

using stan::partials_type;

TEST(MetaTraitsPrimScal, PartialsTypeDouble) {
  test::expect_same_type<double, partials_type<double>::type>();
}

TEST(MetaTraitsPrimScal, PartialsTypeFloat) {
  test::expect_same_type<float, partials_type<float>::type>();
}

TEST(MetaTraitsPrimScal, PartialsTypeInt) {
  test::expect_same_type<int, partials_type<int>::type>();
}
