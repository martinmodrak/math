#include <stan/math/prim/meta.hpp>
#include <gtest/gtest.h>

TEST(MathMetaPrimScal, or_type) {
  bool temp = stan::math::disjunction<std::true_type, std::true_type,
                                      std::true_type>::value;
  EXPECT_TRUE(temp);
  temp = stan::math::disjunction<std::false_type, std::false_type,
                                 std::false_type>::value;
  EXPECT_FALSE(temp);
  temp = stan::math::disjunction<std::false_type, std::true_type,
                                 std::true_type>::value;
  EXPECT_TRUE(temp);
  temp = stan::math::disjunction<std::true_type, std::true_type,
                                 std::false_type>::value;
  EXPECT_TRUE(temp);
}
