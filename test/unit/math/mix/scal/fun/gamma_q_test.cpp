#include <test/unit/math/test_ad.hpp>
#include <limits>

TEST(mathMixScalFun, gammaQ) {
  auto f = [](const auto& x1, const auto& x2) {
    return stan::math::gamma_q(x1, x2);
  };
  stan::test::expect_ad(f, 0.5001, 1.0001);
  stan::test::expect_ad(f, 0.5, 1.0);
  stan::test::expect_ad(f, 1.0, 1.0);

  double nan = std::numeric_limits<double>::quiet_NaN();
  stan::test::expect_ad(f, 1.0, nan);
  stan::test::expect_ad(f, nan, 1.0);
  stan::test::expect_ad(f, nan, nan);

  // this still fails forward mode; left regression test in rev/scal/fun
  // stan::test::expect_value(f, 8.01006, 2.47579e+215);
}
