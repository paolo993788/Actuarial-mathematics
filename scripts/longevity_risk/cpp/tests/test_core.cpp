// Unit tests of the header-only C++ mortality engine, compiled without Python (see ../CMakeLists.txt).
//
// The Python test suite (tests/longevity_risk) compares the engine with a NumPy reference implementation through
// the pybind11 bindings. These tests check the C++ code on its own, so that it can be built with strict warnings
// and run under AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer:
// * published reference outputs of the random number generators;
// * closed forms (constant force of mortality: geometric annuity and life expectancy) and hand computations of
//   the deterministic Lee-Carter projection;
// * exact identities (a consistent one-year recalibration reproduces the best estimate; a one-member portfolio
//   reproduces the single-life simulation, stream for stream);
// * the distribution of individual lifetimes against its exact mean and variance;
// * results that must not depend on the number of threads;
// * argument validation.
// Statistical checks use four standard errors (two-sided type I error about 6e-5 each) with fixed seeds.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mortality.hpp"
#include "parallel.hpp"
#include "random.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

void report(bool ok, const char* what, const char* file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, what);
    }
}

void report_close(double a, double b, double tol, const char* what, const char* file, int line) {
    const bool ok = std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol;
    report(ok, what, file, line);
    if (!ok) std::fprintf(stderr, "    %.17g vs %.17g (tolerance %.3g)\n", a, b, tol);
}

#define CHECK(cond) report((cond), #cond, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, tol) report_close((a), (b), (tol), #a " ~ " #b, __FILE__, __LINE__)

template <class Fn>
bool throws_invalid_argument(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

double mean_of(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s / static_cast<double>(x.size());
}

double var_of(const std::vector<double>& x) {
    const double m = mean_of(x);
    double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return s / static_cast<double>(x.size() - 1);
}

std::vector<double> flat_discount(double rate, std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t t = 0; t < n; ++t) v[t] = std::pow(1.0 + rate, -static_cast<double>(t));
    return v;
}

// A Lee-Carter model for ages 60-110 with a Gompertz-like ax and a bx that falls with age.
lr::LeeCarter gompertz_model(double sigma, double drift_se) {
    lr::LeeCarter lc;
    lc.age_min = 60;
    for (int age = 60; age <= 110; ++age) {
        lc.ax.push_back(-4.8 + 0.1 * (age - 60));
        lc.bx.push_back(0.02 * std::exp(-0.02 * (age - 60)));
    }
    lc.k_last = -20.0;
    lc.drift = -1.5;
    lc.sigma = sigma;
    lc.drift_se = drift_se;
    return lc;
}

// ------------------------------------------------------------------------------------------------ random numbers

void test_splitmix64_reference() {
    // Reference sequence of Vigna's splitmix64.c for the seed 1234567.
    std::uint64_t state = 1234567;
    const std::uint64_t expected[] = {6457827717110365317ULL, 3203168211198807973ULL, 9817491932198370423ULL,
                                      4593380528125082431ULL, 16408922859458223821ULL};
    for (std::uint64_t e : expected) CHECK(lr::splitmix64(state) == e);
}

void test_xoshiro_streams() {
    // Values from a transcription of the reference xoshiro256** algorithm (Blackman and Vigna), with the state
    // seeded as in lr::Xoshiro256; the transcription reproduces the published outputs from the state {1, 2, 3, 4}.
    lr::Xoshiro256 a(42, 0), b(42, 7);
    const std::uint64_t ea[] = {17604071880264941726ULL, 13049929662915288091ULL, 16314220431934199612ULL,
                                16017857136869241811ULL};
    const std::uint64_t eb[] = {15260628496888913177ULL, 12080209494996618911ULL, 16192203974094658541ULL,
                                4943040130464974858ULL};
    for (int k = 0; k < 4; ++k) {
        CHECK(a.next() == ea[k]);
        CHECK(b.next() == eb[k]);
    }
}

void test_uniform_and_normal_moments() {
    const std::size_t n = 1000000;
    const double dn = static_cast<double>(n);
    lr::Xoshiro256 rng(2026);
    double s = 0.0, q = 0.0, lo = 1.0, hi = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = rng.uniform();
        s += u;
        q += u * u;
        lo = std::min(lo, u);
        hi = std::max(hi, u);
    }
    CHECK(lo > 0.0 && hi < 1.0);  // open interval
    CHECK_CLOSE(s / dn, 0.5, 4.0 * std::sqrt(1.0 / 12.0 / dn));
    CHECK_CLOSE(q / dn - (s / dn) * (s / dn), 1.0 / 12.0, 4.0 * std::sqrt(1.0 / 180.0 / dn));
    lr::Xoshiro256 g(7);
    double m = 0.0, v = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = g.normal();
        m += z;
        v += z * z;
    }
    m /= dn;
    v = v / dn - m * m;
    CHECK_CLOSE(m, 0.0, 4.0 / std::sqrt(dn));
    CHECK_CLOSE(v, 1.0, 4.0 * std::sqrt(2.0 / dn));
}

void test_parallel_for() {
    const std::size_t n = 1000;
    for (int threads : {1, 3, 8}) {
        std::vector<double> out(n, 0.0);
        lr::parallel_for(n, threads, [&](std::size_t i) { out[i] = static_cast<double>(i) * static_cast<double>(i); });
        double total = 0.0;
        for (double v : out) total += v;
        CHECK(total == 332833500.0);  // sum of i^2 for i < 1000
    }
    bool rethrown = false;
    try {
        lr::parallel_for(100, 4, [](std::size_t i) {
            if (i == 57) throw std::runtime_error("task failed");
        });
    } catch (const std::runtime_error&) {
        rethrown = true;
    }
    CHECK(rethrown);
}

// ------------------------------------------------------------------------------------------------ Lee-Carter

void test_lee_carter_basics() {
    const lr::LeeCarter lc = gompertz_model(1.0, 0.1);
    CHECK(lc.omega() == 110);
    // q = 1 - exp(-m) with ln m = a + b k, and the table closes at omega.
    const double m70 = std::exp(lc.ax[10] + lc.bx[10] * (-25.0));
    CHECK_CLOSE(lc.death_probability(70, -25.0), 1.0 - std::exp(-m70), 1e-15);
    CHECK(lc.death_probability(110, -25.0) == 1.0);
    CHECK(lc.death_probability(115, -25.0) == 1.0);
    CHECK(lr::horizon_for(lc, 65) == 46);
    CHECK(lr::horizon_for(lc, 110) == 1);
    CHECK(throws_invalid_argument([&] { lr::horizon_for(lc, 59); }));
    CHECK(throws_invalid_argument([&] { lr::horizon_for(lc, 111); }));
    lr::LeeCarter bad = lc;
    bad.bx.pop_back();
    CHECK(throws_invalid_argument([&] { bad.validate(); }));
    bad = lc;
    bad.ax[3] = std::numeric_limits<double>::quiet_NaN();
    CHECK(throws_invalid_argument([&] { bad.validate(); }));
    bad = lc;
    bad.sigma = -1.0;
    CHECK(throws_invalid_argument([&] { bad.validate(); }));
}

void test_constant_force_of_mortality() {
    // bx = 0 and ax = ln(mu): p = e^{-mu} at every age below omega, so for a life aged x0 with H years to omega
    //   annuity-due = sum_{t<H} (p v)^t = (1 - (p v)^H) / (1 - p v),  e = sum_{t=1}^{H-1} p^t = p (1 - p^{H-1}) / (1 - p).
    const double mu = 0.03, rate = 0.02;
    lr::LeeCarter lc;
    lc.age_min = 60;
    lc.ax.assign(41, std::log(mu));  // ages 60-100
    lc.bx.assign(41, 0.0);
    lc.sigma = 0.7;                  // irrelevant when bx = 0
    const int age0 = 65, H = lr::horizon_for(lc, age0);
    const std::vector<double> discount = flat_discount(rate, static_cast<std::size_t>(H));
    const lr::AnnuitySimulation sim = lr::simulate_annuity(lc, age0, discount, 50, 0, 3, 2);
    const double p = std::exp(-mu), pv = p / (1.0 + rate);
    const double annuity = (1.0 - std::pow(pv, H)) / (1.0 - pv);
    const double expectancy = p * (1.0 - std::pow(p, H - 1)) / (1.0 - p);
    bool all = true;
    for (std::size_t s = 0; s < sim.pv_systematic.size(); ++s)
        all = all && std::abs(sim.pv_systematic[s] - annuity) < 1e-12 && std::abs(sim.life_expectancy[s] - expectancy) < 1e-12;
    CHECK(all);
    CHECK(sim.horizon == H && sim.pv_portfolio.empty());
}

// Deterministic central projection written out: k_{T+h} = k_T + h d.
double central_annuity(const lr::LeeCarter& lc, int age0, const std::vector<double>& discount, double& expectancy) {
    const int H = lr::horizon_for(lc, age0);
    double S = 1.0, value = 0.0;
    expectancy = 0.0;
    for (int t = 0; t < H; ++t) {
        value += S * discount[static_cast<std::size_t>(t)];
        const double k = lc.k_last + (t + 1) * lc.drift;
        S *= 1.0 - lc.death_probability(age0 + t, k);
        expectancy += S;
    }
    return value;
}

void test_deterministic_projection_and_monotonicity() {
    lr::LeeCarter lc = gompertz_model(0.0, 0.0);
    const std::vector<double> discount = flat_discount(0.025, 60);
    double e = 0.0;
    const double by_hand = central_annuity(lc, 65, discount, e);
    const lr::AnnuitySimulation sim = lr::simulate_annuity(lc, 65, discount, 8, 0, 11, 2, true);
    for (std::size_t s = 0; s < 8; ++s) {
        CHECK_CLOSE(sim.pv_systematic[s], by_hand, 1e-12);
        CHECK_CLOSE(sim.life_expectancy[s], e, 1e-12);
        CHECK_CLOSE(sim.k_paths[s * static_cast<std::size_t>(sim.horizon)], lc.k_last + lc.drift, 1e-12);
    }
    // With bx > 0 a faster fall in k (more negative drift) lowers mortality and raises the annuity value.
    lc.drift = -2.5;
    double e_fast = 0.0;
    CHECK(central_annuity(lc, 65, discount, e_fast) > by_hand);
    CHECK(e_fast > e);
    CHECK(lr::simulate_annuity(lc, 65, discount, 1, 0, 11).pv_systematic[0] > by_hand);
}

void test_stochastic_projection() {
    const lr::LeeCarter lc = gompertz_model(1.2, 0.3);
    const std::vector<double> discount = flat_discount(0.02, 60);
    const long long n = 40000;
    const lr::AnnuitySimulation sim = lr::simulate_annuity(lc, 65, discount, n, 0, 5, 4, true);
    // k_{T+1} = k_T + d + sigma e_1 with d ~ N(drift, drift_se^2): mean k_T + drift, variance sigma^2 + drift_se^2.
    std::vector<double> k1(static_cast<std::size_t>(n)), kH(static_cast<std::size_t>(n));
    const std::size_t H = static_cast<std::size_t>(sim.horizon);
    for (std::size_t s = 0; s < k1.size(); ++s) {
        k1[s] = sim.k_paths[s * H];
        kH[s] = sim.k_paths[s * H + H - 1];
    }
    const double var1 = lc.sigma * lc.sigma + lc.drift_se * lc.drift_se;
    CHECK_CLOSE(mean_of(k1), lc.k_last + lc.drift, 4.0 * std::sqrt(var1 / static_cast<double>(n)));
    CHECK_CLOSE(var_of(k1), var1, 4.0 * var1 * std::sqrt(2.0 / static_cast<double>(n)));
    // After h years: variance h sigma^2 + h^2 drift_se^2 (parameter uncertainty grows linearly in the horizon).
    const double h = static_cast<double>(H);
    const double varH = h * lc.sigma * lc.sigma + h * h * lc.drift_se * lc.drift_se;
    CHECK_CLOSE(mean_of(kH), lc.k_last + h * lc.drift, 4.0 * std::sqrt(varH / static_cast<double>(n)));
    CHECK_CLOSE(var_of(kH), varH, 4.0 * varH * std::sqrt(2.0 / static_cast<double>(n)));
    // Each scenario has its own random stream: identical results whatever the number of threads.
    const lr::AnnuitySimulation one = lr::simulate_annuity(lc, 70, discount, 3000, 50, 9, 1);
    const lr::AnnuitySimulation many = lr::simulate_annuity(lc, 70, discount, 3000, 50, 9, 5);
    CHECK(one.pv_systematic == many.pv_systematic && one.pv_portfolio == many.pv_portfolio &&
          one.life_expectancy == many.life_expectancy);
    CHECK(throws_invalid_argument([&] { lr::simulate_annuity(lc, 65, flat_discount(0.02, 10), 10, 0, 1); }));
    CHECK(throws_invalid_argument([&] { lr::simulate_annuity(lc, 65, discount, 0, 0, 1); }));
    CHECK(throws_invalid_argument([&] { lr::simulate_annuity(lc, 65, discount, 10, -1, 1); }));
}

void test_individual_lifetimes() {
    // Deterministic mortality: the realised PV of one life is Y = sum_{t<=K} v(t) with P(K = t) = S(t) - S(t+1),
    // whose mean is the annuity-due. The average over n lives is compared with the exact mean and variance.
    const lr::LeeCarter lc = gompertz_model(0.0, 0.0);
    const int age0 = 75;
    const std::vector<double> discount = flat_discount(0.03, 40);
    const int H = lr::horizon_for(lc, age0);
    std::vector<double> S(static_cast<std::size_t>(H) + 1, 0.0);
    S[0] = 1.0;
    for (int t = 0; t < H; ++t)
        S[static_cast<std::size_t>(t) + 1] = S[static_cast<std::size_t>(t)] *
            (1.0 - lc.death_probability(age0 + t, lc.k_last + (t + 1) * lc.drift));
    CHECK(S[static_cast<std::size_t>(H)] == 0.0);  // closed table
    double mean = 0.0, second = 0.0, cum = 0.0;
    for (int t = 0; t < H; ++t) {
        const std::size_t i = static_cast<std::size_t>(t);
        cum += discount[i];
        const double prob = S[i] - S[i + 1];
        mean += prob * cum;
        second += prob * cum * cum;
    }
    const long long n_lives = 200000;
    const lr::AnnuitySimulation sim = lr::simulate_annuity(lc, age0, discount, 3, n_lives, 21, 3);
    CHECK_CLOSE(sim.pv_systematic[0], mean, 1e-12);  // E[Y] is the annuity-due
    const double se = std::sqrt((second - mean * mean) / static_cast<double>(n_lives));
    for (double pv : sim.pv_portfolio) CHECK_CLOSE(pv, mean, 4.0 * se);
}

// ------------------------------------------------------------------------------------------------ one-year view

void test_one_year_recalibration() {
    // Without randomness and with k_first consistent with the drift, the re-estimated drift equals the old one and
    // the revalued annuity equals the best estimate exactly.
    lr::LeeCarter lc = gompertz_model(0.0, 0.0);
    const int n_increments = 30;
    const double k_first = lc.k_last - n_increments * lc.drift;
    const std::vector<double> discount = flat_discount(0.02, 60);
    const lr::OneYearRecalibration r = lr::one_year_recalibration(lc, k_first, n_increments, 65, discount, 4, 1, 2);
    double e = 0.0;
    CHECK_CLOSE(r.best_estimate, central_annuity(lc, 65, discount, e), 1e-12);
    for (std::size_t s = 0; s < 4; ++s) {
        CHECK_CLOSE(r.drift_next[s], lc.drift, 1e-12);
        CHECK_CLOSE(r.value[s], r.best_estimate, 1e-12);
    }
    // With randomness: d' = (k_{T+1} - k_first) / (n + 1) scenario by scenario, and thread invariance.
    lc = gompertz_model(1.0, 0.2);
    const lr::OneYearRecalibration one = lr::one_year_recalibration(lc, k_first, n_increments, 65, discount, 5000, 8, 1);
    const lr::OneYearRecalibration many = lr::one_year_recalibration(lc, k_first, n_increments, 65, discount, 5000, 8, 4);
    CHECK(one.value == many.value && one.k_next == many.k_next);
    bool drift_rule = true;
    for (std::size_t s = 0; s < one.value.size(); ++s)
        drift_rule = drift_rule && std::abs(one.drift_next[s] - (one.k_next[s] - k_first) / (n_increments + 1.0)) < 1e-12;
    CHECK(drift_rule);
    CHECK(throws_invalid_argument([&] { lr::one_year_recalibration(lc, k_first, n_increments, 110, discount, 10, 1); }));
    CHECK(throws_invalid_argument([&] { lr::one_year_recalibration(lc, k_first, 0, 65, discount, 10, 1); }));
}

// ------------------------------------------------------------------------------------------------ portfolios

void test_portfolios() {
    const lr::LeeCarter lc = gompertz_model(1.0, 0.2);
    const std::vector<double> discount = flat_discount(0.02, 60);
    // Grouping: distinct ascending ages, total amount per age, group index per member.
    const lr::PortfolioGroups g = lr::group_by_age(lc, {70, 65, 70, 80}, {1.0, 2.0, 3.0, 4.0});
    CHECK((g.ages == std::vector<int>{65, 70, 80}));
    CHECK((g.totals == std::vector<double>{2.0, 4.0, 4.0}));
    CHECK((g.group == std::vector<std::size_t>{1, 0, 1, 2}));
    CHECK(throws_invalid_argument([&] { lr::group_by_age(lc, {65, 70}, {1.0}); }));
    CHECK(throws_invalid_argument([&] { lr::group_by_age(lc, {65, 120}, {1.0, 1.0}); }));

    // A one-member portfolio reproduces the single-life simulation stream for stream: systematic values scale
    // with the amount, and one simulated lifetime matches n_lives = 1.
    const long long n = 2000;
    const lr::AnnuitySimulation single = lr::simulate_annuity(lc, 68, discount, n, 1, 13, 3);
    const lr::PortfolioSimulation port = lr::simulate_portfolio(lc, {68}, {2500.0}, discount, n, true, 13, 2);
    bool same = true;
    for (std::size_t s = 0; s < single.pv_systematic.size(); ++s)
        same = same && std::abs(port.pv_systematic[s] - 2500.0 * single.pv_systematic[s]) < 1e-9 &&
               std::abs(port.pv_realised[s] - 2500.0 * single.pv_portfolio[s]) < 1e-9;
    CHECK(same);
    // Systematic value of a portfolio = sum over ages of amount x annuity in the same scenario.
    const lr::PortfolioSimulation mixed = lr::simulate_portfolio(lc, {65, 80, 65}, {1.0, 2.0, 3.0}, discount, 500, false, 4, 3);
    const lr::AnnuitySimulation a65 = lr::simulate_annuity(lc, 65, discount, 500, 0, 4, 1);
    const lr::AnnuitySimulation a80 = lr::simulate_annuity(lc, 80, discount, 500, 0, 4, 1);
    bool additive = true;
    for (std::size_t s = 0; s < 500; ++s)
        additive = additive && std::abs(mixed.pv_systematic[s] - (4.0 * a65.pv_systematic[s] + 2.0 * a80.pv_systematic[s])) < 1e-9;
    CHECK(additive);
    CHECK(mixed.pv_realised.empty());
    const lr::PortfolioSimulation mixed_threads = lr::simulate_portfolio(lc, {65, 80, 65}, {1.0, 2.0, 3.0}, discount, 500, true, 4, 1);
    const lr::PortfolioSimulation mixed_threads5 = lr::simulate_portfolio(lc, {65, 80, 65}, {1.0, 2.0, 3.0}, discount, 500, true, 4, 5);
    CHECK(mixed_threads.pv_realised == mixed_threads5.pv_realised);

    // One-year view of a portfolio: amount-weighted sum of the members' one-year values, scenario by scenario.
    const double k_first = lc.k_last - 30 * lc.drift;
    const lr::OneYearRecalibration p1 = lr::portfolio_one_year(lc, k_first, 30, {65, 80, 65}, {1.0, 2.0, 3.0}, discount, 300, 6, 2);
    const lr::OneYearRecalibration r65 = lr::one_year_recalibration(lc, k_first, 30, 65, discount, 300, 6, 1);
    const lr::OneYearRecalibration r80 = lr::one_year_recalibration(lc, k_first, 30, 80, discount, 300, 6, 1);
    CHECK_CLOSE(p1.best_estimate, 4.0 * r65.best_estimate + 2.0 * r80.best_estimate, 1e-12);
    bool weighted = true;
    for (std::size_t s = 0; s < 300; ++s)
        weighted = weighted && std::abs(p1.value[s] - (4.0 * r65.value[s] + 2.0 * r80.value[s])) < 1e-12;
    CHECK(weighted);
    CHECK(p1.k_next == r65.k_next);
}

}  // namespace

int main() {
    test_splitmix64_reference();
    test_xoshiro_streams();
    test_uniform_and_normal_moments();
    test_parallel_for();
    test_lee_carter_basics();
    test_constant_force_of_mortality();
    test_deterministic_projection_and_monotonicity();
    test_stochastic_projection();
    test_individual_lifetimes();
    test_one_year_recalibration();
    test_portfolios();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
