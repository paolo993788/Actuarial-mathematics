// Stochastic mortality engine: Lee-Carter projections, cohort survival and
// life annuity valuation by Monte Carlo.
//
// Lee-Carter model (Lee and Carter, 1992) for the central death rate at age x
// in calendar year t:
//     ln m(x, t) = a_x + b_x k_t,
// with k_t a random walk with drift, k_t = k_{t-1} + d + sigma * eps_t.
// Within each year of age the force of mortality is constant, so the one-year
// death probability is q = 1 - exp(-m). The table is closed at the highest
// age (omega) with q = 1.
//
// Timing: the valuation date is the end of the last calibration year T. A
// policyholder aged x0 at valuation is aged x0 + h - 1 during projection year
// T + h (h = 1, 2, ...), which uses k_{T+h} (cohort, or diagonal, rates).
// A life annuity-due pays 1 at times t = 0, 1, ... while the annuitant is alive.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"
#include "random.hpp"

namespace lr {

struct LeeCarter {
    int age_min = 0;          // age of the first element of ax and bx
    std::vector<double> ax;   // ages age_min, ..., omega (already extrapolated)
    std::vector<double> bx;
    double k_last = 0.0;      // k_T, period index of the last calibration year
    double drift = 0.0;       // random walk drift d
    double sigma = 0.0;       // standard deviation of the innovations
    double drift_se = 0.0;    // standard error of d (0 = no parameter uncertainty)

    int omega() const { return age_min + static_cast<int>(ax.size()) - 1; }

    void validate() const {
        if (ax.empty() || ax.size() != bx.size()) throw std::invalid_argument("ax and bx must be non-empty and of equal length");
        for (std::size_t i = 0; i < ax.size(); ++i)
            if (!std::isfinite(ax[i]) || !std::isfinite(bx[i])) throw std::invalid_argument("ax and bx must be finite");
        if (!(sigma >= 0.0) || !(drift_se >= 0.0)) throw std::invalid_argument("sigma and drift_se must be non-negative");
    }

    // One-year death probability at an integer age for period index k.
    double death_probability(int age, double k) const {
        if (age >= omega()) return 1.0;
        const std::size_t i = static_cast<std::size_t>(age - age_min);
        return 1.0 - std::exp(-std::exp(ax[i] + bx[i] * k));
    }
};

// Number of projection years until the table closes for a life aged age0.
inline int horizon_for(const LeeCarter& lc, int age0) {
    if (age0 < lc.age_min || age0 > lc.omega()) throw std::invalid_argument("age0 outside the age range of the table");
    return lc.omega() - age0 + 1;
}

// Survival probabilities S[0..H] of a life aged age0 along k_path[0..H-1],
// where k_path[h - 1] is the period index of projection year h.
inline void cohort_survival(const LeeCarter& lc, int age0, const double* k_path, int horizon, double* S) {
    S[0] = 1.0;
    for (int h = 1; h <= horizon; ++h) S[h] = S[h - 1] * (1.0 - lc.death_probability(age0 + h - 1, k_path[h - 1]));
}

inline double annuity_due(const double* S, const double* discount, int horizon) {
    double value = 0.0;
    for (int t = 0; t < horizon; ++t) value += S[t] * discount[t];
    return value;
}

struct AnnuitySimulation {
    int horizon = 0;
    std::vector<double> pv_systematic;    // expected PV per policy in each scenario (infinitely large portfolio)
    std::vector<double> pv_portfolio;     // realised average PV per policy of a portfolio of n_lives
    std::vector<double> life_expectancy;  // curtate cohort life expectancy in each scenario
    std::vector<double> k_paths;          // (scenario, year) if requested
};

// Monte Carlo valuation of a life annuity-due of 1 per year.
// Scenario s uses its own random stream (seed, s): the drift is drawn from
// N(d, drift_se^2) when drift_se > 0 (parameter uncertainty), then the path
// of k is simulated. With n_lives > 0 the curtate lifetime K of each life is
// drawn by inversion, P(K >= t) = S(t), and the portfolio PV per policy is
// the average of sum_{t=0}^{K} v(t).
inline AnnuitySimulation simulate_annuity(const LeeCarter& lc, int age0, const std::vector<double>& discount,
                                          long long n_scenarios, long long n_lives, std::uint64_t seed,
                                          int n_threads = 0, bool keep_paths = false) {
    lc.validate();
    const int H = horizon_for(lc, age0);
    if (static_cast<int>(discount.size()) < H) throw std::invalid_argument("discount factors must cover the whole horizon");
    if (n_scenarios < 1 || n_lives < 0) throw std::invalid_argument("need n_scenarios >= 1 and n_lives >= 0");
    std::vector<double> cum_v(H);
    double acc = 0.0;
    for (int t = 0; t < H; ++t) cum_v[t] = (acc += discount[t]);

    AnnuitySimulation out;
    out.horizon = H;
    const std::size_t n = static_cast<std::size_t>(n_scenarios);
    out.pv_systematic.assign(n, 0.0);
    out.life_expectancy.assign(n, 0.0);
    if (n_lives > 0) out.pv_portfolio.assign(n, 0.0);
    if (keep_paths) out.k_paths.assign(n * H, 0.0);

    parallel_for(n, n_threads, [&](std::size_t s) {
        Xoshiro256 rng(seed, s);
        std::vector<double> k(H), S(H + 1);
        const double drift = lc.drift + lc.drift_se * rng.normal();
        double kt = lc.k_last;
        for (int h = 0; h < H; ++h) {
            kt += drift + lc.sigma * rng.normal();
            k[h] = kt;
        }
        cohort_survival(lc, age0, k.data(), H, S.data());
        out.pv_systematic[s] = annuity_due(S.data(), discount.data(), H);
        double e = 0.0;
        for (int t = 1; t <= H; ++t) e += S[t];
        out.life_expectancy[s] = e;
        if (keep_paths) std::copy(k.begin(), k.end(), out.k_paths.begin() + s * H);
        if (n_lives > 0) {
            double total = 0.0;
            const auto first = S.begin() + 1, last = S.end();
            for (long long i = 0; i < n_lives; ++i) {
                const double u = rng.uniform();
                // K = number of t in 1..H with S(t) > u (S is non-increasing and S(H) = 0).
                const auto it = std::partition_point(first, last, [u](double surv) { return surv > u; });
                total += cum_v[static_cast<std::size_t>(it - first)];
            }
            out.pv_portfolio[s] = total / static_cast<double>(n_lives);
        }
    });
    return out;
}

struct OneYearRecalibration {
    double best_estimate = 0.0;       // value at time 0 with the central projection
    std::vector<double> value;        // value at time 0 after one simulated year and re-estimation of the drift
    std::vector<double> k_next;       // simulated k_{T+1}
    std::vector<double> drift_next;   // re-estimated drift
};

// One-year view of longevity trend risk (Richards, Currie and Ritchie, 2014):
// simulate the next year's period index, re-estimate the random-walk drift
// with the extra observation, d' = (k_{T+1} - k_first) / (n + 1), and revalue
// the annuity with the updated central projection. The value at time 0 is
//     X = v(0) + p_{x0}(k_{T+1}) * sum_{j >= 0} S'(j) v(1 + j),
// where S' is the survival of the life aged x0 + 1 under k_{T+1} + (h - 1) d'.
// The central projection of the best estimate is k_{T+h} = k_T + h d.
inline OneYearRecalibration one_year_recalibration(const LeeCarter& lc, double k_first, int n_increments, int age0,
                                                   const std::vector<double>& discount, long long n_scenarios,
                                                   std::uint64_t seed, int n_threads = 0) {
    lc.validate();
    const int H = horizon_for(lc, age0);
    if (H < 2) throw std::invalid_argument("age0 must be below the closing age");
    if (static_cast<int>(discount.size()) < H) throw std::invalid_argument("discount factors must cover the whole horizon");
    if (n_increments < 1 || n_scenarios < 1) throw std::invalid_argument("need n_increments >= 1 and n_scenarios >= 1");

    OneYearRecalibration out;
    {
        std::vector<double> k(H), S(H + 1);
        for (int h = 0; h < H; ++h) k[h] = lc.k_last + (h + 1) * lc.drift;
        cohort_survival(lc, age0, k.data(), H, S.data());
        out.best_estimate = annuity_due(S.data(), discount.data(), H);
    }
    const std::size_t n = static_cast<std::size_t>(n_scenarios);
    out.value.assign(n, 0.0);
    out.k_next.assign(n, 0.0);
    out.drift_next.assign(n, 0.0);
    parallel_for(n, n_threads, [&](std::size_t s) {
        Xoshiro256 rng(seed, s);
        const double drift = lc.drift + lc.drift_se * rng.normal();
        const double k1 = lc.k_last + drift + lc.sigma * rng.normal();
        const double new_drift = (k1 - k_first) / (n_increments + 1.0);
        const double p = 1.0 - lc.death_probability(age0, k1);
        // Future survival of the life aged age0 + 1, projection years T + 2, T + 3, ...
        std::vector<double> k(H - 1), S(H);
        for (int j = 0; j < H - 1; ++j) k[j] = k1 + (j + 1) * new_drift;
        cohort_survival(lc, age0 + 1, k.data(), H - 1, S.data());
        double future = 0.0;
        for (int j = 0; j < H - 1; ++j) future += S[j] * discount[j + 1];
        out.value[s] = discount[0] + p * future;
        out.k_next[s] = k1;
        out.drift_next[s] = new_drift;
    });
    return out;
}

}  // namespace lr
