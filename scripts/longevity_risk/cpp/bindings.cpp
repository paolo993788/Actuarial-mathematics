// pybind11 bindings exposing the stochastic mortality engine as
// longevity_risk._core. Simulations release the GIL and run in parallel.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "mortality.hpp"

namespace py = pybind11;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

std::vector<double> to_vector(const DoubleArray& a) {
    const auto buf = a.request();
    const double* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

py::array_t<double> to_array(const std::vector<double>& v) {
    py::array_t<double> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

lr::LeeCarter make_model(int age_min, const DoubleArray& ax, const DoubleArray& bx, double k_last, double drift,
                         double sigma, double drift_se) {
    lr::LeeCarter lc;
    lc.age_min = age_min;
    lc.ax = to_vector(ax);
    lc.bx = to_vector(bx);
    lc.k_last = k_last;
    lc.drift = drift;
    lc.sigma = sigma;
    lc.drift_se = drift_se;
    lc.validate();
    return lc;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ Lee-Carter simulation and life annuity valuation engine for longevity_risk.";

    m.def(
        "cohort_survival",
        [](int age_min, const DoubleArray& ax, const DoubleArray& bx, int age0, const DoubleArray& k_path) {
            const lr::LeeCarter lc = make_model(age_min, ax, bx, 0.0, 0.0, 0.0, 0.0);
            const int H = lr::horizon_for(lc, age0);
            const auto k = to_vector(k_path);
            if (static_cast<int>(k.size()) < H) throw py::value_error("k_path must cover the whole horizon");
            std::vector<double> S(H + 1);
            lr::cohort_survival(lc, age0, k.data(), H, S.data());
            return to_array(S);
        },
        py::arg("age_min"), py::arg("ax"), py::arg("bx"), py::arg("age0"), py::arg("k_path"),
        "Survival probabilities S(0..H) of a life aged age0 along a path of the period index.");

    m.def(
        "simulate_annuity",
        [](int age_min, const DoubleArray& ax, const DoubleArray& bx, double k_last, double drift, double sigma,
           double drift_se, int age0, const DoubleArray& discount, long long n_scenarios, long long n_lives,
           std::uint64_t seed, int n_threads, bool keep_paths) {
            const lr::LeeCarter lc = make_model(age_min, ax, bx, k_last, drift, sigma, drift_se);
            const auto v = to_vector(discount);
            lr::AnnuitySimulation res;
            {
                py::gil_scoped_release release;
                res = lr::simulate_annuity(lc, age0, v, n_scenarios, n_lives, seed, n_threads, keep_paths);
            }
            py::dict d;
            d["horizon"] = res.horizon;
            d["pv_systematic"] = to_array(res.pv_systematic);
            d["life_expectancy"] = to_array(res.life_expectancy);
            d["pv_portfolio"] = to_array(res.pv_portfolio);
            if (keep_paths) {
                py::array_t<double> paths(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_scenarios), res.horizon});
                std::copy(res.k_paths.begin(), res.k_paths.end(), paths.mutable_data());
                d["k_paths"] = paths;
            }
            return d;
        },
        py::arg("age_min"), py::arg("ax"), py::arg("bx"), py::arg("k_last"), py::arg("drift"), py::arg("sigma"),
        py::arg("drift_se"), py::arg("age0"), py::arg("discount"), py::arg("n_scenarios"), py::arg("n_lives") = 0,
        py::arg("seed") = 12345, py::arg("n_threads") = 0, py::arg("keep_paths") = false,
        "Monte Carlo valuation of a life annuity-due under Lee-Carter with a random walk with drift.");

    m.def(
        "one_year_recalibration",
        [](int age_min, const DoubleArray& ax, const DoubleArray& bx, double k_last, double drift, double sigma,
           double drift_se, double k_first, int n_increments, int age0, const DoubleArray& discount,
           long long n_scenarios, std::uint64_t seed, int n_threads) {
            const lr::LeeCarter lc = make_model(age_min, ax, bx, k_last, drift, sigma, drift_se);
            const auto v = to_vector(discount);
            lr::OneYearRecalibration res;
            {
                py::gil_scoped_release release;
                res = lr::one_year_recalibration(lc, k_first, n_increments, age0, v, n_scenarios, seed, n_threads);
            }
            py::dict d;
            d["best_estimate"] = res.best_estimate;
            d["value"] = to_array(res.value);
            d["k_next"] = to_array(res.k_next);
            d["drift_next"] = to_array(res.drift_next);
            return d;
        },
        py::arg("age_min"), py::arg("ax"), py::arg("bx"), py::arg("k_last"), py::arg("drift"), py::arg("sigma"),
        py::arg("drift_se"), py::arg("k_first"), py::arg("n_increments"), py::arg("age0"), py::arg("discount"),
        py::arg("n_scenarios"), py::arg("seed") = 12345, py::arg("n_threads") = 0,
        "One-year value-at-risk view of longevity trend risk with re-estimation of the drift.");
}
