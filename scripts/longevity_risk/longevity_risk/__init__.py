"""Stochastic mortality and longevity risk for life annuities.

The compiled extension ``longevity_risk._core`` simulates Lee-Carter
mortality scenarios and values life annuities by Monte Carlo. The Python
modules fit the Lee-Carter model (SVD and Poisson maximum likelihood), build
discount curves (ECB Svensson curve with Smith-Wilson extrapolation),
compute the Solvency II standard-formula longevity shock, provide NumPy
reference implementations and download official Eurostat and ECB data.
"""

__version__ = "0.1.0"

try:
    from . import _core  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the local build
    _core = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

HAS_CPP = _core is not None


def require_cpp():
    """Return the compiled extension or raise an informative error."""
    if _core is None:
        raise ImportError(
            "The C++ extension longevity_risk._core is not built. From the repository root run "
            "`python -m pip install -e scripts/longevity_risk` (a C++17 compiler is required)."
        ) from _IMPORT_ERROR
    return _core
