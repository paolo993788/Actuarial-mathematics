"""The README figures render in both themes on simulated mortality."""

import matplotlib

matplotlib.use("Agg")

from longevity_risk import readme_figures  # noqa: E402
from longevity_risk.figstyle import render  # noqa: E402


def test_all_readme_figures_render_in_both_themes(tmp_path):
    inputs = {"national": readme_figures.load(official=False, n_scenarios=500),
              "multipopulation": readme_figures.load_multipopulation(official=False)}
    assert set(inputs) == set(readme_figures.LOADERS)
    for name, (builder, key) in readme_figures.FIGURES.items():
        paths = render(builder, name, tmp_path, inputs[key])
        assert [p.name for p in paths] == [f"{name}-light.png", f"{name}-dark.png"]
        assert all(p.stat().st_size > 10_000 for p in paths)
