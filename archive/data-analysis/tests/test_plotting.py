from __future__ import annotations

from dcft_analysis.campaign import load_campaign
from dcft_analysis.plotting import plot_campaign


def test_plotting_smoke(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "L4x6"
    synthetic_writer(root, "z", "p000")
    synthetic_writer(root, "z", "p005")
    campaign = load_campaign(root)
    out = tmp_path / "analysis-out"

    artifacts = plot_campaign(campaign.scalars, campaign.correlators, out)

    assert "figures/scalar_observables_vs_p.png" in artifacts
    assert "figures/spin_correlators_vs_r_z.pdf" in artifacts
    assert "figures/linear_p_independence_check_z.png" in artifacts
    assert all((out / artifact).stat().st_size > 0 for artifact in artifacts)
