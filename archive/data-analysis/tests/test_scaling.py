from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dcft_analysis.campaign import CampaignValidationError
from dcft_analysis.scaling import discover_campaign_tables, load_scaling_analysis
from dcft_analysis.scaling_cli import main


def test_scaling_discovery_and_derived_distances(scaling_writer, tmp_path) -> None:
    scaling_writer(tmp_path, 4, analysis_name="analysis")
    scaling_writer(tmp_path, 8)

    pairs = discover_campaign_tables(tmp_path)
    analysis = load_scaling_analysis(tmp_path)

    assert len(pairs) == 2
    assert analysis.sizes == [4, 8]
    assert analysis.long_distance.groupby("L")["r_max"].first().to_dict() == {4: 2, 8: 4}
    row = analysis.correlators.query("L == 4 and r == 1").iloc[0]
    assert row["r_over_L"] == pytest.approx(0.25)
    assert row["chord_distance"] == pytest.approx(4 / np.pi * np.sin(np.pi / 4))


def test_power_law_fit_recovers_synthetic_dimension(scaling_writer, tmp_path) -> None:
    for L in (4, 8, 16):
        scaling_writer(tmp_path, L, decay_exponent=0.6)
    analysis = load_scaling_analysis(tmp_path)

    fit = analysis.fits.query("noise == 'z' and observable == 'C_sigma_lin'").iloc[0]
    assert fit["fit_status"] == "ok"
    assert fit["decay_exponent"] == pytest.approx(0.6)
    assert fit["scaling_dimension"] == pytest.approx(0.3)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["n_sizes_fit"] == 3


def test_scaling_rejects_rectangular_campaign_output(scaling_writer, tmp_path) -> None:
    scalar_path, _ = scaling_writer(tmp_path, 4)
    frame = pd.read_csv(scalar_path)
    frame["lt"] = 6
    frame.to_csv(scalar_path, index=False)

    with pytest.raises(CampaignValidationError, match="requires square systems"):
        load_scaling_analysis(tmp_path)


def test_scaling_requires_multiple_sizes(scaling_writer, tmp_path) -> None:
    scaling_writer(tmp_path, 8)
    with pytest.raises(CampaignValidationError, match="at least two distinct sizes"):
        load_scaling_analysis(tmp_path)


def test_scaling_grid_is_strict_by_default(scaling_writer, tmp_path) -> None:
    scaling_writer(tmp_path, 4, p_values=(0.0, 0.05))
    scaling_writer(tmp_path, 8, p_values=(0.0,))

    with pytest.raises(CampaignValidationError, match=r"L=8, z, p005"):
        load_scaling_analysis(tmp_path)
    analysis = load_scaling_analysis(tmp_path, allow_incomplete=True)
    assert analysis.missing_points == [(8, "z", "p005")]
    assert analysis.warnings


def test_scaling_cli_synthetic_end_to_end(scaling_writer, tmp_path) -> None:
    for L in (4, 8, 16):
        scaling_writer(tmp_path, L, noises=("z", "zz"), p_values=(0.0, 0.25))
    out = tmp_path / "scaling-analysis"

    result = main(["--root", str(tmp_path), "--out", str(out)])

    assert result == 0
    for name in (
        "scalar_observables_by_size.csv",
        "correlators_by_size.csv",
        "long_distance_observables.csv",
        "power_law_fits.csv",
    ):
        assert (out / "tables" / name).is_file()
    assert (out / "figures" / "long_distance_vs_p_z.png").stat().st_size > 0
    assert (out / "figures" / "chord_distance_spin_zz_p025.pdf").stat().st_size > 0
    manifest = json.loads((out / "summary.json").read_text())
    assert manifest["sizes"] == [4, 8, 16]
    assert manifest["point_count"] == 12
    assert manifest["uncertainty_estimates"] is False
