from __future__ import annotations

import pandas as pd
import pytest

from dcft_analysis.campaign import (
    CampaignValidationError,
    discover_points,
    load_campaign,
    parse_expected_p,
    parse_p_tag,
    read_point_csv,
)


def test_parse_p_tag_and_expected_p() -> None:
    assert parse_p_tag("p000") == (0, 0.0)
    assert parse_p_tag("p025") == (25, 0.25)
    assert parse_p_tag("p049") == (49, 0.49)
    assert parse_expected_p(".05") == 5
    with pytest.raises(ValueError, match="p000 through p049"):
        parse_p_tag("p050")
    with pytest.raises(ValueError, match="increments of 0.01"):
        parse_expected_p("0.055")


def test_recursive_discovery(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "L4x6"
    first = synthetic_writer(root, "z", "p000")
    second = synthetic_writer(root, "zz", "p025")
    points, problems = discover_points(root)
    assert problems == []
    assert [(point.noise, point.p_tag) for point in points] == [("z", "p000"), ("zz", "p025")]
    assert {point.path for point in points} == {first, second}


def test_schema_validation_reports_missing_column(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "L4x6"
    path = synthetic_writer(root)
    frame = pd.read_csv(path).drop(columns="bond_ea_corr")
    frame.to_csv(path, index=False)
    point = discover_points(root)[0][0]
    with pytest.raises(CampaignValidationError, match="missing=.*bond_ea_corr"):
        read_point_csv(point, root)


def test_duplicate_r_is_rejected(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "L4x6"
    path = synthetic_writer(root)
    frame = pd.read_csv(path)
    frame.loc[2, "r"] = 1
    frame.to_csv(path, index=False)
    point = discover_points(root)[0][0]
    with pytest.raises(CampaignValidationError, match="duplicate r"):
        read_point_csv(point, root)


def test_scalar_extraction_and_physics_names(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "L4x6"
    synthetic_writer(root, "z", "p005")
    campaign = load_campaign(root)
    row = campaign.scalars.iloc[0]
    assert row["E_D"] == pytest.approx(-1.1)
    assert row["M"] == pytest.approx(0.02)
    assert row["M_partial"] == pytest.approx(0.03)
    assert row["F_sigma_loc"] == pytest.approx(0.2)
    assert row["F_epsilon_loc"] == pytest.approx(0.7)
    assert set(campaign.correlators.columns) == {
        "noise",
        "p",
        "p_tag",
        "r",
        "C_sigma_lin",
        "C_epsilon_lin",
        "F_sigma",
        "F_epsilon",
        "C_sigma_EA",
        "C_epsilon_EA",
    }


def test_missing_cross_product_is_strict_by_default(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "L4x6"
    synthetic_writer(root, "z", "p000")
    synthetic_writer(root, "z", "p005")
    synthetic_writer(root, "zz", "p000")
    with pytest.raises(CampaignValidationError, match=r"\(zz, p005\)"):
        load_campaign(root)

    campaign = load_campaign(root, allow_incomplete=True)
    assert campaign.missing_points == [("zz", "p005")]
    assert campaign.warnings


def test_existing_point_directory_without_csv_is_recorded(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "L4x6"
    synthetic_writer(root, "z", "p000")
    (root / "z" / "p005" / "analysis").mkdir(parents=True)

    with pytest.raises(CampaignValidationError, match="missing analysis CSV"):
        load_campaign(root)
    campaign = load_campaign(root, allow_incomplete=True)
    assert campaign.missing_points == [("z", "p005")]


def test_metadata_must_be_consistent_across_points(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "campaign"
    synthetic_writer(root, "z", "p000", samples=12)
    synthetic_writer(root, "z", "p005", samples=13)
    with pytest.raises(CampaignValidationError, match="inconsistent metadata"):
        load_campaign(root)
