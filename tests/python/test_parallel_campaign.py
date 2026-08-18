from __future__ import annotations

from pathlib import Path

from dcft.analysis import analyze_campaign
from dcft.artifacts import discover_artifacts, read_table
from dcft.campaign import _chain_schedule, _packed_initial_configuration, run_campaign
from dcft.config import load_config
from dcft.planning import read_state


def test_chain_schedule_separates_production_budget_and_convergence(
    smoke_config_path: Path,
) -> None:
    config = load_config(smoke_config_path)
    schedule = _chain_schedule(config, 0)
    assert schedule[0].role == "production"
    assert schedule[0].initialization == "planted"
    assert sum(chain.role == "finite-inner" for chain in schedule) == 5
    convergence = [chain for chain in schedule if chain.role == "convergence"]
    assert [chain.initialization for chain in convergence] == ["all-plus", "all-minus"]
    assert _chain_schedule(config, config.mc.diagnostic_outer_records) == (schedule[0],)

    planted = bytes([0x5A])
    plus = _packed_initial_configuration(
        4, 2, "all-plus", seed=7, global_id=0, stream_label="plus", planted=planted
    )
    minus = _packed_initial_configuration(
        4, 2, "all-minus", seed=7, global_id=0, stream_label="minus", planted=planted
    )
    assert plus == bytes([0x00])
    assert minus == bytes([0xFF])
    assert _packed_initial_configuration(
        3, 3, "all-minus", seed=7, global_id=0, stream_label="minus", planted=bytes(2)
    ) == bytes([0xFF, 0x01])


def test_chunked_campaign_runs_in_rayon_and_merges_complete_ids(
    smoke_config_path: Path, tmp_path: Path
) -> None:
    source = smoke_config_path.read_text()
    replacements = {
        'noises = ["z", "zz"]': 'noises = ["z"]',
        'measurements = ["heterodyne", "homodyne", "local-x"]': (
            'measurements = ["heterodyne"]'
        ),
        "gaussian_fractions = [0.5]": "gaussian_fractions = []",
        "outer_records = 16": "outer_records = 4",
        "clean_thermalization_sweeps = 100": "clean_thermalization_sweeps = 8",
        "clean_saving_interval = 10": "clean_saving_interval = 2",
        "posterior_decorrelation_gap = 20": "posterior_decorrelation_gap = 2",
        "inner_measurements = 32": "inner_measurements = 4",
        "inner_saving_interval = 2": "inner_saving_interval = 1",
        'updates = ["metropolis", "corrected-wolff", "tnmc", "tnmc-global"]': (
            'updates = ["metropolis"]'
        ),
        "diagnostic_outer_records = 4": "diagnostic_outer_records = 2",
        "enabled = true\nmax_sites = 4": "enabled = false\nmax_sites = 4",
    }
    for before, after in replacements.items():
        source = source.replace(before, after)
    config_path = tmp_path / "parallel-smoke.toml"
    config_path.write_text(source)
    config = load_config(config_path)

    result = run_campaign(config, workers=2)
    assert result == {"completed": 4, "skipped": 0, "workers": 2}
    state = read_state(config)
    assert {item["status"] for item in state["tasks"].values()} == {"pending", "complete"}

    artifacts = discover_artifacts(config.campaign.output_root)
    chunks = [artifact for artifact in artifacts if artifact.manifest["kind"] == "mc-chunk"]
    merged = [artifact for artifact in artifacts if artifact.manifest["kind"] == "mc-records"]
    assert len(chunks) == 2
    assert len(merged) == 1
    rows = read_table(merged[0]).to_pylist()
    assert {row["chain_role"] for row in rows} == {
        "production",
        "finite-inner",
        "convergence",
    }
    assert {row["initialization"] for row in rows if row["chain_role"] == "convergence"} == {
        "all-plus",
        "all-minus",
    }
    production_ids = [
        row["global_id"]
        for row in rows
        if row["chain_role"] == "production"
    ]
    assert production_ids == [0, 1, 2, 3]
    analysis = analyze_campaign(config)
    assert analysis["status"] == "complete"
    analyzed_kinds = {
        artifact.manifest["kind"]
        for artifact in discover_artifacts(config.campaign.output_root)
    }
    assert "analysis-diagnostics" in analyzed_kinds
    assert "analysis-outer-autocorrelation" in analyzed_kinds
