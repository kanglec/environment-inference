from __future__ import annotations

from pathlib import Path

from dcft.artifacts import discover_artifacts, read_table
from dcft.campaign import run_campaign
from dcft.config import load_config
from dcft.planning import read_state


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
        'updates = ["metropolis", "corrected-wolff", "tnmc"]': 'updates = ["metropolis"]',
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
    production_ids = [
        row["global_id"]
        for row in rows
        if row["inner_budget_multiplier"] == 1 and row["replica"] == 0
    ]
    assert production_ids == [0, 1, 2, 3]
