from __future__ import annotations

import json

from dcft_analysis.cli import main


def test_synthetic_end_to_end(synthetic_writer, tmp_path) -> None:
    root = tmp_path / "L4x6"
    for noise in ("z", "zz"):
        for p_tag in ("p000", "p025"):
            synthetic_writer(root, noise, p_tag)
    out = tmp_path / "campaign-analysis"

    result = main(["--root", str(root), "--out", str(out)])

    assert result == 0
    assert (out / "tables" / "scalar_observables.csv").is_file()
    assert (out / "tables" / "correlators.csv").is_file()
    manifest = json.loads((out / "summary.json").read_text())
    assert manifest["discovered_point_count"] == 4
    assert manifest["metadata"] == {"lx": 4, "lt": 6, "samples_per_point": 12}
    assert manifest["uncertainty_estimates"] is False
