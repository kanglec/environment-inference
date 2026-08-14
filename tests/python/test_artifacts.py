from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pytest

from dcft.artifacts import (
    ArtifactError,
    load_artifact,
    read_table,
    source_digest,
    verify_artifact,
    write_artifact,
)


def test_source_digest_is_self_contained(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    package = checkout / "python" / "dcft"
    package.mkdir(parents=True)
    source = package / "model.py"
    source.write_text("VALUE = 1\n")
    sibling_notes = tmp_path / "notes"
    sibling_notes.mkdir()
    (sibling_notes / "main.tex").write_text("not part of the simulation repository")

    digest = source_digest(checkout)
    (sibling_notes / "main.tex").write_text("changed external notes")
    (checkout / "README.md").write_text("documentation")
    configs = checkout / "configs"
    configs.mkdir()
    (configs / "request.toml").write_text("outer_records = 10\n")

    assert source_digest(checkout) == digest
    source.write_text("VALUE = 2\n")
    assert source_digest(checkout) != digest


def test_partitioned_parquet_round_trip_is_content_addressed(
    tmp_path: Path, project_root: Path
) -> None:
    schema = pa.schema(
        [
            pa.field("global_id", pa.uint64(), nullable=False),
            pa.field("noise", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
        ]
    )
    rows = [
        {"global_id": 2, "noise": "zz", "value": 3.0},
        {"global_id": 0, "noise": "z", "value": 1.0},
        {"global_id": 1, "noise": "z", "value": 2.0},
    ]
    table = pa.Table.from_pylist(rows, schema=schema)
    first = write_artifact(
        tmp_path,
        "test",
        table,
        metadata={"purpose": "roundtrip"},
        project_root=project_root,
        partition_by=("noise",),
    )
    second = write_artifact(
        tmp_path,
        "test",
        table,
        metadata={"purpose": "roundtrip"},
        project_root=project_root,
        partition_by=("noise",),
    )
    assert first.artifact_id == second.artifact_id
    assert (
        read_table(first).sort_by("global_id").to_pylist() == table.sort_by("global_id").to_pylist()
    )
    assert verify_artifact(first.path)["status"] == "valid"
    manifest = load_artifact(first.path).manifest
    assert manifest["rng_contract"]["stream_key_version"] == "dcft-stream-v1"
    assert manifest["parents"] == []
    assert len(manifest["source_digest"]) == 64


def test_archived_formats_have_no_compatibility_reader(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"schema_version":"DCFT_AGG_V3"}')
    with pytest.raises(ArtifactError, match="archived formats have no reader"):
        load_artifact(tmp_path)


def test_parquet_bytes_are_reproducible_across_input_chunking(
    tmp_path: Path, project_root: Path
) -> None:
    schema = pa.schema(
        [
            pa.field("global_id", pa.uint64(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
        ]
    )
    rows = [{"global_id": index, "value": index / 7.0} for index in range(25)]
    contiguous = pa.Table.from_pylist(rows, schema=schema)
    chunked = pa.concat_tables(
        [
            pa.Table.from_pylist(rows[:6], schema=schema),
            pa.Table.from_pylist(rows[6:19], schema=schema),
            pa.Table.from_pylist(rows[19:], schema=schema),
        ]
    )
    left = write_artifact(
        tmp_path / "left",
        "bytes",
        contiguous,
        metadata={"threads": "independent"},
        project_root=project_root,
    )
    right = write_artifact(
        tmp_path / "right",
        "bytes",
        chunked,
        metadata={"threads": "independent"},
        project_root=project_root,
    )
    assert left.artifact_id == right.artifact_id
    assert left.manifest["checksums"] == right.manifest["checksums"]
    assert (
        next((left.path / "data").glob("*.parquet")).read_bytes()
        == next((right.path / "data").glob("*.parquet")).read_bytes()
    )


def test_concurrent_identical_artifact_commit_is_idempotent(
    tmp_path: Path, project_root: Path
) -> None:
    table = pa.Table.from_pylist(
        [{"global_id": index, "value": index / 11.0} for index in range(32)]
    )

    def commit() -> str:
        return write_artifact(
            tmp_path,
            "concurrent",
            table,
            metadata={"purpose": "concurrent-idempotence"},
            project_root=project_root,
        ).artifact_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        identifiers = list(executor.map(lambda _: commit(), range(8)))
    assert len(set(identifiers)) == 1
    artifact = load_artifact(tmp_path / "concurrent" / identifiers[0])
    assert verify_artifact(artifact.path)["status"] == "valid"
