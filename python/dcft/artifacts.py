"""Immutable content-addressed Parquet artifacts and JSON manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq

from . import QUALIFICATION_STATUS, SCHEMA_VERSION, __version__, _core


class ArtifactError(RuntimeError):
    """An artifact is incomplete, corrupt, or incompatible."""


@dataclass(frozen=True)
class Artifact:
    path: Path
    manifest: dict[str, Any]

    @property
    def artifact_id(self) -> str:
        return cast(str, self.manifest["artifact_id"])


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_hash(configuration: bytes) -> str:
    return sha256_bytes(configuration)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        raise ArtifactError("artifact metadata and rows must not contain NaN or infinity")
    return value


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def source_digest(project_root: Path) -> str:
    """Digest runtime-affecting source and locks, independent of campaign inputs."""
    root = project_root.resolve()
    candidates = [
        root / name
        for name in (
            ".python-version",
            "Cargo.lock",
            "Cargo.toml",
            "pyproject.toml",
            "rust-toolchain.toml",
            "uv.lock",
        )
        if (root / name).is_file()
    ]
    for source_root, suffixes in ((root / "python", {".py", ".pyi"}), (root / "src", {".rs"})):
        if source_root.is_dir():
            candidates.extend(
                path
                for path in source_root.rglob("*")
                if path.is_file() and path.suffix in suffixes
            )
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: str(item)):
        label = str(path.relative_to(root))
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sort_table(table: pa.Table) -> pa.Table:
    preferred = [
        name
        for name in (
            "global_id",
            "lx",
            "noise",
            "p",
            "protocol_id",
            "prior",
            "observable",
            "separation",
            "update",
            "inner_budget_multiplier",
            "replica",
        )
        if name in table.column_names
    ]
    return table.sort_by([(name, "ascending") for name in preferred]) if preferred else table


def _artifact_id(
    kind: str,
    table: pa.Table,
    metadata: Mapping[str, Any],
    parents: Iterable[str],
    code_digest: str,
) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "kind": kind,
        "source_digest": code_digest,
        "metadata": metadata,
        "parents": sorted(parents),
        "rows": table.to_pylist(),
    }
    return sha256_bytes(canonical_json(payload))[:24]


def _partition_groups(
    table: pa.Table, columns: tuple[str, ...]
) -> list[tuple[tuple[Any, ...], pa.Table]]:
    if not columns:
        return [((), table)]
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        raise ArtifactError(f"partition columns absent from table: {missing}")
    rows = table.to_pylist()
    keys = sorted({tuple(row[name] for name in columns) for row in rows}, key=repr)
    groups: list[tuple[tuple[Any, ...], pa.Table]] = []
    for key in keys:
        selected = [row for row in rows if tuple(row[name] for name in columns) == key]
        groups.append((key, pa.Table.from_pylist(selected, schema=table.schema)))
    return groups


def _write_parquet(table: pa.Table, path: Path) -> None:
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="2.0",
        version="2.6",
        row_group_size=max(1, min(65_536, table.num_rows)),
    )


def write_artifact(
    root: Path,
    kind: str,
    table: pa.Table,
    *,
    metadata: Mapping[str, Any],
    project_root: Path,
    parents: Iterable[str] = (),
    partition_by: Iterable[str] = (),
) -> Artifact:
    """Atomically create an immutable artifact or reuse an identical one."""
    if table.num_rows == 0:
        raise ArtifactError("scientific artifacts cannot contain an empty table")
    normalized = _sort_table(table.combine_chunks())
    parent_ids = tuple(sorted(set(parents)))
    code_digest = source_digest(project_root)
    artifact_id = _artifact_id(kind, normalized, metadata, parent_ids, code_digest)
    destination = root.resolve() / kind / artifact_id
    if destination.exists():
        artifact = load_artifact(destination)
        verify_artifact(artifact.path)
        return artifact

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (f".{artifact_id}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    data_root = temporary / "data"
    data_root.mkdir()
    partition_columns = tuple(partition_by)
    try:
        files: list[Path] = []
        for part_number, (key, group) in enumerate(
            _partition_groups(normalized, partition_columns)
        ):
            directory = data_root
            for name, value in zip(partition_columns, key, strict=True):
                directory /= f"{name}={quote(str(value), safe='._-')}"
            directory.mkdir(parents=True, exist_ok=True)
            part = directory / f"part-{part_number:05d}.parquet"
            _write_parquet(group, part)
            files.append(part)

        checksums = {
            str(path.relative_to(temporary)): sha256_file(path)
            for path in sorted(files, key=lambda item: str(item))
        }
        global_ids: list[int] = []
        if "global_id" in normalized.column_names:
            global_ids = [int(value) for value in normalized["global_id"].to_pylist()]
        config_hashes: list[str] = []
        for column in ("configuration_hash", "planted_configuration_hash"):
            if column in normalized.column_names:
                config_hashes.extend(str(value) for value in normalized[column].to_pylist())
        rng_algorithm, key_version = _core.rng_contract()
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "kind": kind,
            "status": "complete",
            "qualification": QUALIFICATION_STATUS,
            "package_version": __version__,
            "rust_core_version": _core.version(),
            "source_digest": code_digest,
            "parents": list(parent_ids),
            "partition_columns": list(partition_columns),
            "row_count": normalized.num_rows,
            "arrow_schema": normalized.schema.to_string(show_field_metadata=True),
            "global_id_range": {
                "start": min(global_ids) if global_ids else None,
                "stop_exclusive": max(global_ids) + 1 if global_ids else None,
                "unique_count": len(set(global_ids)),
            },
            "configuration_hash_count": len(config_hashes),
            "configuration_hash_digest": sha256_bytes(canonical_json(config_hashes)),
            "rng_contract": {
                "algorithm": rng_algorithm,
                "stream_key_version": key_version,
                "key_fields": ["base_seed", "domain_label", "global_id"],
            },
            "metadata": dict(metadata),
            "checksums": checksums,
        }
        (temporary / "manifest.json").write_bytes(canonical_json(manifest))
        try:
            temporary.rename(destination)
        except OSError:
            # Another worker may have committed the identical content-addressed
            # artifact after our initial existence check.  Reuse it only after
            # full validation; otherwise preserve the original error.
            if not destination.exists():
                raise
            shutil.rmtree(temporary)
            artifact = load_artifact(destination)
            verify_artifact(destination)
            return artifact
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return Artifact(destination, manifest)


def load_artifact(path: Path) -> Artifact:
    resolved = path.expanduser().resolve()
    manifest_path = resolved / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"missing artifact manifest: {manifest_path}")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"unsupported artifact schema {manifest.get('schema_version')!r}; archived formats have no reader"
        )
    return Artifact(resolved, manifest)


def read_table(artifact: Artifact | Path) -> pa.Table:
    item = load_artifact(artifact) if isinstance(artifact, Path) else artifact
    files = sorted((item.path / "data").rglob("*.parquet"))
    if not files:
        raise ArtifactError(f"artifact has no Parquet parts: {item.path}")
    # ``pq.read_table`` performs dataset discovery and interprets the Hive
    # parent directories, which duplicates partition columns already retained
    # in each immutable part. Reading each file directly avoids that coercion.
    tables = [pq.ParquetFile(path).read() for path in files]
    return pa.concat_tables(tables, promote_options="none").combine_chunks()


def verify_artifact(path: Path) -> dict[str, Any]:
    artifact = load_artifact(path)
    problems: list[str] = []
    if artifact.manifest.get("status") != "complete":
        problems.append("manifest status is not complete")
    checksums = cast(dict[str, str], artifact.manifest.get("checksums", {}))
    for relative, expected in checksums.items():
        candidate = artifact.path / relative
        if not candidate.is_file():
            problems.append(f"missing file {relative}")
        elif sha256_file(candidate) != expected:
            problems.append(f"checksum mismatch for {relative}")
    try:
        table = read_table(artifact)
    except Exception as error:  # pyarrow errors vary by release.
        problems.append(f"cannot read Parquet dataset: {error}")
    else:
        if table.num_rows != artifact.manifest.get("row_count"):
            problems.append("row count differs from manifest")
        if "global_id" in table.column_names:
            ids = [int(value) for value in table["global_id"].to_pylist()]
            described = artifact.manifest.get("global_id_range", {})
            if len(set(ids)) != described.get("unique_count"):
                problems.append("global id unique count differs from manifest")
    if problems:
        raise ArtifactError("; ".join(problems))
    return {"artifact_id": artifact.artifact_id, "status": "valid", "files": len(checksums)}


def discover_artifacts(root: Path, kind: str | None = None) -> list[Artifact]:
    base = root.expanduser().resolve()
    search = base / kind if kind else base
    if not search.exists():
        return []
    artifacts: list[Artifact] = []
    for path in sorted(search.rglob("manifest.json")):
        try:
            header = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if header.get("schema_version") == SCHEMA_VERSION and "artifact_id" in header:
            artifacts.append(load_artifact(path.parent))
    return artifacts
