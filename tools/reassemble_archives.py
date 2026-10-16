#!/usr/bin/env python3
"""Stream split assets into an original archive and verify upstream identity.

Only the Python standard library is used. No download, extraction, or analysis
is performed. A destination that already exists is never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


CHUNK_SIZE = 8 * 1024 * 1024


def integer_size(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def digest_value(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value):
        raise ValueError(f"{label} must contain {length} hexadecimal characters")
    return value.lower()


def load_archive(manifest_path, archive_name):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version")
    matches = [entry for entry in manifest.get("archives", [])
               if entry.get("file_name") == archive_name]
    if len(matches) != 1:
        raise ValueError("Select exactly one archive listed in the manifest")
    archive = matches[0]
    integer_size(archive.get("bytes"), "archive bytes")
    digest_value(archive.get("md5"), 32, "archive MD5")
    return manifest, archive


def new_md5():
    # MD5 matches the upstream archive identifier, rather than providing security.
    try:
        return hashlib.md5(usedforsecurity=False)
    except TypeError:
        return hashlib.md5()


def verify_existing(path, archive):
    total = 0
    digest = new_md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            total += len(chunk)
            digest.update(chunk)
    if total != archive["bytes"]:
        raise ValueError(f"Archive size mismatch: got {total}, expected {archive['bytes']}")
    if digest.hexdigest() != archive["md5"].lower():
        raise ValueError("Archive MD5 does not match the original upstream file")
    return total, digest.hexdigest()


def checked_parts(manifest, archive, parts_dir):
    if manifest.get("status") != "ready":
        raise ValueError("Split release is not ready; use the official source or a completed manifest")
    parts = archive.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Archive has no completed split-part list")
    seen = set()
    checked = []
    base = parts_dir.resolve(strict=True)
    for item in parts:
        name = item.get("name")
        if (not isinstance(name, str) or not name or name in (".", "..")
                or "/" in name or "\\" in name or ":" in name):
            raise ValueError("Part names must be plain filenames, without directory components")
        if name in seen:
            raise ValueError(f"Duplicate part name: {name}")
        seen.add(name)
        size = integer_size(item.get("bytes"), f"{name} bytes")
        sha256 = digest_value(item.get("sha256"), 64, f"{name} SHA-256")
        path = base / name
        resolved = path.resolve(strict=True)
        if resolved.parent != base or not resolved.is_file():
            raise ValueError(f"Part escapes the selected directory or is not a file: {name}")
        if resolved.stat().st_size != size:
            raise ValueError(f"Part size mismatch: {name}")
        checked.append((resolved, size, sha256))
    if sum(item[1] for item in checked) != archive["bytes"]:
        raise ValueError("Part byte counts do not sum to the expected original archive size")
    return checked


def reassemble(manifest, archive, parts_dir, output):
    parts = checked_parts(manifest, archive, parts_dir)
    if any(output.resolve() == path for path, _, _ in parts):
        raise ValueError("Output cannot be an input part")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    total = 0
    full_md5 = new_md5()
    try:
        # Exclusive creation prevents accidental replacement of any existing file.
        with output.open("xb") as destination:
            created = True
            for path, expected_size, expected_sha256 in parts:
                part_size = 0
                part_digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                        destination.write(chunk)
                        part_size += len(chunk)
                        total += len(chunk)
                        part_digest.update(chunk)
                        full_md5.update(chunk)
                if part_size != expected_size or part_digest.hexdigest() != expected_sha256:
                    raise ValueError(f"Part verification failed: {path.name}")
                print(f"Verified part: {path.name}", flush=True)
            if total != archive["bytes"] or full_md5.hexdigest() != archive["md5"].lower():
                raise ValueError("Reconstructed archive fails original size/MD5 verification")
    except BaseException:
        # Only this invocation's newly created output may be removed.
        if created:
            output.unlink(missing_ok=True)
        raise
    return total, full_md5.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--archive", required=True, help="Exact original filename from the manifest")
    parser.add_argument("--verify-existing", type=Path, help="Verify a complete original archive only")
    parser.add_argument("--parts-dir", type=Path, help="Directory containing every listed split part")
    parser.add_argument("--output", type=Path, help="New output file; existing paths are refused")
    args = parser.parse_args(argv)
    if args.verify_existing is not None:
        if args.parts_dir is not None or args.output is not None:
            parser.error("--verify-existing cannot be combined with --parts-dir or --output")
    elif args.parts_dir is None or args.output is None:
        parser.error("Reassembly requires both --parts-dir and --output")
    manifest, archive = load_archive(args.manifest, args.archive)
    if args.verify_existing is not None:
        total, digest = verify_existing(args.verify_existing, archive)
        destination = args.verify_existing
    else:
        total, digest = reassemble(manifest, archive, args.parts_dir, args.output)
        destination = args.output
    print(f"PASS: {destination.name}; {total} bytes; original MD5 {digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
