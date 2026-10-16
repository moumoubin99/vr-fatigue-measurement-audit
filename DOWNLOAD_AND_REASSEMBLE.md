# Download and reconstruct original archives

The official Figshare data versions in `DATA_SOURCES.md` are available as the primary download route. The [raw-data-v1 Release](https://github.com/moumoubin99/vr-fatigue-measurement-audit/releases/tag/raw-data-v1) is an additional route for the same archives. It contains 18 binary parts and the attached `raw-data-v1.manifest.json`. Both archives passed local original-size/MD5 checks before splitting; every part matched GitHub API size and server SHA-256. The repository manifest has `status: ready`. These checks do not mean that all data were downloaded again from GitHub or that study analyses were rerun.

## Direct official download

1. Open the version DOI in `DATA_SOURCES.md` and download the archive with the exact listed filename.
2. Verify its byte count and MD5 against that table before extracting or analyzing it. MD5 is used to match the upstream file identity; it is not an authenticity guarantee on its own.
3. Keep raw archives outside the Git checkout, or under an ignored local `raw/` directory. The reconstruction tool also supports checking an intact archive:

```sh
python tools/reassemble_archives.py --manifest release-manifests/raw-data-v1.json --archive gazebasevr.zip --verify-existing raw/gazebasevr.zip
python tools/reassemble_archives.py --manifest release-manifests/raw-data-v1.json --archive GazeBase_v2_0.zip --verify-existing raw/GazeBase_v2_0.zip
```

## Split-release route

Download the published `raw-data-v1.manifest.json` asset or use the matching `release-manifests/raw-data-v1.json` from the current repository. The manifest records every actual part in reconstruction order, with its byte count, SHA-256, and download URL. The Git tag is the code snapshot at release creation; the attached manifest is the binary asset index. A later main-branch documentation commit does not move the tag.

Download every listed part for one archive into a single local `downloads/` directory. Preserve exact names. Do not extract parts individually. The manifest order determines concatenation order; the tool does not infer it from directory listings.

```sh
python tools/reassemble_archives.py --manifest release-manifests/raw-data-v1.json --archive gazebasevr.zip --parts-dir downloads --output raw/gazebasevr.zip
python tools/reassemble_archives.py --manifest release-manifests/raw-data-v1.json --archive GazeBase_v2_0.zip --parts-dir downloads --output raw/GazeBase_v2_0.zip
```

The tool uses bounded streaming I/O, checks each part's bytes and SHA-256, then checks the reconstructed archive's total bytes and upstream MD5. It never downloads files, executes extraction, runs an experiment, or trains a model. It refuses an existing output path. A failed reconstruction removes only the newly created incomplete output. Allow storage for both the downloaded parts and the reconstructed archive. A successful checksum match establishes byte identity with the listed upstream archive, not successful study reproduction.

The ordered local part streams were independently concatenated into hashes without extraction: both total byte counts and original upstream MD5 values matched. No second full archive copy or study rerun was needed for this reconstruction-integrity check.

## Analysis layout

Extraction and study execution are separate, optional operations. Before doing either, read `analysis/README_REPRO.md`. That guide describes the expected data tree and distinguishes recomputation from selected frozen derived inputs from rebuilding historical processing outputs. Use a separate working copy for any new study run. The original datasets may contain material beyond the task subsets used here; preserve the upstream license and attribution alongside reused material.
