# Download and reconstruct original archives

The official Figshare data versions in `DATA_SOURCES.md` are available as the primary download route. A GitHub Release tagged `raw-data-v1` is planned as an additional route for the same archives. Its split assets have not been uploaded during this repository-preparation step. Empty part lists and `status: planned` in the manifest are deliberate; the reconstruction tool refuses an unfinished list.

## Direct official download

1. Open the version DOI in `DATA_SOURCES.md` and download the archive with the exact listed filename.
2. Verify its byte count and MD5 against that table before extracting or analyzing it. MD5 is used to match the upstream file identity; it is not an authenticity guarantee on its own.
3. Keep raw archives outside the Git checkout, or under an ignored local `raw/` directory. The reconstruction tool also supports checking an intact archive:

```sh
python tools/reassemble_archives.py --manifest release-manifests/raw-data-v1.json --archive gazebasevr.zip --verify-existing raw/gazebasevr.zip
python tools/reassemble_archives.py --manifest release-manifests/raw-data-v1.json --archive GazeBase_v2_0.zip --verify-existing raw/GazeBase_v2_0.zip
```

## Planned split-release route

After release publication, the maintainer must replace the empty part arrays in `release-manifests/raw-data-v1.json` with the actual ordered part records and change `status` to `ready`. Each record has this structure; the values below are a schema illustration, not an actual asset:

```json
{"name": "archive.zip.part001", "bytes": 123, "sha256": "64 hexadecimal characters", "download_url": "actual published asset URL"}
```

Download every listed part for one archive into a single local `downloads/` directory. Preserve exact names. Do not extract parts individually. The manifest order determines concatenation order; the tool does not infer it from directory listings.

```sh
python tools/reassemble_archives.py --manifest release-manifests/raw-data-v1.json --archive gazebasevr.zip --parts-dir downloads --output raw/gazebasevr.zip
python tools/reassemble_archives.py --manifest release-manifests/raw-data-v1.json --archive GazeBase_v2_0.zip --parts-dir downloads --output raw/GazeBase_v2_0.zip
```

The tool uses bounded streaming I/O, checks each part's bytes and SHA-256, then checks the reconstructed archive's total bytes and upstream MD5. It never downloads files, executes extraction, runs an experiment, or trains a model. It refuses an existing output path. A failed reconstruction removes only the newly created incomplete output. Allow storage for both the downloaded parts and the reconstructed archive. A successful checksum match establishes byte identity with the listed upstream archive, not successful study reproduction.

## Analysis layout

Extraction and study execution are separate, optional operations. Before doing either, read `analysis/README_REPRO.md`. That guide describes the expected data tree and distinguishes recomputation from selected frozen derived inputs from rebuilding historical processing outputs. Use a separate working copy for any new study run. The original datasets may contain material beyond the task subsets used here; preserve the upstream license and attribution alongside reused material.
