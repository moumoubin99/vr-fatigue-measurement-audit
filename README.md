# Headset gaze and visual fatigue: analysis code and derived results

This repository contains selected code and frozen derived results for a secondary analysis of public GazeBaseVR and GazeBase recordings. The study is titled *Evaluating headset gaze measures of visual fatigue: a benchmark audit and paired-session analysis*. Repository preparation does not imply publication, acceptance, or a new reproduction of the experiments.

The analysis separates baseline-label prediction, associations between paired rating and gaze changes, and session effects with operational agreement bounds. It does not supply a validated individual fatigue detector. Dataset creators collected the original recordings; this project performed a secondary analysis.

## Contents

- `analysis/`: selected implementations, reconstructed labels, participant-disjoint splits, frozen derived inputs, result tables, saved predictions, and historical comparison records.
- `analysis/README_REPRO.md`: result map, dependency specification, execution layout, and limits of each reproducibility route.
- `analysis/MANIFEST.json`: relative file paths, byte counts, and SHA-256 digests for the analysis package.
- `DATA_SOURCES.md`: original dataset citations, version DOIs, licenses, and archive checksums.
- `DOWNLOAD_AND_REASSEMBLE.md`: original-source download route and published split-release route.
- `release-manifests/raw-data-v1.json`: original-archive identities and the ordered part records used for reconstruction.
- `tools/reassemble_archives.py`: standard-library-only, streaming reconstruction and checksum verification. It performs no extraction, data analysis, or model training.

The historical X4 reading branch remains in frozen outputs and wrappers to preserve their recorded dependencies. It is additional historical material, not a contribution of the current manuscript. Original filenames are preserved, including earlier result-table numbering.

## Inspect without running the study

Within `analysis/`, run the optional package-reading check in an environment with NumPy, pandas, and a Parquet engine:

```sh
python code/verify_package.py
```

This checks manifest digests and file readability/syntax. It does not rerun study analyses. Recomputing from the retained frozen derived inputs is distinct from rebuilding all historical extraction outputs from original recordings. See `analysis/README_REPRO.md` before executing analysis wrappers in a separate working copy. Those wrappers can write outputs and run substantial CPU/GPU workloads.

## Original recordings

Raw archives are excluded from Git history. The official Figshare releases are the authoritative source. The [raw-data-v1 Release](https://github.com/moumoubin99/vr-fatigue-measurement-audit/releases/tag/raw-data-v1) provides 18 binary parts of the unchanged original archives under their CC BY 4.0 terms, plus an attached `raw-data-v1.manifest.json`. Both complete archives passed local size and original MD5 checks before splitting. Every part matched the GitHub API size and server SHA-256 digest. Use the ordered manifest and reconstruction guide; these integrity checks do not imply a study rerun. The release tag records the code snapshot at release creation. The attached binary manifest indexes the data assets; later main-branch documentation can change without moving that tag.

The ordered local part streams were independently concatenated into hashes without extraction: both total byte counts and original upstream MD5 values matched. No second full archive copy or study rerun was needed for this reconstruction-integrity check.

## Attribution and code rights

Please cite this code/derived-results release using `CITATION.cff`, and cite the original datasets separately as listed in `DATA_SOURCES.md`. The repository citation is not a claim that the associated manuscript has a DOI or has been published.

Original dataset rights remain with their creators under CC BY 4.0. Per-file and upstream notices remain applicable. This repository does not impose a blanket MIT or other software license on code for which no such license has been supplied. See `analysis/THIRD_PARTY_NOTICES.md` for the rights boundary.
