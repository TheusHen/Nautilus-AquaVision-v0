# Data Audit

## Evidence available in this release

- The training notebook reports 5,282 train, 667 validation, and 654 test images.
- Generated filenames encode the source directory as a prefix.
- The retained full test inference log embedded in the original notebook covers all 654 test images:
  - 373 filenames start with `floating_marine_litter_`.
  - 281 filenames start with `iwhr_floater_`.
  - no test filename starts with `lars_`, `marida_`, `mastr1325_`, or `risid_`.
- The supplied model contains a single class named `floating_object`.

## Missing evidence

The merged dataset's `manifest.csv` and `REPORT.md` were not supplied for this packaging task. Consequently, exact accepted sample counts per source across train and validation cannot be independently verified here.

## Release decision

Only the two sources evidenced in the complete test log are credited as training datasets in this release. The other curation directories are documented as present but not claimed as contributors.

## Recommended v0.2 audit fields

Publish a manifest with no absolute local paths and these columns:

- transformed filename;
- split;
- source dataset identifier;
- source image hash;
- transformed image hash;
- object count;
- transformation version;
- duplicate-group identifier.

Do not publish original images, labels, personal paths, or source archive credentials in the manifest.
