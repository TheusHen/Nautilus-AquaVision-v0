# Dataset Sources, Licenses, and Attribution

This repository does **not** redistribute the original dataset images or annotations.

## Official dataset release

- Kaggle notebook: https://www.kaggle.com/code/theushen/aquavision-v0-1-0-alpha
- Official dataset: https://doi.org/10.34740/kaggle/dsv/17844306

Dataset citation:

```bibtex
@misc{matheus_henrique_2026,
  title={Nautilus 'AquaVision v0' Floating Object Detection},
  url={https://www.kaggle.com/dsv/17844306},
  DOI={10.34740/KAGGLE/DSV/17844306},
  publisher={Kaggle},
  author={Matheus Henrique},
  year={2026}
}
```

## 1. floating marine litter dataset

- **Creators:** Guido Lazzerini, Fausto Ferreira, Alessandro Ridolfi
- **Year:** 2025
- **Publisher:** SEANOE
- **DOI:** https://doi.org/10.17882/106148
- **Source:** https://www.seanoe.org/data/00950/106148/
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **License text:** https://creativecommons.org/licenses/by/4.0/legalcode
- **Upstream description:** approximately 5,500 optical images and 14,000 bounding-box annotations.

**Changes made for AquaVision:** compatible annotations were converted to YOLO detection labels; source categories were collapsed into the single `floating_object` class; accepted images could be resized, converted to JPEG, deduplicated using perceptual dHash, and assigned to deterministic train/validation/test splits.

**Required attribution:** cite the creators, title, year, DOI, source, and CC BY 4.0 license, and indicate that the data were modified.

Suggested citation:

> Lazzerini, G.; Ferreira, F.; Ridolfi, A. (2025). *floating marine litter dataset*. SEANOE. https://doi.org/10.17882/106148. Licensed under CC BY 4.0. Modified for Nautilus AquaVision.

## 2. IWHR_AI_Lable_Floater_V1

- **Creators:** Guangchao Qiao, Mingxiang Yang, Hao Wang
- **Year:** 2024
- **Publisher:** Figshare
- **DOI:** https://doi.org/10.6084/m9.figshare.27376851
- **Source:** https://figshare.com/articles/dataset/27376851
- **License:** Apache License 2.0
- **License text:** https://www.apache.org/licenses/LICENSE-2.0
- **Upstream description:** 3,000 annotated images for water-surface floating-object detection.

**Changes made for AquaVision:** compatible annotations were converted or normalized to YOLO detection labels; source categories were collapsed into the single `floating_object` class; accepted images could be resized, converted to JPEG, deduplicated using perceptual dHash, and assigned to deterministic train/validation/test splits.

Suggested citation:

> Qiao, G.; Yang, M.; Wang, H. (2024). *IWHR_AI_Lable_Floater_V1: An annotated Dataset and Benchmark for Detecting Floating Debris in Inland Waters*. Figshare. https://doi.org/10.6084/m9.figshare.27376851. Licensed under Apache-2.0. Modified for Nautilus AquaVision.

## Directories that were present during curation

The working `sources/` tree also contained LaRS, MARIDA, MaSTr1325, and RiSID directories. They are **not listed as released training sources** because the provided release artifacts do not show accepted samples from those prefixes. In particular, the complete 654-image test prediction log contains 373 `floating_marine_litter_*` files and 281 `iwhr_floater_*` files, with no other source prefix.

The original `manifest.csv` was not supplied with the release files, so this conclusion cannot replace a manifest-level audit. Recovering and publishing a privacy-safe manifest containing only source name, split, hash, and transformed filename is recommended for the next release.

## License interaction

- Dataset licenses govern the original data and attribution.
- This repository's code and YOLO26-derived weights are under AGPL-3.0.
- CC BY 4.0 and Apache-2.0 do not add a non-commercial restriction to this release.
- Users remain responsible for complying with all applicable upstream terms and with AGPL-3.0 when redistributing or deploying the model.
