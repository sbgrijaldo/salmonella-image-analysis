# Qualitative image analysis of *Salmonella*–HCT116 widefield fluorescence images

Analysis code and derived results for the widefield fluorescence microscopy figure in
*[manuscript title]*, Grijaldo-Alvarez et al.

This repository contains the script that produces Figure 6 and Supplementary Figures S6
and S7 from the raw microscopy TIFFs, together with the derived tables and a processing
log. It does **not** contain the raw images (see [Data](#data)).

## What the script does

`make_qualitative_figure.py` reads the raw 8-bit TIFFs for the two treatment conditions
and produces:

| Output | Description |
|---|---|
| `Figure_qualitative_*.png` / `.tif` | Main figure: DAPI, phalloidin and merge for one representative field per treatment, with an enlarged view of the host-cell periphery |
| `Validation_*.png` | Per-treatment identification sheet: every near-edge candidate object beside unambiguous objects from the clean background of the same field |
| `Supplementary_gallery_*.png` | All fields for each treatment |
| `field_selection_table.csv` | Per-field metrics used to choose the representative fields |
| `membrane_adhesion_table.csv` | Per-field peri-membrane metrics |
| `processing_log.json` | Run parameters and SHA-256 checksums of every input file |

### Analysis steps

1. **Channel assignment** from filename convention (`*_b_*` = DAPI, `*_r_*` =
   Alexa Fluor 594–phalloidin). Composite files are read per plane with `tifffile`.
2. **Cell segmentation** from the phalloidin channel (Otsu threshold, morphological
   cleanup), giving host-cell boundaries and a Euclidean distance map from the cell edge.
3. **Background correction for display and detection**: white top-hat with a 2.4 µm
   circular structuring element, which removes the smooth out-of-focus nuclear halo
   without altering object intensities. Applied uniformly to the whole image.
4. **Candidate object detection** on the glare-corrected DAPI channel, by criteria fixed
   in advance and applied identically to both treatments:

   | Criterion | Value |
   |---|---|
   | Object area | 0.16–3.1 µm² |
   | Major axis length | 0.65–4.5 µm |
   | Convex-hull solidity | ≥ 0.80 |
   | Local signal-to-noise ratio | ≥ 5 |
   | Centre-minus-annulus intensity | ≥ 2 grey levels |
   | Proximity to cell edge ("near-edge") | ≤ 3.0 µm |

   The solidity and centre-minus-annulus tests reject hollow ring artefacts, which are
   the characteristic signature of out-of-focus debris in widefield epifluorescence.
5. **Field selection**, by a rule fixed before inspection: the field with the most
   near-edge candidates, ties broken by lower DAPI saturation. Both displayed fields are
   therefore clearest-example fields, not median fields; all fields are given in the
   supplementary galleries.
6. **Provenance**: SHA-256 checksums of all input files are recorded before and after the
   run and compared, so the log demonstrates the sources were not modified.

### Image integrity

Only linear brightness and contrast adjustment was applied, identically to every image of
a given channel (DAPI 20–255; phalloidin 15–110; enlarged-panel DAPI after top-hat,
2–28 grey levels). The white top-hat is a uniform whole-image morphological operation,
disclosed in the figure legend and Methods. No selective, non-linear or generative
adjustment was performed at any stage, and no generative model was used to reconstruct,
sharpen or alter any biological structure.

## Running it

Requires Python 3.9+ and:

```
pip install numpy scipy tifffile pillow matplotlib scikit-image
```

OpenCV is **not** required.

Edit the two paths at the top of the script:

```python
ROOT_FOLDER   = r"...\Salmonella"                 # folder containing 20251222\
OUTPUT_FOLDER = r"...\salmonella_figure_out"      # where outputs are written
```

then run it from an editor (Spyder: F5) or the command line:

```
python make_qualitative_figure.py
```

Command-line arguments are also accepted:

```
python make_qualitative_figure.py --root "...\Salmonella" --out figure_out
```

The script reads the source TIFFs read-only and writes only into `OUTPUT_FOLDER`.

## Repository contents

```
make_qualitative_figure.py    analysis and figure-generation script
figure_legends.txt            legends for Figure 6, S6, S7
figures/                      generated figures
results/                      derived tables and the processing log
```

## Data

Raw microscopy TIFFs are not included here because of their size. They are available
[from the corresponding author on reasonable request / at <repository and accession>].
The SHA-256 checksums in `results/processing_log.json` identify the exact files used.

## Limitations

These images support qualitative visualisation only. Specifically:

- The Alexa Fluor 488 anti-*Salmonella* channel was not acquired in this imaging session,
  so bacteria are identified by DAPI signal and rod morphology alone, which is not
  specific. Detected objects are therefore reported as **candidates** throughout.
- Out-of-focus nuclear DAPI emission raises the local background at the host-cell
  periphery (median 110–179 grey levels, against approximately 40 in cell-free regions),
  leaving under 4% of the peri-membrane area evaluable. Association of bacteria with
  host-cell regions, adhesion and internalisation are **not** assessable from these
  images, and none is claimed.
- A phalloidin image is available for 10 of 10 fields in one condition but only 3 of 10 in
  the other, so field-level statistics are not meaningful.
- Two-dimensional widefield fluorescence without differential inside/outside staining
  cannot separate intracellular from extracellular bacteria. No internalisation claim is
  made from these images.

Adherence and invasion were quantified by CFU assay and are reported in the manuscript.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`).

## Licence

MIT — see `LICENSE`.
